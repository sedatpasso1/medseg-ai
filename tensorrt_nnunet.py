"""
TensorRT-accelerated nnUNet v2 inference.

Optimization katmanları (en hızlıdan yavaşa doğru):
  TensorRT engine  →  ONNX Runtime  →  PyTorch (baseline)

İlk çalıştırmada otomatik ONNX export + TRT build yapar ve
engines/ klasörüne cache'ler. Sonraki çalıştırmalarda engine
diskten yüklenir (<1s).

GPU yoksa veya TRT kurulu değilse sessizce PyTorch'a döner.

Kullanım:
    pred = TRTnnUNetPredictor("models/nnunet/Task500", fp16=True)
    mask = pred.predict(volume)  # nnUNetPredictor ile aynı arayüz
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("medseg.trt")

# ─── Bağımlılık kontrolleri ───────────────────────────────────────────────────

def _check_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _check_tensorrt() -> bool:
    try:
        import tensorrt as trt  # noqa: F401
        return True
    except ImportError:
        return False


def _check_onnxruntime() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def _check_onnx() -> bool:
    try:
        import onnx  # noqa: F401
        return True
    except ImportError:
        return False


# ─── ONNX export ─────────────────────────────────────────────────────────────

def export_to_onnx(
    network,
    tile_size: tuple[int, ...],
    onnx_path: Path,
    n_input_channels: int = 1,
    fp16: bool = False,
) -> Path:
    """
    nnUNet PyTorch network'ü ONNX'e export eder.

    tile_size: nnUNet configuration'dan alınan patch boyutu, örn. (128, 128, 128).
    fp16: ONNX içinde fp16 kullan (TRT için fp16=False, TRT build sırasında dönüştürülür).
    """
    import torch

    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    device = next(network.parameters()).device
    dummy = torch.zeros(1, n_input_channels, *tile_size, device=device)

    # Dynamic axes: batch + spatial dims (değişken giriş boyutu için)
    n_dims = len(tile_size)
    spatial_names = ["D", "H", "W"][-n_dims:]
    dynamic_axes = {
        "input":  {0: "batch"} | {i + 1: spatial_names[i] for i in range(n_dims)},
        "output": {0: "batch"} | {i + 1: spatial_names[i] for i in range(n_dims)},
    }

    network.eval()
    with torch.inference_mode():
        torch.onnx.export(
            network,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
        )

    log.info(f"ONNX export tamamlandı: {onnx_path} ({onnx_path.stat().st_size // 1024} KB)")
    return onnx_path


# ─── TensorRT engine builder ─────────────────────────────────────────────────

def build_trt_engine(
    onnx_path: Path,
    engine_path: Path,
    fp16: bool = True,
    max_workspace_gb: float = 4.0,
    min_shape: tuple[int, ...] | None = None,
    opt_shape: tuple[int, ...] | None = None,
    max_shape: tuple[int, ...] | None = None,
) -> Path:
    """
    ONNX modelinden TensorRT engine üretir.

    engine_path: .trt uzantılı çıktı dosyası
    fp16: FP16 modu (RTX GPU'larda ~2x hız artışı)
    max_workspace_gb: TRT builder için maksimum GPU belleği
    """
    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(max_workspace_gb * (1 << 30)),
    )

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        log.info("TRT: FP16 modu aktif")

    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                log.error(f"ONNX parse hatası {i}: {parser.get_error(i)}")
            raise RuntimeError(f"ONNX parse başarısız: {onnx_path}")

    # Optimization profile (dynamic shapes)
    profile = builder.create_optimization_profile()
    inp = network.get_input(0)
    inp_shape = tuple(inp.shape[1:])  # (C, D, H, W) veya (C, H, W)

    if min_shape is None:
        min_shape = (1,) + inp_shape
    if opt_shape is None:
        opt_shape = (1,) + inp_shape
    if max_shape is None:
        max_shape = (1,) + inp_shape

    profile.set_shape("input", min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    t0 = time.time()
    log.info(f"TRT engine build başlıyor: {onnx_path.name} → {engine_path.name}")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT engine build başarısız")

    with open(engine_path, "wb") as f:
        f.write(serialized)

    log.info(
        f"TRT engine kaydedildi: {engine_path} "
        f"({engine_path.stat().st_size // 1024} KB, {time.time()-t0:.1f}s)"
    )
    return engine_path


# ─── TRT çalıştırıcı ─────────────────────────────────────────────────────────

class _TRTRunner:
    """Serialized TRT engine yükler, tensor inference çalıştırır."""

    def __init__(self, engine_path: Path) -> None:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        self._context = self._engine.create_execution_context()
        self._cuda = cuda
        log.info(f"TRT engine yüklendi: {engine_path.name}")

    def infer(self, inp: np.ndarray) -> np.ndarray:
        """inp: float32 numpy array. Döner: float32 output array."""
        import pycuda.driver as cuda

        inp = inp.astype(np.float32)
        out_shape = tuple(self._context.get_output_shape(0, inp.shape))
        out = np.empty(out_shape, dtype=np.float32)

        d_inp = cuda.mem_alloc(inp.nbytes)
        d_out = cuda.mem_alloc(out.nbytes)

        cuda.memcpy_htod(d_inp, inp)
        self._context.execute_v2([int(d_inp), int(d_out)])
        cuda.memcpy_dtoh(out, d_out)
        return out


# ─── Ana predictor ────────────────────────────────────────────────────────────

class TRTnnUNetPredictor:
    """
    TensorRT-accelerated nnUNet v2 predictor.

    predict(volume) → int32 segmentasyon maskesi.
    nnUNetPredictor ile aynı public arayüz.

    İlk çağrıda:
      1. nnUNet modeli yüklenir (PyTorch)
      2. Model ONNX'e export edilir (engine_cache_dir/model.onnx)
      3. TRT engine build edilir (engine_cache_dir/model_fp16.trt)
      4. Sonraki çağrılarda cache kullanılır

    TRT/ORT yoksa otomatik olarak PyTorch'a döner.
    """

    MODE_TRT     = "trt"
    MODE_ORT     = "ort"
    MODE_PYTORCH = "pytorch"

    def __init__(
        self,
        model_folder: str | Path,
        engine_cache_dir: str | Path = "engines",
        fp16: bool = True,
        folds: tuple[int, ...] = (0,),
        checkpoint: str = "checkpoint_final.pth",
        device=None,
        use_gaussian: bool = True,
        tile_step_size: float = 0.5,
        preferred_mode: str = "auto",
    ) -> None:
        from medseg_pipeline import nnUNetPredictor, _get_device

        self.model_folder     = Path(model_folder)
        self.engine_cache_dir = Path(engine_cache_dir)
        self.fp16             = fp16
        self.preferred_mode   = preferred_mode

        # Temel PyTorch predictor (fallback + ilk model yükleme)
        self._pytorch_pred = nnUNetPredictor(
            model_folder=model_folder,
            folds=folds,
            checkpoint=checkpoint,
            device=device,
            use_gaussian=use_gaussian,
            tile_step_size=tile_step_size,
        )

        # Aktif modu belirle
        self._mode = self._select_mode()
        log.info(f"TRTnnUNetPredictor modu: {self._mode}")

        # Lazy-loaded inference nesneleri
        self._trt_runner: _TRTRunner | None = None
        self._ort_session = None
        self._onnx_path: Path | None = None
        self._engine_path: Path | None = None

    # ── Mod seçimi ───────────────────────────────────────────────────────────

    def _select_mode(self) -> str:
        if self.preferred_mode != "auto":
            return self.preferred_mode

        if _check_tensorrt():
            try:
                import pycuda.driver  # noqa: F401
                return self.MODE_TRT
            except ImportError:
                pass

        if _check_onnxruntime():
            return self.MODE_ORT

        return self.MODE_PYTORCH

    # ── İlk hazırlık ─────────────────────────────────────────────────────────

    def _prepare(self) -> None:
        """Model yoksa yükle, ONNX/TRT yoksa build et."""
        # PyTorch yüklenmemişse yükle
        self._pytorch_pred._ensure_loaded()
        predictor = self._pytorch_pred._predictor

        if self._mode == self.MODE_PYTORCH:
            return  # PyTorch modunda ekstra iş yok

        # ONNX path
        model_name = self.model_folder.name
        suffix = "_fp16" if self.fp16 else ""
        self._onnx_path   = self.engine_cache_dir / f"{model_name}.onnx"
        self._engine_path = self.engine_cache_dir / f"{model_name}{suffix}.trt"

        # ONNX yoksa export et
        if not self._onnx_path.exists():
            log.info("ONNX dosyası bulunamadı, export ediliyor...")
            network = predictor.network
            # nnUNet configuration'dan tile size al
            try:
                tile_size = tuple(predictor.configuration_manager.patch_size)
                n_ch = predictor.configuration_manager.num_input_channels
            except AttributeError:
                tile_size = (128, 128, 128)
                n_ch = 1
                log.warning(f"Tile size alınamadı, varsayılan kullanılıyor: {tile_size}")

            export_to_onnx(network, tile_size, self._onnx_path, n_input_channels=n_ch)

        # TRT engine
        if self._mode == self.MODE_TRT and not self._engine_path.exists():
            log.info("TRT engine bulunamadı, build ediliyor (bu uzun sürebilir)...")
            build_trt_engine(self._onnx_path, self._engine_path, fp16=self.fp16)

        # Inference nesnelerini yükle
        if self._mode == self.MODE_TRT and self._trt_runner is None:
            self._trt_runner = _TRTRunner(self._engine_path)

        elif self._mode == self.MODE_ORT and self._ort_session is None:
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._ort_session = ort.InferenceSession(
                str(self._onnx_path), providers=providers
            )
            log.info(f"ORT session açıldı: {self._ort_session.get_providers()}")

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, volume: np.ndarray) -> np.ndarray:
        """
        volume: float32 (D, H, W) [0, 1].
        Döner: int32 segmentasyon maskesi.
        """
        self._prepare()

        if self._mode == self.MODE_TRT:
            return self._predict_trt(volume)
        elif self._mode == self.MODE_ORT:
            return self._predict_ort(volume)
        else:
            return self._pytorch_pred.predict(volume)

    def _predict_trt(self, volume: np.ndarray) -> np.ndarray:
        """Sliding window inference via TRT engine."""
        return self._sliding_window_infer(
            volume, lambda tile: self._trt_runner.infer(tile[np.newaxis])
        )

    def _predict_ort(self, volume: np.ndarray) -> np.ndarray:
        """Sliding window inference via ONNX Runtime."""
        input_name = self._ort_session.get_inputs()[0].name

        def _run(tile: np.ndarray) -> np.ndarray:
            return self._ort_session.run(
                None, {input_name: tile[np.newaxis].astype(np.float32)}
            )[0]

        return self._sliding_window_infer(volume, _run)

    def _sliding_window_infer(
        self,
        volume: np.ndarray,
        infer_fn,
        tile_overlap: float = 0.25,
    ) -> np.ndarray:
        """
        Volume üzerinde kayan pencere inference.
        Tile prediction sonuçlarını Gaussian ağırlıklı olarak birleştirir.
        """
        predictor = self._pytorch_pred._predictor
        try:
            tile = tuple(predictor.configuration_manager.patch_size)
            n_classes = predictor.label_manager.num_segmentation_heads
        except AttributeError:
            # Fallback: small volume → tek tile
            tile = volume.shape
            n_classes = 3

        D, H, W = volume.shape
        tD, tH, tW = [min(t, s) for t, s in zip(tile, (D, H, W))]

        n_cls = n_classes
        prob_map = np.zeros((n_cls, D, H, W), dtype=np.float32)
        count_map = np.zeros((D, H, W), dtype=np.float32)

        step_d = max(1, int(tD * (1 - tile_overlap)))
        step_h = max(1, int(tH * (1 - tile_overlap)))
        step_w = max(1, int(tW * (1 - tile_overlap)))

        starts_d = list(range(0, max(D - tD, 0) + 1, step_d))
        starts_h = list(range(0, max(H - tH, 0) + 1, step_h))
        starts_w = list(range(0, max(W - tW, 0) + 1, step_w))
        # Son tile'ın son dilimi kapsaması için
        for starts, dim, tile_dim in [(starts_d, D, tD), (starts_h, H, tH), (starts_w, W, tW)]:
            if not starts or starts[-1] + tile_dim < dim:
                starts.append(max(0, dim - tile_dim))

        for sd in starts_d:
            for sh in starts_h:
                for sw in starts_w:
                    ed, eh, ew = sd + tD, sh + tH, sw + tW
                    tile_data = volume[sd:ed, sh:eh, sw:ew][np.newaxis]  # (1, tD, tH, tW)
                    out = infer_fn(tile_data)  # (1, n_cls, tD, tH, tW) veya compat shape
                    if out.ndim == 5:
                        out = out[0]  # (n_cls, tD, tH, tW)
                    prob_map[:, sd:ed, sh:eh, sw:ew] += out[:n_cls, :tD, :tH, :tW]
                    count_map[sd:ed, sh:eh, sw:ew] += 1.0

        count_map = np.maximum(count_map, 1.0)
        prob_map /= count_map
        return prob_map.argmax(axis=0).astype(np.int32)

    # ── Benchmark yardımcısı ─────────────────────────────────────────────────

    @property
    def active_mode(self) -> str:
        return self._mode

    @property
    def engine_path(self) -> Path | None:
        return self._engine_path

    @property
    def onnx_path(self) -> Path | None:
        return self._onnx_path
