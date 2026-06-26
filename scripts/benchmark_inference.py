#!/usr/bin/env python3
"""
MedSeg-AI inference backend benchmark.

nnUNet PyTorch vs ONNX Runtime vs TensorRT inference sürelerini karşılaştırır.
Gerçek model yoksa mock predictor kullanır (mimari doğrulama için).

Kullanım:
    python scripts/benchmark_inference.py
    python scripts/benchmark_inference.py --model_folder models/nnunet/Task500
    python scripts/benchmark_inference.py --n_warmup 2 --n_runs 10 --fp16

Çıktı: benchmark_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from medseg_pipeline import nnUNetPredictor, _get_device


# ─── Sentetik veri ───────────────────────────────────────────────────────────

def make_synthetic_volume(shape: tuple[int, int, int] = (64, 64, 64),
                          seed: int = 42) -> np.ndarray:
    """Sahte CT hacmi: [0,1] float32, shape (D, H, W)."""
    rng = np.random.default_rng(seed)
    vol = rng.uniform(0.0, 1.0, shape).astype(np.float32)
    # Basit organ-benzeri blob ekle
    cx, cy, cz = [s // 2 for s in shape]
    r = min(shape) // 4
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    blob = ((zz - cx) ** 2 + (yy - cy) ** 2 + (xx - cz) ** 2) < r ** 2
    vol[blob] = 0.8
    return vol


# ─── Mock predictor (model olmadan test için) ─────────────────────────────────

class _MockPredictor:
    """nnUNetPredictor mock'u — gerçek model yüklemez, sabit maske döner."""

    def __init__(self, volume_shape: tuple[int, int, int], n_classes: int = 3) -> None:
        self.volume_shape = volume_shape
        self.n_classes = n_classes

    def predict(self, volume: np.ndarray) -> np.ndarray:
        D, H, W = volume.shape
        mask = np.zeros((D, H, W), dtype=np.int32)
        mask[D // 4:3 * D // 4, H // 4:3 * H // 4, W // 4:3 * W // 4] = 1
        time.sleep(0.05)  # gerçek inference'ı simüle et
        return mask


# ─── Backend çalıştırıcılar ───────────────────────────────────────────────────

def _run_pytorch(predictor, volume: np.ndarray, n_warmup: int, n_runs: int) -> dict:
    """PyTorch baseline benchmark."""
    for _ in range(n_warmup):
        predictor.predict(volume)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        predictor.predict(volume)
        times.append(time.perf_counter() - t0)

    return _stats(times, "pytorch")


def _run_ort(onnx_path: Path, volume: np.ndarray,
             n_warmup: int, n_runs: int) -> dict:
    """ONNX Runtime benchmark."""
    try:
        import onnxruntime as ort
    except ImportError:
        return {"backend": "ort", "available": False}

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    inp_name = sess.get_inputs()[0].name

    # Giriş boyutunu al (model tile boyutunda çalışır)
    inp_shape = tuple(sess.get_inputs()[0].shape)
    # Dynamic axes → (1, 1, D, H, W) veya benzer; volume reshape
    D, H, W = volume.shape
    inp_data = volume[np.newaxis, np.newaxis].astype(np.float32)  # (1, 1, D, H, W)

    for _ in range(n_warmup):
        sess.run(None, {inp_name: inp_data})

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {inp_name: inp_data})
        times.append(time.perf_counter() - t0)

    result = _stats(times, "ort")
    result["providers"] = sess.get_providers()
    return result


def _run_trt(engine_path: Path, volume: np.ndarray,
             n_warmup: int, n_runs: int) -> dict:
    """TensorRT benchmark."""
    try:
        from tensorrt_nnunet import _TRTRunner
        runner = _TRTRunner(engine_path)
    except Exception as e:
        return {"backend": "trt", "available": False, "error": str(e)}

    inp = volume[np.newaxis, np.newaxis].astype(np.float32)

    for _ in range(n_warmup):
        runner.infer(inp)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        runner.infer(inp)
        times.append(time.perf_counter() - t0)

    return _stats(times, "trt")


def _stats(times: list[float], backend: str) -> dict:
    t = np.array(times) * 1000  # → ms
    return {
        "backend":      backend,
        "available":    True,
        "n_runs":       len(times),
        "mean_ms":      round(float(t.mean()), 2),
        "median_ms":    round(float(np.median(t)), 2),
        "std_ms":       round(float(t.std()), 2),
        "min_ms":       round(float(t.min()), 2),
        "max_ms":       round(float(t.max()), 2),
        "p95_ms":       round(float(np.percentile(t, 95)), 2),
        "throughput_fps": round(1000.0 / float(t.mean()), 2),
    }


# ─── Ana ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MedSeg inference benchmark")
    parser.add_argument("--model_folder", default=None,
                        help="nnUNet model klasörü (yoksa mock kullanılır)")
    parser.add_argument("--volume_shape", nargs=3, type=int, default=[64, 64, 64],
                        metavar=("D", "H", "W"))
    parser.add_argument("--n_warmup", type=int, default=2)
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--engine_dir", default="engines/")
    parser.add_argument("--output", default="benchmark_results.json")
    args = parser.parse_args()

    shape = tuple(args.volume_shape)
    volume = make_synthetic_volume(shape)
    print(f"\nBenchmark: volume={shape}, fp16={args.fp16}, "
          f"warmup={args.n_warmup}, runs={args.n_runs}")

    results = {"volume_shape": list(shape), "fp16": args.fp16, "backends": []}

    # ── PyTorch backend ──────────────────────────────────────────────
    if args.model_folder:
        pytorch_pred = nnUNetPredictor(args.model_folder)
    else:
        print("Model klasörü belirtilmedi — mock predictor kullanılıyor.")
        pytorch_pred = _MockPredictor(shape)

    print("PyTorch baseline...")
    r_pytorch = _run_pytorch(pytorch_pred, volume, args.n_warmup, args.n_runs)
    results["backends"].append(r_pytorch)
    print(f"  mean={r_pytorch['mean_ms']}ms  p95={r_pytorch['p95_ms']}ms  "
          f"fps={r_pytorch['throughput_fps']}")

    # ── ONNX / TRT (sadece gerçek model varsa) ────────────────────────
    engine_dir = Path(args.engine_dir)
    if args.model_folder:
        model_name = Path(args.model_folder).name
        suffix = "_fp16" if args.fp16 else ""
        onnx_path   = engine_dir / f"{model_name}.onnx"
        engine_path = engine_dir / f"{model_name}{suffix}.trt"

        try:
            from tensorrt_nnunet import TRTnnUNetPredictor, export_to_onnx, build_trt_engine

            trt_pred = TRTnnUNetPredictor(
                args.model_folder,
                engine_cache_dir=args.engine_dir,
                fp16=args.fp16,
            )
            # ORT
            if onnx_path.exists():
                print("ONNX Runtime...")
                r_ort = _run_ort(onnx_path, volume, args.n_warmup, args.n_runs)
                results["backends"].append(r_ort)
                if r_ort.get("available"):
                    speedup = round(r_pytorch["mean_ms"] / r_ort["mean_ms"], 2)
                    print(f"  mean={r_ort['mean_ms']}ms  speedup={speedup}x")
            # TRT
            if engine_path.exists():
                print("TensorRT...")
                r_trt = _run_trt(engine_path, volume, args.n_warmup, args.n_runs)
                results["backends"].append(r_trt)
                if r_trt.get("available"):
                    speedup = round(r_pytorch["mean_ms"] / r_trt["mean_ms"], 2)
                    print(f"  mean={r_trt['mean_ms']}ms  speedup={speedup}x")
        except Exception as exc:
            print(f"TRT/ORT atlandı: {exc}")
    else:
        print("TRT/ORT benchmark: gerçek model klasörü gerekiyor (--model_folder).")

    # ─── Hız artışı özeti ──────────────────────────────────────────────
    baseline = next((r for r in results["backends"] if r["backend"] == "pytorch"), None)
    if baseline:
        for r in results["backends"]:
            if r.get("available") and r["backend"] != "pytorch":
                r["speedup_vs_pytorch"] = round(
                    baseline["mean_ms"] / r["mean_ms"], 2
                )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSonuçlar kaydedildi: {args.output}")


if __name__ == "__main__":
    main()
