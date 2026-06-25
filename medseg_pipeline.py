"""
MedSeg-AI: Tıbbi Görüntü Segmentasyon Pipeline
===============================================
nnUNet v2 + SAM2 hibrit pipeline — organ segmentasyonu ve maske rafine etme.

Kullanım:
    python medseg_pipeline.py --dicom_path /data/ct_series --output_dir /out
    python medseg_pipeline.py --dicom_path scan.dcm --output_dir /out --model_folder /models/nnunet

Referanslar:
    - MIC-DKFZ/nnUNet: https://github.com/MIC-DKFZ/nnUNet
    - nnSAM: Segment Anything Model + nnUNet (PMC10787050)
    - SAM2: https://github.com/facebookresearch/segment-anything-2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import contextlib
import numpy as np
import pydicom
import nibabel as nib

try:
    import torch as _torch
except ImportError:
    _torch = None  # type: ignore[assignment]

_nullcontext = contextlib.nullcontext

log = logging.getLogger("medseg")


def _get_device(device=None):
    """torch.device döner; torch kurulu değilse None."""
    if _torch is None:
        return None
    if device is not None:
        return device
    return _torch.device("cuda") if _torch.cuda.is_available() else _torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════════════
# 1. DICOM LOADER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DicomMeta:
    patient_id: str = ""
    study_date: str = ""
    modality: str = ""
    rows: int = 0
    cols: int = 0
    n_slices: int = 0
    pixel_spacing: list[float] = field(default_factory=list)
    slice_thickness: float = 1.0
    hu_min: float = 0.0
    hu_max: float = 0.0


class DICOMLoader:
    """
    DICOM serisi veya tek dosya yükler.
    HU (Hounsfield Unit) windowing uygular.
    """

    # Yaygın HU penceresi ön ayarları
    WINDOW_PRESETS: dict[str, tuple[float, float]] = {
        "soft_tissue": (400.0,  40.0),   # (genişlik, merkez)
        "lung":        (1500.0, -600.0),
        "bone":        (1800.0, 400.0),
        "brain":       (80.0,   40.0),
        "abdomen":     (350.0,  40.0),
    }

    def __init__(self, window: str = "soft_tissue") -> None:
        if window not in self.WINDOW_PRESETS:
            raise ValueError(f"Geçersiz pencere: {window}. Seçenekler: {list(self.WINDOW_PRESETS)}")
        self.window = window
        self.meta: Optional[DicomMeta] = None

    def load(self, path: str | Path) -> np.ndarray:
        """
        DICOM dosyası veya klasörü yükle.
        Döner: float32 numpy array, shape (D, H, W), HU windowing uygulanmış [0, 1].
        """
        path = Path(path)
        if path.is_dir():
            volume, self.meta = self._load_series(path)
        else:
            volume, self.meta = self._load_single(path)

        log.info(
            f"DICOM yüklendi: {self.meta.modality} | "
            f"{self.meta.n_slices}x{self.meta.rows}x{self.meta.cols} | "
            f"HU [{self.meta.hu_min:.0f}, {self.meta.hu_max:.0f}]"
        )
        return self._apply_windowing(volume)

    def _load_series(self, folder: Path) -> tuple[np.ndarray, DicomMeta]:
        """Klasördeki tüm DICOM dosyalarını sıraya göre yükler."""
        dcm_files = sorted(
            [f for f in folder.iterdir() if f.suffix.lower() in (".dcm", ".dicom", "")
             and f.is_file()],
            key=lambda f: self._get_instance_number(f),
        )
        if not dcm_files:
            raise FileNotFoundError(f"DICOM dosyası bulunamadı: {folder}")

        slices = [pydicom.dcmread(str(f), force=True) for f in dcm_files]
        meta = self._extract_meta(slices[0], n_slices=len(slices))

        pixel_arrays = []
        for ds in slices:
            arr = self._to_hu(ds)
            pixel_arrays.append(arr)

        volume = np.stack(pixel_arrays, axis=0).astype(np.float32)
        meta.hu_min = float(volume.min())
        meta.hu_max = float(volume.max())
        return volume, meta

    def _load_single(self, filepath: Path) -> tuple[np.ndarray, DicomMeta]:
        """Tek DICOM dosyası yükler."""
        ds = pydicom.dcmread(str(filepath), force=True)
        meta = self._extract_meta(ds, n_slices=1)
        arr = self._to_hu(ds).astype(np.float32)
        volume = arr[np.newaxis]  # (1, H, W)
        meta.hu_min = float(volume.min())
        meta.hu_max = float(volume.max())
        return volume, meta

    @staticmethod
    def _get_instance_number(filepath: Path) -> int:
        try:
            ds = pydicom.dcmread(str(filepath), stop_before_pixels=True, force=True)
            return int(getattr(ds, "InstanceNumber", 0))
        except Exception:
            return 0

    @staticmethod
    def _to_hu(ds: pydicom.Dataset) -> np.ndarray:
        """Ham piksel → Hounsfield Unit dönüşümü."""
        pixels = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", -1024.0))
        return pixels * slope + intercept

    @staticmethod
    def _extract_meta(ds: pydicom.Dataset, n_slices: int) -> DicomMeta:
        spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
        return DicomMeta(
            patient_id=str(getattr(ds, "PatientID", "")),
            study_date=str(getattr(ds, "StudyDate", "")),
            modality=str(getattr(ds, "Modality", "CT")),
            rows=int(getattr(ds, "Rows", 0)),
            cols=int(getattr(ds, "Columns", 0)),
            n_slices=n_slices,
            pixel_spacing=[float(spacing[0]), float(spacing[1])],
            slice_thickness=float(getattr(ds, "SliceThickness", 1.0)),
        )

    def _apply_windowing(self, volume: np.ndarray) -> np.ndarray:
        """HU pencerelemesi → [0, 1] normalize."""
        width, center = self.WINDOW_PRESETS[self.window]
        hu_min = center - width / 2.0
        hu_max = center + width / 2.0
        windowed = np.clip(volume, hu_min, hu_max)
        return (windowed - hu_min) / (hu_max - hu_min)


# ══════════════════════════════════════════════════════════════════════════════
# 2. nnUNet PREDICTOR WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

class nnUNetPredictor:
    """
    nnUNet v2 organ segmentasyon sarmalayıcısı.

    model_folder: eğitilmiş nnUNet model klasörü
        (nnUNetTrainer__nnUNetPlans__3d_fullres formatında)
    """

    def __init__(
        self,
        model_folder: str | Path,
        folds: tuple[int, ...] = (0,),
        checkpoint: str = "checkpoint_final.pth",
        device: Optional[torch.device] = None,
        use_gaussian: bool = True,
        tile_step_size: float = 0.5,
    ) -> None:
        self.model_folder = Path(model_folder)
        self.folds = folds
        self.checkpoint = checkpoint
        self.device = _get_device(device)
        self.use_gaussian = use_gaussian
        self.tile_step_size = tile_step_size
        self._predictor = None

    def _ensure_loaded(self) -> None:
        if self._predictor is not None:
            return
        try:
            from nnunetv2.inference.predict_from_raw_data import (
                nnUNetPredictor as _NNUNetPredictor,
            )
        except ImportError as e:
            raise ImportError(
                "nnunetv2 kurulu değil: pip install nnunetv2"
            ) from e

        pred = _NNUNetPredictor(
            tile_step_size=self.tile_step_size,
            use_gaussian=self.use_gaussian,
            use_mirroring=True,
            perform_everything_on_device=(self.device is not None and self.device.type == "cuda"),
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        pred.initialize_from_trained_model_folder(
            str(self.model_folder),
            use_folds=self.folds,
            checkpoint_name=self.checkpoint,
        )
        self._predictor = pred
        log.info(f"nnUNet yüklendi: {self.model_folder.name} | device={self.device}")

    def predict(self, volume: np.ndarray) -> np.ndarray:
        """
        volume: float32, shape (D, H, W), değerler [0, 1].
        Döner: int32 segmentasyon maskesi, shape (D, H, W).
        """
        self._ensure_loaded()

        # nnUNet (D, H, W) → (C, D, H, W) bekler (C=1 için CT)
        img_4d = volume[np.newaxis].astype(np.float32)

        # properties dict — spacing bilgisi gerekiyor
        props = {
            "sitk_stuff": None,
            "spacing": [1.0, 1.0, 1.0],  # isovolümetrik varsayım
        }

        log.info(f"nnUNet inference: {volume.shape}")
        t0 = time.time()
        seg = self._predictor.predict_single_npy_array(img_4d, props,
                                                        None, None, False)
        log.info(f"nnUNet tamamlandı: {time.time() - t0:.1f}s | "
                 f"sınıflar={np.unique(seg).tolist()}")
        return seg.astype(np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# 3. SAM2 REFINER
# ══════════════════════════════════════════════════════════════════════════════

class SAM2Refiner:
    """
    Segment Anything Model 2 ile nnUNet maskelerini dilim dilim rafine eder.
    Her dilim için nnUNet maskesinden bounding-box prompt üretir.
    """

    def __init__(
        self,
        checkpoint: str | Path = "sam2_hiera_large.pt",
        model_cfg: str = "sam2_hiera_l.yaml",
        device: Optional[torch.device] = None,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.model_cfg = model_cfg
        self.device = _get_device(device)
        self._predictor = None

    def _ensure_loaded(self) -> None:
        if self._predictor is not None:
            return
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as e:
            raise ImportError(
                "sam2 kurulu değil: pip install git+https://github.com/facebookresearch/segment-anything-2"
            ) from e

        sam2_model = build_sam2(
            self.model_cfg,
            str(self.checkpoint),
            device=self.device,
        )
        self._predictor = SAM2ImagePredictor(sam2_model)
        log.info(f"SAM2 yüklendi: {self.checkpoint.name} | device={self.device}")

    def refine(self, volume: np.ndarray, coarse_mask: np.ndarray) -> np.ndarray:
        """
        volume: float32, shape (D, H, W), [0,1].
        coarse_mask: int32, shape (D, H, W) — nnUNet çıktısı.
        Döner: int32 rafine maske, shape (D, H, W).
        """
        self._ensure_loaded()
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        refined = np.zeros_like(coarse_mask)
        class_ids = [c for c in np.unique(coarse_mask) if c > 0]

        D = volume.shape[0]
        log.info(f"SAM2 refine: {D} dilim, {len(class_ids)} sınıf")

        ctx = _torch.inference_mode() if _torch is not None else _nullcontext()
        with ctx:
            for d in range(D):
                slice_img = self._to_rgb(volume[d])
                self._predictor.set_image(slice_img)

                for cls_id in class_ids:
                    cls_mask_2d = (coarse_mask[d] == cls_id)
                    if cls_mask_2d.sum() < 9:
                        # Çok küçük alan — rafine etme, orijinali koru
                        refined[d][cls_mask_2d] = cls_id
                        continue

                    bbox = self._mask_to_bbox(cls_mask_2d)
                    if bbox is None:
                        continue

                    masks, scores, _ = self._predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=bbox[np.newaxis],
                        multimask_output=False,
                    )
                    best_mask = masks[np.argmax(scores)]
                    refined[d][best_mask.astype(bool)] = cls_id

        log.info("SAM2 refine tamamlandı")
        return refined.astype(np.int32)

    @staticmethod
    def _to_rgb(slice_2d: np.ndarray) -> np.ndarray:
        """Gri tonlamalı dilim → uint8 RGB (SAM2 RGB bekler)."""
        uint8 = (np.clip(slice_2d, 0.0, 1.0) * 255).astype(np.uint8)
        return np.stack([uint8, uint8, uint8], axis=-1)

    @staticmethod
    def _mask_to_bbox(mask: np.ndarray) -> Optional[np.ndarray]:
        """Boolean maske → [x1, y1, x2, y2] bounding box."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return None
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        return np.array([x1, y1, x2, y2], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 4. ANA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SegmentationResult:
    dicom_path: str
    modality: str
    shape: list[int]
    pixel_spacing: list[float]
    slice_thickness: float
    class_ids: list[int]
    class_voxel_counts: dict[str, int]
    nnunet_inference_sec: float
    sam2_refine_sec: float
    total_sec: float
    output_nifti: str
    window: str
    device: str


class MedSegPipeline:
    """
    Tam tıbbi segmentasyon pipeline:
    DICOMLoader → nnUNetPredictor → SAM2Refiner

    Örnek:
        pipeline = MedSegPipeline(model_folder="/models/nnunet/Task500_MultiOrgan")
        result = pipeline.predict("/data/patient01/ct_series")
    """

    def __init__(
        self,
        model_folder: str | Path,
        sam2_checkpoint: str | Path = "sam2_hiera_large.pt",
        sam2_cfg: str = "sam2_hiera_l.yaml",
        window: str = "soft_tissue",
        device: Optional[torch.device] = None,
        use_sam2: bool = True,
        nnunet_folds: tuple[int, ...] = (0,),
    ) -> None:
        self.device = _get_device(device)
        self.use_sam2 = use_sam2

        self.loader   = DICOMLoader(window=window)
        self.nnunet   = nnUNetPredictor(
            model_folder=model_folder,
            folds=nnunet_folds,
            device=self.device,
        )
        self.sam2     = SAM2Refiner(
            checkpoint=sam2_checkpoint,
            model_cfg=sam2_cfg,
            device=self.device,
        ) if use_sam2 else None

        log.info(f"MedSegPipeline hazır | device={self.device} | SAM2={'açık' if use_sam2 else 'kapalı'}")

    def predict(self, dicom_path: str | Path) -> SegmentationResult:
        """
        DICOM yolu → segmentasyon sonucu.
        Maske bellekte tutulur; kaydetmek için save_nifti() kullan.
        """
        dicom_path = Path(dicom_path)
        t_total = time.time()

        # ── 1. DICOM yükle ──────────────────────────────────────────────
        log.info(f"DICOM yükleniyor: {dicom_path}")
        volume = self.loader.load(dicom_path)
        meta   = self.loader.meta

        # ── 2. nnUNet segmentasyonu ──────────────────────────────────────
        t_nn = time.time()
        coarse_mask = self.nnunet.predict(volume)
        nn_sec = time.time() - t_nn

        # ── 3. SAM2 refinement ───────────────────────────────────────────
        t_sam = time.time()
        if self.use_sam2 and self.sam2 is not None:
            final_mask = self.sam2.refine(volume, coarse_mask)
        else:
            final_mask = coarse_mask
        sam_sec = time.time() - t_sam

        total_sec = time.time() - t_total

        # ── Sonuç metadata ───────────────────────────────────────────────
        class_ids = [int(c) for c in np.unique(final_mask) if c > 0]
        voxel_counts = {
            str(c): int((final_mask == c).sum()) for c in class_ids
        }

        result = SegmentationResult(
            dicom_path=str(dicom_path),
            modality=meta.modality if meta else "CT",
            shape=list(final_mask.shape),
            pixel_spacing=meta.pixel_spacing if meta else [1.0, 1.0],
            slice_thickness=meta.slice_thickness if meta else 1.0,
            class_ids=class_ids,
            class_voxel_counts=voxel_counts,
            nnunet_inference_sec=round(nn_sec, 2),
            sam2_refine_sec=round(sam_sec, 2),
            total_sec=round(total_sec, 2),
            output_nifti="",
            window=self.loader.window,
            device=str(self.device),
        )

        # Maske referansını saklıyoruz (save_nifti için)
        self._last_mask = final_mask
        self._last_meta = meta
        log.info(
            f"Pipeline tamamlandı: {total_sec:.1f}s | "
            f"sınıflar={class_ids} | "
            f"nnUNet={nn_sec:.1f}s SAM2={sam_sec:.1f}s"
        )
        return result

    def save_nifti(self, result: SegmentationResult, output_dir: str | Path) -> Path:
        """Segmentasyon maskesini NIfTI formatında kaydeder."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = output_dir / f"segmentation_{ts}.nii.gz"

        mask = self._last_mask
        meta = self._last_meta

        # Affine: piksel boyutlarından oluştur
        dx = meta.pixel_spacing[0] if meta else 1.0
        dy = meta.pixel_spacing[1] if meta else 1.0
        dz = meta.slice_thickness  if meta else 1.0
        affine = np.diag([dx, dy, dz, 1.0])

        nifti_img = nib.Nifti1Image(mask.astype(np.int16), affine)
        nib.save(nifti_img, str(out_path))
        log.info(f"NIfTI kaydedildi: {out_path}")
        result.output_nifti = str(out_path)
        return out_path

    def save_json(self, result: SegmentationResult, output_dir: str | Path) -> Path:
        """JSON metadata kaydeder."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        json_path = output_dir / f"segmentation_{ts}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)
        log.info(f"JSON kaydedildi: {json_path}")
        return json_path


# ══════════════════════════════════════════════════════════════════════════════
# 5. CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MedSeg-AI: nnUNet + SAM2 tıbbi segmentasyon pipeline"
    )
    parser.add_argument(
        "--dicom_path", required=True,
        help="DICOM dosyası (.dcm) veya serinin bulunduğu klasör yolu",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="NIfTI maske ve JSON metadata çıktı klasörü",
    )
    parser.add_argument(
        "--model_folder", default=os.getenv("NNUNET_MODEL", "models/nnunet"),
        help="nnUNet eğitilmiş model klasörü (varsayılan: models/nnunet)",
    )
    parser.add_argument(
        "--sam2_checkpoint", default="sam2_hiera_large.pt",
        help="SAM2 checkpoint dosyası",
    )
    parser.add_argument(
        "--sam2_cfg", default="sam2_hiera_l.yaml",
        help="SAM2 model config",
    )
    parser.add_argument(
        "--window", default="soft_tissue",
        choices=list(DICOMLoader.WINDOW_PRESETS.keys()),
        help="HU pencere ön ayarı",
    )
    parser.add_argument(
        "--no_sam2", action="store_true",
        help="SAM2 rafine etme adımını atla (sadece nnUNet)",
    )
    parser.add_argument(
        "--folds", nargs="+", type=int, default=[0],
        help="nnUNet inference fold'ları",
    )
    parser.add_argument(
        "--device", default=None,
        help="cuda | cpu (varsayılan: otomatik)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Ayrıntılı log",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    device = _torch.device(args.device) if (args.device and _torch) else None

    pipeline = MedSegPipeline(
        model_folder=args.model_folder,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_cfg=args.sam2_cfg,
        window=args.window,
        device=device,
        use_sam2=not args.no_sam2,
        nnunet_folds=tuple(args.folds),
    )

    result = pipeline.predict(args.dicom_path)
    nifti_path = pipeline.save_nifti(result, args.output_dir)
    json_path  = pipeline.save_json(result, args.output_dir)

    print("\n" + "=" * 60)
    print("SEGMENTASYON TAMAMLANDI")
    print("=" * 60)
    print(f"NIfTI maske : {nifti_path}")
    print(f"JSON meta   : {json_path}")
    print(f"Süre        : {result.total_sec:.1f}s "
          f"(nnUNet {result.nnunet_inference_sec:.1f}s + "
          f"SAM2 {result.sam2_refine_sec:.1f}s)")
    print(f"Sınıflar    : {result.class_ids}")
    print(f"Voxel count : {result.class_voxel_counts}")
    print("=" * 60)
