"""
MedSeg-AI pipeline testleri — brief'te istenen test_pipeline.py.
Gerçek DICOM veya model dosyası gerektirmez; synthetic numpy array kullanır.

Kullanım:
    pytest tests/test_pipeline.py -v
"""
from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from medseg_pipeline import (
    DICOMLoader,
    DicomMeta,
    MedSegPipeline,
    SAM2Refiner,
    SegmentationResult,
    nnUNetPredictor,
)


# ── Sentetik veri yardımcıları ────────────────────────────────────────────────

def synthetic_volume(D: int = 8, H: int = 64, W: int = 64,
                     seed: int = 0) -> np.ndarray:
    """Rastgele float32 hacim — DICOM yüklemek yerine doğrudan kullanılır."""
    rng = np.random.default_rng(seed)
    return rng.random((D, H, W)).astype(np.float32)


def synthetic_mask(D: int = 8, H: int = 64, W: int = 64) -> np.ndarray:
    """Basit organlar içeren sentetik segmentasyon maskesi."""
    mask = np.zeros((D, H, W), dtype=np.int32)
    mask[:, 10:30, 10:30] = 1   # organ 1 — sol üst
    mask[:, 35:55, 35:55] = 2   # organ 2 — sağ alt
    return mask


# ── DICOMLoader testleri ──────────────────────────────────────────────────────

class TestDICOMLoaderUnit(unittest.TestCase):
    """DICOMLoader'ın windowing mantığını numpy dizisiyle doğrular."""

    def _loader_with_fake_meta(self) -> DICOMLoader:
        loader = DICOMLoader(window="soft_tissue")
        loader.meta = DicomMeta(
            patient_id="SYN001", study_date="20240101", modality="CT",
            rows=64, cols=64, n_slices=8,
            pixel_spacing=[1.0, 1.0], slice_thickness=2.0,
            hu_min=-1000.0, hu_max=500.0,
        )
        return loader

    def test_windowing_output_range(self):
        """Windowing sonucu [0, 1] aralığında olmalı."""
        loader = self._loader_with_fake_meta()
        volume = np.linspace(-1000, 1000, 64 * 64).reshape(1, 64, 64).astype(np.float32)
        result = loader._apply_windowing(volume)
        self.assertGreaterEqual(float(result.min()), 0.0 - 1e-6)
        self.assertLessEqual(float(result.max()), 1.0 + 1e-6)

    def test_windowing_dtype(self):
        """Windowing float32 döndürmeli."""
        loader = self._loader_with_fake_meta()
        vol = np.zeros((4, 32, 32), dtype=np.float32)
        result = loader._apply_windowing(vol)
        self.assertEqual(result.dtype, np.float32)

    def test_all_window_presets_valid(self):
        """Tüm pencere ön ayarları geçerli (width > 0, center mantıklı)."""
        for name, (width, center) in DICOMLoader.WINDOW_PRESETS.items():
            self.assertGreater(width, 0, f"{name}: genişlik <= 0")
            self.assertGreater(center, -2000, f"{name}: merkez çok düşük")

    def test_invalid_window_raises(self):
        with self.assertRaises(ValueError):
            DICOMLoader(window="xray_special")

    def test_mask_to_bbox_correctness(self):
        """SAM2Refiner._mask_to_bbox doğru koordinatlar döndürmeli."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:60, 30:80] = True
        bbox = SAM2Refiner._mask_to_bbox(mask)
        self.assertIsNotNone(bbox)
        x1, y1, x2, y2 = bbox
        self.assertEqual(int(x1), 30)
        self.assertEqual(int(y1), 20)
        self.assertEqual(int(x2), 79)
        self.assertEqual(int(y2), 59)

    def test_mask_to_bbox_empty_returns_none(self):
        mask = np.zeros((64, 64), dtype=bool)
        self.assertIsNone(SAM2Refiner._mask_to_bbox(mask))

    def test_to_rgb_shape_and_dtype(self):
        gray = np.random.rand(128, 128).astype(np.float32)
        rgb = SAM2Refiner._to_rgb(gray)
        self.assertEqual(rgb.shape, (128, 128, 3))
        self.assertEqual(rgb.dtype, np.uint8)


# ── MedSegPipeline testleri ───────────────────────────────────────────────────

class TestMedSegPipeline(unittest.TestCase):
    """MedSegPipeline'ı nnUNet ve SAM2 mock'layarak sentetik veriyle test eder."""

    def _build_pipeline(self, use_sam2: bool = False) -> MedSegPipeline:
        """Mock nnUNet ile pipeline oluşturur."""
        with patch.object(nnUNetPredictor, "_ensure_loaded"):
            pipeline = MedSegPipeline(
                model_folder="mock_model",
                use_sam2=use_sam2,
            )
        mask = synthetic_mask()
        pipeline.nnunet._ensure_loaded = lambda: None
        pipeline.nnunet._predictor = MagicMock()
        pipeline.nnunet._predictor.predict_single_npy_array.return_value = mask
        return pipeline

    def _predict_with_synthetic(self, pipeline: MedSegPipeline) -> SegmentationResult:
        """Loader'ı bypass edip sentetik hacimle direkt predict çalıştırır."""
        volume = synthetic_volume()
        # Loader'ı mock'la
        pipeline.loader.load = MagicMock(return_value=volume)
        pipeline.loader.meta = DicomMeta(
            patient_id="SYN", study_date="20240101", modality="CT",
            rows=64, cols=64, n_slices=8,
            pixel_spacing=[1.0, 1.0], slice_thickness=2.0,
            hu_min=0.0, hu_max=1.0,
        )
        return pipeline.predict("synthetic_path")

    # ── Temel tahmin ─────────────────────────────────────────────────

    def test_predict_returns_segmentation_result(self):
        pipeline = self._build_pipeline()
        result = self._predict_with_synthetic(pipeline)
        self.assertIsInstance(result, SegmentationResult)

    def test_predict_class_ids_correct(self):
        """Sentetik maskede 1 ve 2 sınıfları var."""
        pipeline = self._build_pipeline()
        result = self._predict_with_synthetic(pipeline)
        self.assertIn(1, result.class_ids)
        self.assertIn(2, result.class_ids)
        self.assertNotIn(0, result.class_ids)  # arka plan sınıf listesine eklenmemeli

    def test_predict_shape_matches_mask(self):
        pipeline = self._build_pipeline()
        result = self._predict_with_synthetic(pipeline)
        D, H, W = synthetic_mask().shape
        self.assertEqual(result.shape, [D, H, W])

    def test_predict_timing_fields(self):
        pipeline = self._build_pipeline()
        result = self._predict_with_synthetic(pipeline)
        self.assertGreaterEqual(result.total_sec, 0)
        self.assertGreaterEqual(result.nnunet_inference_sec, 0)
        self.assertGreaterEqual(result.sam2_refine_sec, 0)

    def test_predict_voxel_counts_positive(self):
        pipeline = self._build_pipeline()
        result = self._predict_with_synthetic(pipeline)
        for cls_id, count in result.class_voxel_counts.items():
            self.assertGreater(count, 0, f"Sınıf {cls_id} için voxel_count <= 0")

    # ── NIfTI kayıt ──────────────────────────────────────────────────

    def test_save_nifti_creates_file(self):
        pipeline = self._build_pipeline()
        result = self._predict_with_synthetic(pipeline)
        with tempfile.TemporaryDirectory() as tmp:
            out = pipeline.save_nifti(result, tmp)
            self.assertTrue(out.exists())
            self.assertTrue(str(out).endswith(".nii.gz"))

    def test_nifti_affine_uses_pixel_spacing(self):
        import nibabel as nib
        pipeline = self._build_pipeline()
        result = self._predict_with_synthetic(pipeline)
        with tempfile.TemporaryDirectory() as tmp:
            out = pipeline.save_nifti(result, tmp)
            img = nib.load(str(out))
            # meta.pixel_spacing = [1.0, 1.0], slice_thickness = 2.0
            self.assertAlmostEqual(float(img.affine[0, 0]), 1.0, places=3)
            self.assertAlmostEqual(float(img.affine[2, 2]), 2.0, places=3)

    def test_nifti_volume_shape_preserved(self):
        import nibabel as nib
        pipeline = self._build_pipeline()
        result = self._predict_with_synthetic(pipeline)
        with tempfile.TemporaryDirectory() as tmp:
            out = pipeline.save_nifti(result, tmp)
            data = nib.load(str(out)).get_fdata()
            self.assertEqual(list(data.shape), result.shape)

    # ── JSON kayıt ───────────────────────────────────────────────────

    def test_save_json_creates_file(self):
        pipeline = self._build_pipeline()
        result = self._predict_with_synthetic(pipeline)
        with tempfile.TemporaryDirectory() as tmp:
            json_path = pipeline.save_json(result, tmp)
            self.assertTrue(json_path.exists())

    def test_json_contains_required_keys(self):
        pipeline = self._build_pipeline()
        result = self._predict_with_synthetic(pipeline)
        with tempfile.TemporaryDirectory() as tmp:
            pipeline.save_nifti(result, tmp)
            json_path = pipeline.save_json(result, tmp)
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        required = [
            "dicom_path", "modality", "shape", "class_ids",
            "class_voxel_counts", "nnunet_inference_sec",
            "sam2_refine_sec", "total_sec", "output_nifti",
        ]
        for key in required:
            self.assertIn(key, data, f"JSON'da '{key}' eksik")

    def test_json_class_ids_match_result(self):
        pipeline = self._build_pipeline()
        result = self._predict_with_synthetic(pipeline)
        with tempfile.TemporaryDirectory() as tmp:
            pipeline.save_nifti(result, tmp)
            json_path = pipeline.save_json(result, tmp)
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        self.assertEqual(sorted(data["class_ids"]), sorted(result.class_ids))

    # ── SAM2 kapalı modu ─────────────────────────────────────────────

    def test_no_sam2_mode_runs(self):
        pipeline = self._build_pipeline(use_sam2=False)
        self.assertFalse(pipeline.use_sam2)
        self.assertIsNone(pipeline.sam2)
        result = self._predict_with_synthetic(pipeline)
        self.assertIsInstance(result, SegmentationResult)

    def test_sam2_refine_sec_zero_without_sam2(self):
        """SAM2 kapalıyken refine süresi ~0 olmalı."""
        pipeline = self._build_pipeline(use_sam2=False)
        result = self._predict_with_synthetic(pipeline)
        self.assertLess(result.sam2_refine_sec, 0.5)

    # ── Harita boyutu sınırı ─────────────────────────────────────────

    def test_mask_max_scans_limit(self):
        """_MAP_MAX_SCANS benzeri sınır: büyük hacim işlenebilmeli."""
        big_vol = synthetic_volume(D=100, H=64, W=64)
        big_mask = np.zeros((100, 64, 64), dtype=np.int32)
        big_mask[50:, 10:30, 10:30] = 1

        pipeline = self._build_pipeline()
        pipeline.nnunet._predictor.predict_single_npy_array.return_value = big_mask
        pipeline.loader.load = MagicMock(return_value=big_vol)
        pipeline.loader.meta = DicomMeta(
            rows=64, cols=64, n_slices=100,
            pixel_spacing=[1.0, 1.0], slice_thickness=1.0,
        )
        result = pipeline.predict("big_synthetic")
        self.assertEqual(result.shape[0], 100)


# ── CLI testleri ──────────────────────────────────────────────────────────────

class TestCLI(unittest.TestCase):
    """cli.py komut satırı araçlarını test eder."""

    def test_colorize_mask_shape(self):
        from cli import _colorize_mask
        mask = np.zeros((64, 64), dtype=np.int32)
        mask[10:30, 10:30] = 1
        mask[40:60, 40:60] = 3
        rgb = _colorize_mask(mask)
        self.assertEqual(rgb.shape, (64, 64, 3))
        self.assertEqual(rgb.dtype, np.uint8)

    def test_colorize_background_black(self):
        from cli import _colorize_mask
        mask = np.zeros((32, 32), dtype=np.int32)
        rgb = _colorize_mask(mask)
        np.testing.assert_array_equal(rgb[0, 0], [0, 0, 0])

    def test_colorize_organ1_red(self):
        from cli import _colorize_mask, _CLASS_COLORS
        mask = np.ones((32, 32), dtype=np.int32)
        rgb = _colorize_mask(mask)
        np.testing.assert_array_equal(rgb[0, 0], list(_CLASS_COLORS[1]))

    def test_parser_required_args(self):
        from cli import _build_parser
        parser = _build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])  # dicom_path ve output_dir zorunlu

    def test_parser_defaults(self):
        from cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "--dicom_path", "/data/ct",
            "--output_dir", "/out",
        ])
        self.assertEqual(args.window, "soft_tissue")
        self.assertFalse(args.no_sam2)
        self.assertIsNone(args.slice_idx)
        self.assertEqual(args.folds, [0])

    def test_parser_no_sam2_flag(self):
        from cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "--dicom_path", "/data/ct",
            "--output_dir", "/out",
            "--no_sam2",
        ])
        self.assertTrue(args.no_sam2)


if __name__ == "__main__":
    unittest.main()
