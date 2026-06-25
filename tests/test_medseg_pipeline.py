"""
MedSeg-AI birim testleri.
Gerçek model veya DICOM dosyası olmadan çalışır — mock/stub kullanır.

Kullanım:
    pytest tests/test_medseg_pipeline.py -v
"""
from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import UID

# Test altındaki modül
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from medseg_pipeline import (
    DICOMLoader,
    DicomMeta,
    SAM2Refiner,
    SegmentationResult,
    MedSegPipeline,
    nnUNetPredictor,
)


# ── Yardımcı fabrika ──────────────────────────────────────────────────────────

def _make_dicom_file(tmp_dir: Path, hu_value: float = 50.0,
                     rows: int = 64, cols: int = 64,
                     instance_num: int = 1) -> Path:
    """Basit sentetik DICOM dosyası oluşturur (geçerli DICM header ile)."""
    out = tmp_dir / f"slice_{instance_num:03d}.dcm"

    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID    = UID("1.2.840.10008.5.1.4.1.1.2")
    file_meta.MediaStorageSOPInstanceUID = UID(f"1.2.3.{instance_num}")
    file_meta.TransferSyntaxUID          = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(str(out), {}, file_meta=file_meta,
                     is_implicit_VR=False, is_little_endian=True)
    ds.is_implicit_VR  = False
    ds.is_little_endian = True

    ds.PatientID        = "TEST001"
    ds.StudyDate        = "20240101"
    ds.Modality         = "CT"
    ds.Rows             = rows
    ds.Columns          = cols
    ds.PixelSpacing     = [1.0, 1.0]
    ds.SliceThickness   = 2.5
    ds.RescaleSlope     = 1.0
    ds.RescaleIntercept = -1024.0
    ds.InstanceNumber   = instance_num
    ds.BitsAllocated    = 16
    ds.BitsStored       = 16
    ds.HighBit          = 15
    ds.PixelRepresentation = 1
    ds.SamplesPerPixel  = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SOPClassUID      = "1.2.840.10008.5.1.4.1.1.2"
    ds.SOPInstanceUID   = f"1.2.3.{instance_num}"

    # hu_value = pixel * slope + intercept → pixel = hu_value - intercept
    pixel_value = int(hu_value - ds.RescaleIntercept)
    ds.PixelData = np.full((rows, cols), pixel_value, dtype=np.int16).tobytes()

    ds.save_as(str(out), write_like_original=False)
    return out


# ── Test sınıfları ────────────────────────────────────────────────────────────

class TestDICOMLoader(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_load_single_file(self):
        f = _make_dicom_file(self.tmp, hu_value=50.0)
        loader = DICOMLoader(window="soft_tissue")
        vol = loader.load(f)
        self.assertEqual(vol.ndim, 3)
        self.assertEqual(vol.shape[0], 1)
        self.assertGreaterEqual(vol.min(), 0.0 - 1e-6)
        self.assertLessEqual(vol.max(), 1.0 + 1e-6)

    def test_load_series(self):
        for i in range(5):
            _make_dicom_file(self.tmp, hu_value=float(i * 10), instance_num=i + 1)
        loader = DICOMLoader(window="soft_tissue")
        vol = loader.load(self.tmp)
        self.assertEqual(vol.shape[0], 5)
        self.assertIsNotNone(loader.meta)
        self.assertEqual(loader.meta.n_slices, 5)

    def test_windowing_clip(self):
        # HU=1500 (kemik aralığının dışı) soft_tissue penceresinde 1.0'a clip edilmeli
        f = _make_dicom_file(self.tmp, hu_value=1500.0)
        loader = DICOMLoader(window="soft_tissue")
        vol = loader.load(f)
        self.assertAlmostEqual(float(vol.max()), 1.0, places=3)

    def test_windowing_below(self):
        f = _make_dicom_file(self.tmp, hu_value=-2000.0)
        loader = DICOMLoader(window="soft_tissue")
        vol = loader.load(f)
        self.assertAlmostEqual(float(vol.min()), 0.0, places=3)

    def test_meta_fields(self):
        f = _make_dicom_file(self.tmp)
        loader = DICOMLoader()
        loader.load(f)
        self.assertEqual(loader.meta.modality, "CT")
        self.assertEqual(loader.meta.patient_id, "TEST001")
        self.assertAlmostEqual(loader.meta.slice_thickness, 2.5)

    def test_invalid_window(self):
        with self.assertRaises(ValueError):
            DICOMLoader(window="nonexistent_window")

    def test_empty_folder_raises(self):
        loader = DICOMLoader()
        with self.assertRaises(FileNotFoundError):
            loader.load(self.tmp)  # boş klasör


class TestDICOMLoaderWindows(unittest.TestCase):
    """Her pencere ön ayarı için sanity check."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        _make_dicom_file(self.tmp, hu_value=50.0)

    def test_all_presets(self):
        for preset in DICOMLoader.WINDOW_PRESETS:
            loader = DICOMLoader(window=preset)
            vol = loader.load(list(self.tmp.glob("*.dcm"))[0])
            self.assertGreaterEqual(vol.min(), 0.0 - 1e-6,
                                    f"{preset}: min<0")
            self.assertLessEqual(vol.max(), 1.0 + 1e-6,
                                 f"{preset}: max>1")


class TestSAM2Refiner(unittest.TestCase):

    def _make_volume_mask(self, D=4, H=32, W=32):
        vol  = np.random.rand(D, H, W).astype(np.float32)
        mask = np.zeros((D, H, W), dtype=np.int32)
        mask[:, 8:24, 8:24] = 1  # basit dikdörtgen organ
        return vol, mask

    def test_refine_with_mock_sam2(self):
        """SAM2 kurulu olmasa bile mock ile refine çalışmalı."""
        vol, coarse = self._make_volume_mask()

        # sam2 import'unu mock'la
        fake_sam2_mod = types.ModuleType("sam2")
        fake_build   = types.ModuleType("sam2.build_sam")
        fake_pred_mod = types.ModuleType("sam2.sam2_image_predictor")

        mock_predictor = MagicMock()
        D, H, W = vol.shape
        mock_predictor.predict.return_value = (
            np.ones((1, H, W), dtype=bool),  # mask tam dolu
            np.array([0.95]),
            None,
        )
        fake_pred_mod.SAM2ImagePredictor = MagicMock(return_value=mock_predictor)
        fake_build.build_sam2 = MagicMock(return_value=MagicMock())

        import sys
        sys.modules.setdefault("sam2", fake_sam2_mod)
        sys.modules["sam2.build_sam"] = fake_build
        sys.modules["sam2.sam2_image_predictor"] = fake_pred_mod

        refiner = SAM2Refiner(checkpoint="mock.pt")
        refiner._predictor = mock_predictor  # inject directly

        result = refiner.refine(vol, coarse)
        self.assertEqual(result.shape, coarse.shape)
        self.assertEqual(result.dtype, np.int32)

    def test_mask_to_bbox(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:30, 20:50] = True
        bbox = SAM2Refiner._mask_to_bbox(mask)
        self.assertIsNotNone(bbox)
        self.assertEqual(bbox.shape, (4,))
        x1, y1, x2, y2 = bbox
        self.assertEqual(int(x1), 20)
        self.assertEqual(int(y1), 10)
        self.assertEqual(int(x2), 49)
        self.assertEqual(int(y2), 29)

    def test_mask_to_bbox_empty(self):
        mask = np.zeros((64, 64), dtype=bool)
        self.assertIsNone(SAM2Refiner._mask_to_bbox(mask))

    def test_to_rgb(self):
        gray = np.random.rand(64, 64).astype(np.float32)
        rgb = SAM2Refiner._to_rgb(gray)
        self.assertEqual(rgb.shape, (64, 64, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertLessEqual(rgb.max(), 255)
        self.assertGreaterEqual(rgb.min(), 0)


class TestMedSegPipeline(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = Path(tempfile.mkdtemp())
        # 3 dilimlik sentetik seri
        for i in range(3):
            _make_dicom_file(self.tmp, hu_value=50.0, instance_num=i + 1)

    def _mock_pipeline(self):
        """nnUNet ve SAM2'yi mock'layıp pipeline döner."""
        with patch.object(nnUNetPredictor, "_ensure_loaded"):
            pipeline = MedSegPipeline(
                model_folder="mock_model",
                use_sam2=False,  # SAM2 olmadan test
            )

        mock_mask = np.zeros((3, 64, 64), dtype=np.int32)
        mock_mask[:, 10:30, 10:30] = 1
        mock_mask[:, 30:50, 30:50] = 2
        pipeline.nnunet._predictor = MagicMock()
        pipeline.nnunet._predictor.predict_single_npy_array.return_value = mock_mask
        pipeline.nnunet._ensure_loaded = lambda: None
        return pipeline

    def test_predict_returns_result(self):
        pipeline = self._mock_pipeline()
        result = pipeline.predict(self.tmp)
        self.assertIsInstance(result, SegmentationResult)
        self.assertIn(1, result.class_ids)
        self.assertIn(2, result.class_ids)
        self.assertGreater(result.total_sec, 0)

    def test_save_nifti(self):
        pipeline = self._mock_pipeline()
        result = pipeline.predict(self.tmp)
        nifti_path = pipeline.save_nifti(result, self.out)
        self.assertTrue(nifti_path.exists())
        self.assertTrue(str(nifti_path).endswith(".nii.gz"))

    def test_save_json(self):
        pipeline = self._mock_pipeline()
        result = pipeline.predict(self.tmp)
        pipeline.save_nifti(result, self.out)
        json_path = pipeline.save_json(result, self.out)
        self.assertTrue(json_path.exists())
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("class_ids", data)
        self.assertIn("nnunet_inference_sec", data)
        self.assertIn("output_nifti", data)

    def test_nifti_shape_matches_mask(self):
        import nibabel as nib
        pipeline = self._mock_pipeline()
        result = pipeline.predict(self.tmp)
        nifti_path = pipeline.save_nifti(result, self.out)
        img = nib.load(str(nifti_path))
        data = img.get_fdata()
        self.assertEqual(list(data.shape), result.shape)

    def test_no_sam2_mode(self):
        pipeline = self._mock_pipeline()
        self.assertFalse(pipeline.use_sam2)
        result = pipeline.predict(self.tmp)
        self.assertIsInstance(result, SegmentationResult)


if __name__ == "__main__":
    unittest.main()
