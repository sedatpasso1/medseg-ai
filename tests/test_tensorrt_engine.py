"""
TensorRT inference katmanı birim testleri.

tensorrt, onnxruntime, pycuda kurulu olmadan da çalışır.
Ağır bağımlılıklar mock'lanır.
"""
from __future__ import annotations

import sys
import types
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Mock nnunetv2 ─────────────────────────────────────────────────────────────

def _inject_nnunet_mock() -> None:
    if "nnunetv2" not in sys.modules:
        pkg = types.ModuleType("nnunetv2")
        infer = types.ModuleType("nnunetv2.inference")
        pred_mod = types.ModuleType("nnunetv2.inference.predict_from_raw_data")
        pred_mod.nnUNetPredictor = MagicMock()
        sys.modules.update({
            "nnunetv2": pkg,
            "nnunetv2.inference": infer,
            "nnunetv2.inference.predict_from_raw_data": pred_mod,
        })


_inject_nnunet_mock()


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı: TRTnnUNetPredictor oluştur (mock bağımlılıklarla)
# ─────────────────────────────────────────────────────────────────────────────

def _make_trt_predictor(trt_ok: bool = False, ort_ok: bool = False,
                         preferred: str = "auto"):
    """TRTnnUNetPredictor'ı mock bağımlılıklarla oluşturur."""
    import tensorrt_nnunet as trt_mod

    mock_pytorch_pred = MagicMock()
    mock_pytorch_pred._predictor = None

    # medseg_pipeline.nnUNetPredictor'ı mock'la → __init__ içinde import edilir
    with patch.object(trt_mod, "_check_tensorrt", return_value=trt_ok), \
         patch.object(trt_mod, "_check_onnxruntime", return_value=ort_ok), \
         patch("medseg_pipeline.nnUNetPredictor") as MockCls:

        MockCls.return_value = mock_pytorch_pred

        pred = trt_mod.TRTnnUNetPredictor(
            model_folder="mock_folder",
            preferred_mode=preferred,
        )

    # _pytorch_pred'i doğrudan set et
    pred._pytorch_pred = mock_pytorch_pred
    return pred


# ─────────────────────────────────────────────────────────────────────────────
# Test: bağımlılık kontrolleri
# ─────────────────────────────────────────────────────────────────────────────

class TestDependencyChecks(unittest.TestCase):

    def test_check_tensorrt_returns_bool(self):
        import tensorrt_nnunet as trt_mod
        result = trt_mod._check_tensorrt()
        self.assertIsInstance(result, bool)

    def test_check_onnxruntime_returns_bool(self):
        import tensorrt_nnunet as trt_mod
        result = trt_mod._check_onnxruntime()
        self.assertIsInstance(result, bool)

    def test_check_tensorrt_false_when_unavailable(self):
        import tensorrt_nnunet as trt_mod
        with patch.dict(sys.modules, {"tensorrt": None}):
            result = trt_mod._check_tensorrt()
        self.assertFalse(result)

    def test_check_onnxruntime_false_when_unavailable(self):
        import tensorrt_nnunet as trt_mod
        with patch.dict(sys.modules, {"onnxruntime": None}):
            result = trt_mod._check_onnxruntime()
        self.assertFalse(result)


# ─────────────────────────────────────────────────────────────────────────────
# Test: mod seçimi
# ─────────────────────────────────────────────────────────────────────────────

class TestModeSelection(unittest.TestCase):

    def test_selects_pytorch_when_no_trt_no_ort(self):
        pred = _make_trt_predictor(trt_ok=False, ort_ok=False)
        self.assertEqual(pred.active_mode, "pytorch")

    def test_selects_ort_when_no_trt(self):
        pred = _make_trt_predictor(trt_ok=False, ort_ok=True)
        self.assertEqual(pred.active_mode, "ort")

    def test_preferred_mode_overrides_auto_to_pytorch(self):
        pred = _make_trt_predictor(trt_ok=True, ort_ok=True, preferred="pytorch")
        self.assertEqual(pred.active_mode, "pytorch")

    def test_preferred_mode_overrides_auto_to_ort(self):
        pred = _make_trt_predictor(trt_ok=False, ort_ok=False, preferred="ort")
        self.assertEqual(pred.active_mode, "ort")

    def test_active_mode_property_returns_string(self):
        pred = _make_trt_predictor()
        self.assertIsInstance(pred.active_mode, str)
        self.assertIn(pred.active_mode, ("pytorch", "ort", "trt"))


# ─────────────────────────────────────────────────────────────────────────────
# Test: ONNX export
# ─────────────────────────────────────────────────────────────────────────────

@unittest.skipUnless(
    __import__("importlib").util.find_spec("torch") is not None,
    "torch kurulu değil",
)
class TestOnnxExport(unittest.TestCase):

    def _make_mock_network(self):
        import torch
        mock_net = MagicMock()
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_net.parameters.return_value = iter([mock_param])
        return mock_net

    def test_export_to_onnx_calls_torch_export(self):
        from tensorrt_nnunet import export_to_onnx

        mock_net = self._make_mock_network()

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "model.onnx"

            with patch("torch.onnx.export") as mock_export, \
                 patch.object(Path, "stat") as mock_stat:
                mock_stat.return_value.st_size = 1024 * 10
                export_to_onnx(mock_net, (32, 32, 32), onnx_path)

            mock_export.assert_called_once()
            call_args = mock_export.call_args
            # İlk positional arg: network
            self.assertIs(call_args[0][0], mock_net)
            # ONNX path str olarak geçmeli
            self.assertIn(str(onnx_path), call_args[0])

    def test_export_creates_parent_dirs(self):
        from tensorrt_nnunet import export_to_onnx

        mock_net = self._make_mock_network()

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "subdir" / "model.onnx"
            self.assertFalse(onnx_path.parent.exists())

            with patch("torch.onnx.export"), \
                 patch.object(Path, "stat") as mock_stat:
                mock_stat.return_value.st_size = 1024
                export_to_onnx(mock_net, (16, 16, 16), onnx_path)

            self.assertTrue(onnx_path.parent.exists())

    def test_export_dynamic_axes_contain_batch(self):
        from tensorrt_nnunet import export_to_onnx

        mock_net = self._make_mock_network()
        captured = {}

        def capture(*args, **kwargs):
            captured.update(kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "model.onnx"
            with patch("torch.onnx.export", side_effect=capture), \
                 patch.object(Path, "stat") as mock_stat:
                mock_stat.return_value.st_size = 0
                export_to_onnx(mock_net, (32, 32, 32), onnx_path)

        axes = captured.get("dynamic_axes", {})
        self.assertIn("input", axes)
        self.assertIn(0, axes["input"])  # batch dim


# ─────────────────────────────────────────────────────────────────────────────
# Test: PyTorch fallback
# ─────────────────────────────────────────────────────────────────────────────

class TestPytorchFallback(unittest.TestCase):

    def test_predict_delegates_to_pytorch_pred(self):
        """Mode=pytorch olduğunda _pytorch_pred.predict() çağrılmalı."""
        pred = _make_trt_predictor(trt_ok=False, ort_ok=False)
        pred._mode = "pytorch"

        expected_mask = np.zeros((4, 8, 8), dtype=np.int32)
        pred._pytorch_pred.predict.return_value = expected_mask
        pred._pytorch_pred._ensure_loaded = MagicMock()

        volume = np.random.rand(4, 8, 8).astype(np.float32)
        result = pred.predict(volume)

        pred._pytorch_pred._ensure_loaded.assert_called_once()
        pred._pytorch_pred.predict.assert_called_once_with(volume)
        np.testing.assert_array_equal(result, expected_mask)

    def test_result_dtype_is_int32(self):
        pred = _make_trt_predictor(trt_ok=False, ort_ok=False)
        pred._mode = "pytorch"
        pred._pytorch_pred.predict.return_value = np.zeros((2, 4, 4), dtype=np.int32)
        pred._pytorch_pred._ensure_loaded = MagicMock()

        result = pred.predict(np.zeros((2, 4, 4), dtype=np.float32))
        self.assertEqual(result.dtype, np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Test: ORT inference
# ─────────────────────────────────────────────────────────────────────────────

class TestOrtInference(unittest.TestCase):

    def _make_ort_session(self, out_shape):
        sess = MagicMock()
        inp = MagicMock()
        inp.name = "input"
        inp.shape = (1, 1, *out_shape)
        sess.get_inputs.return_value = [inp]
        fake_out = np.zeros((1, 3, *out_shape), dtype=np.float32)
        fake_out[0, 1, 1:3, 1:3, 1:3] = 1.0
        sess.run.return_value = [fake_out]
        sess.get_providers.return_value = ["CPUExecutionProvider"]
        return sess

    def test_ort_session_run_called(self):
        """ORT modunda session.run() çağrılmalı."""
        pred = _make_trt_predictor(trt_ok=False, ort_ok=True)
        pred._mode = "ort"

        D, H, W = 8, 8, 8
        mock_sess = self._make_ort_session((D, H, W))
        pred._ort_session = mock_sess

        # _prepare'nin model yüklemesini mock'la
        pred._pytorch_pred._predictor = MagicMock()
        pred._pytorch_pred._predictor.configuration_manager.patch_size = [D, H, W]
        pred._pytorch_pred._predictor.label_manager.num_segmentation_heads = 3
        pred._pytorch_pred._ensure_loaded = MagicMock()

        # ONNX dosya var → ORT session kurulmuş simülasyonu
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "mock_model.onnx"
            onnx_path.touch()
            pred._onnx_path = onnx_path

            volume = np.random.rand(D, H, W).astype(np.float32)
            result = pred._predict_ort(volume)

        mock_sess.run.assert_called()
        self.assertEqual(result.dtype, np.int32)
        self.assertEqual(result.shape, (D, H, W))


# ─────────────────────────────────────────────────────────────────────────────
# Test: engine caching
# ─────────────────────────────────────────────────────────────────────────────

class TestEngineCaching(unittest.TestCase):

    def test_onnx_path_property(self):
        pred = _make_trt_predictor()
        self.assertIsNone(pred.onnx_path)  # _prepare çağrılmadan None

    def test_engine_path_property(self):
        pred = _make_trt_predictor()
        self.assertIsNone(pred.engine_path)  # _prepare çağrılmadan None

    def test_pytorch_mode_no_onnx_export(self):
        """PyTorch modunda ONNX export yapılmamalı."""
        import tensorrt_nnunet as trt_mod
        pred = _make_trt_predictor(trt_ok=False, ort_ok=False)
        pred._pytorch_pred._ensure_loaded = MagicMock()
        pred._pytorch_pred.predict = MagicMock(
            return_value=np.zeros((2, 4, 4), dtype=np.int32)
        )

        with patch.object(trt_mod, "export_to_onnx") as mock_export:
            pred.predict(np.zeros((2, 4, 4), dtype=np.float32))
            mock_export.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Test: MedSegPipeline use_tensorrt
# ─────────────────────────────────────────────────────────────────────────────

class TestMedSegPipelineIntegration(unittest.TestCase):

    def test_use_tensorrt_false_uses_nnunet_predictor(self):
        from medseg_pipeline import MedSegPipeline, nnUNetPredictor

        with patch.object(nnUNetPredictor, "_ensure_loaded"):
            pipeline = MedSegPipeline(
                model_folder="mock",
                use_sam2=False,
                use_tensorrt=False,
            )

        self.assertIsInstance(pipeline.nnunet, nnUNetPredictor)

    def test_use_tensorrt_true_uses_trt_predictor(self):
        import tensorrt_nnunet as trt_mod
        from medseg_pipeline import MedSegPipeline

        mock_pytorch_pred = MagicMock()
        mock_pytorch_pred._predictor = None

        with patch.object(trt_mod, "_check_tensorrt", return_value=False), \
             patch.object(trt_mod, "_check_onnxruntime", return_value=False), \
             patch("medseg_pipeline.nnUNetPredictor") as MockCls:

            MockCls.return_value = mock_pytorch_pred

            pipeline = MedSegPipeline(
                model_folder="mock",
                use_sam2=False,
                use_tensorrt=True,
            )

        self.assertIsInstance(pipeline.nnunet, trt_mod.TRTnnUNetPredictor)
        self.assertEqual(pipeline.nnunet.active_mode, "pytorch")

    def test_backend_type_field_exists_in_result(self):
        """SegmentationResult'ta backend_type field'ı olmalı."""
        from medseg_pipeline import SegmentationResult
        fields = SegmentationResult.__dataclass_fields__
        self.assertIn("backend_type", fields)

    def test_backend_type_default_is_pytorch(self):
        from medseg_pipeline import SegmentationResult
        r = SegmentationResult(
            dicom_path="", modality="CT", shape=[1, 2, 2],
            pixel_spacing=[1.0, 1.0], slice_thickness=1.0,
            class_ids=[], class_voxel_counts={},
            nnunet_inference_sec=0.0, sam2_refine_sec=0.0,
            total_sec=0.0, output_nifti="", window="soft_tissue",
            device="cpu",
        )
        self.assertEqual(r.backend_type, "pytorch")


if __name__ == "__main__":
    unittest.main()
