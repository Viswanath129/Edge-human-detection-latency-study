import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Force prepend experiments folder to path so import namespace matches
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils

class TestUtils(unittest.TestCase):

    @patch('utils.YOLO')
    @patch('utils.cv2.VideoCapture')
    def test_benchmark_model_synthetic_fallback(self, mock_vc, mock_yolo):
        # Setup mocks
        mock_vc_inst = MagicMock()
        mock_vc_inst.isOpened.return_value = False
        mock_vc.return_value = mock_vc_inst

        mock_yolo_inst = MagicMock()
        mock_yolo.return_value = mock_yolo_inst

        # Run benchmark
        avg_latency, fps, actual_half = utils.benchmark_model("yolov8n.pt", resolution=640, half=False, num_frames=10)

        # Assertions
        self.assertGreaterEqual(avg_latency, 0.0)
        self.assertGreaterEqual(fps, 0.0)
        self.assertFalse(actual_half)
        self.assertTrue(mock_yolo_inst.called)

    @patch('utils.YOLO')
    @patch('utils.cv2.VideoCapture')
    @patch('utils.torch.cuda.is_available', return_value=True)
    @patch('utils.torch.cuda.synchronize')
    def test_benchmark_model_cuda_precision(self, mock_sync, mock_is_avail, mock_vc, mock_yolo):
        mock_vc_inst = MagicMock()
        mock_vc_inst.isOpened.return_value = False
        mock_vc.return_value = mock_vc_inst

        mock_yolo_inst = MagicMock()
        mock_yolo.return_value = mock_yolo_inst

        avg_latency, fps, actual_half = utils.benchmark_model("yolov8n.pt", resolution=640, half=True, num_frames=10)

        self.assertTrue(actual_half)
        self.assertTrue(mock_is_avail.called)
        self.assertTrue(mock_sync.called)

    @patch('utils.pd.DataFrame.to_csv', autospec=True)
    @patch('utils.os.path.exists', return_value=False)
    def test_save_summary_creation(self, mock_exists, mock_to_csv):
        # Test save summary when CSV does not exist
        utils.save_summary(640, "yolov8n.pt", "FP32", 10.0, 100.0, "Test Obs")

        # Verify to_csv was called
        self.assertTrue(mock_to_csv.called)

        # Check first argument (which is the DataFrame instance self)
        df_inst = mock_to_csv.call_args[0][0]
        self.assertEqual(len(df_inst), 1)
        self.assertEqual(df_inst.iloc[0]["Resolution"], "640x640")
        self.assertEqual(df_inst.iloc[0]["Model"], "YOLOv8n")
        self.assertEqual(df_inst.iloc[0]["Precision"], "FP32")
        self.assertEqual(df_inst.iloc[0]["Average_FPS"], 10.0)
        self.assertEqual(df_inst.iloc[0]["Average_Latency_ms"], 100.0)
        self.assertEqual(df_inst.iloc[0]["Observation"], "Test Obs")

    @patch('utils.pd.DataFrame.to_csv', autospec=True)
    @patch('utils.pd.read_csv')
    @patch('utils.os.path.exists', return_value=True)
    def test_save_summary_update_existing(self, mock_exists, mock_read_csv, mock_to_csv):
        import pandas as pd
        # Simulate existing CSV with identical key columns
        existing_df = pd.DataFrame({
            "Resolution": ["640x640"],
            "Model": ["YOLOv8n"],
            "Precision": ["FP32"],
            "Average_FPS": [5.0],
            "Average_Latency_ms": [200.0],
            "Observation": ["Old Obs"]
        })
        mock_read_csv.return_value = existing_df

        utils.save_summary(640, "yolov8n.pt", "FP32", 12.5, 80.0, "Updated Obs")

        # Verify updating behaves correctly instead of duplicating
        self.assertTrue(mock_to_csv.called)
        df_inst = mock_to_csv.call_args[0][0]
        self.assertEqual(len(df_inst), 1)
        self.assertEqual(df_inst.iloc[0]["Average_FPS"], 12.5)
        self.assertEqual(df_inst.iloc[0]["Average_Latency_ms"], 80.0)
        self.assertEqual(df_inst.iloc[0]["Observation"], "Updated Obs")

    def test_normalize_model_name(self):
        self.assertEqual(utils.normalize_model_name("yolov8n.pt"), "YOLOv8n")
        self.assertEqual(utils.normalize_model_name("yolov8s"), "YOLOv8s")
        self.assertEqual(utils.normalize_model_name("custom_model"), "custom_model")

if __name__ == "__main__":
    unittest.main()
