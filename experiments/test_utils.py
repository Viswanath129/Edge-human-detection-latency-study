import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Configure sys.path to resolve utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import normalize_model_name, format_resolution, benchmark_model, save_summary
import pandas as pd
import numpy as np

class TestUtils(unittest.TestCase):

    def test_normalize_model_name(self):
        self.assertEqual(normalize_model_name("yolov8n.pt"), "YOLOv8n")
        self.assertEqual(normalize_model_name("YOLOv8s"), "YOLOv8s")
        self.assertEqual(normalize_model_name("yolov8m"), "YOLOv8m")
        self.assertEqual(normalize_model_name("custom_model.pt"), "custom_model")
        self.assertEqual(normalize_model_name("another_model"), "another_model")

    def test_format_resolution(self):
        self.assertEqual(format_resolution(640), "640x640")
        self.assertEqual(format_resolution("416"), "416x416")
        self.assertEqual(format_resolution("640x640"), "640x640")
        self.assertEqual(format_resolution("random_str"), "random_str")

    @patch('utils.YOLO')
    @patch('utils.cv2.VideoCapture')
    @patch.dict(os.environ, {"FORCE_SYNTHETIC": "true"})
    def test_benchmark_model_synthetic(self, mock_video_capture, mock_yolo_class):
        # Setup mock YOLO
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        # When model is called, return anything
        mock_model.return_value = [MagicMock()]

        # Benchmark with synthetic
        avg_latency, fps, actual_half = benchmark_model("yolov8n.pt", 640, half=False, num_frames=10)

        # Check that YOLO was initialized
        mock_yolo_class.assert_called_once_with("yolov8n.pt")

        # Check results are non-zero/reasonable
        self.assertGreater(fps, 0)
        self.assertGreater(avg_latency, 0)
        self.assertFalse(actual_half)

    @patch('utils.YOLO')
    @patch('utils.cv2.VideoCapture')
    @patch.dict(os.environ, {"FORCE_SYNTHETIC": "false"})
    def test_benchmark_model_webcam(self, mock_video_capture, mock_yolo_class):
        # Setup mock YOLO
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        mock_model.return_value = [MagicMock()]

        # Setup mock VideoCapture
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True

        # Return a simple frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)

        # Benchmark
        avg_latency, fps, actual_half = benchmark_model("yolov8n.pt", 640, half=False, num_frames=10)

        # Ensure it attempted to read from webcam
        mock_cap.read.assert_called()
        mock_cap.release.assert_called_once()

        self.assertGreater(fps, 0)
        self.assertGreater(avg_latency, 0)

    @patch('utils.YOLO')
    @patch.dict(os.environ, {"FORCE_SYNTHETIC": "true"})
    def test_benchmark_model_division_by_zero_protection(self, mock_yolo_class):
        # Setup mock YOLO that fails
        mock_yolo_class.side_effect = Exception("Load failure")

        # Should return (0.0, 0.0, False)
        avg_latency, fps, actual_half = benchmark_model("yolov8n.pt", 640, half=False, num_frames=10)
        self.assertEqual(avg_latency, 0.0)
        self.assertEqual(fps, 0.0)
        self.assertFalse(actual_half)

    @patch('utils.os.path.exists')
    @patch('utils.pd.read_csv')
    @patch.object(pd.DataFrame, 'to_csv', autospec=True)
    def test_save_summary_new_file(self, mock_to_csv, mock_read_csv, mock_exists):
        mock_exists.return_value = False

        save_summary(640, "yolov8n.pt", "FP32", 15.5, 65.2, "Test observation")

        # Check that to_csv was called
        self.assertTrue(mock_to_csv.called)

        # Retrieve the DataFrame passed to to_csv
        called_df = mock_to_csv.call_args[0][0]
        self.assertIsInstance(called_df, pd.DataFrame)
        self.assertEqual(len(called_df), 1)
        self.assertEqual(called_df.iloc[0]["Resolution"], "640x640")
        self.assertEqual(called_df.iloc[0]["Model"], "YOLOv8n")
        self.assertEqual(called_df.iloc[0]["Precision"], "FP32")

    @patch('utils.os.path.exists')
    @patch('utils.pd.read_csv')
    @patch.object(pd.DataFrame, 'to_csv', autospec=True)
    def test_save_summary_existing_update(self, mock_to_csv, mock_read_csv, mock_exists):
        mock_exists.return_value = True

        # Existing summary data
        existing_data = pd.DataFrame([
            {"Resolution": "640x640", "Model": "YOLOv8n", "Precision": "FP32", "Average_FPS": 10.0, "Average_Latency_ms": 100.0, "Observation": "Old"}
        ])
        mock_read_csv.return_value = existing_data

        save_summary(640, "yolov8n.pt", "FP32", 12.5, 80.0, "Updated")

        self.assertTrue(mock_to_csv.called)
        called_df = mock_to_csv.call_args[0][0]

        # Check it updated the existing entry instead of appending
        self.assertEqual(len(called_df), 1)
        self.assertEqual(called_df.iloc[0]["Average_FPS"], 12.5)
        self.assertEqual(called_df.iloc[0]["Average_Latency_ms"], 80.0)
        self.assertEqual(called_df.iloc[0]["Observation"], "Updated")

    @patch('utils.os.path.exists')
    @patch('utils.pd.read_csv')
    @patch.object(pd.DataFrame, 'to_csv', autospec=True)
    def test_save_summary_legacy_schema_migration(self, mock_to_csv, mock_read_csv, mock_exists):
        mock_exists.return_value = True

        # Existing summary data lacking 'Model' and 'Precision' (legacy)
        existing_legacy_data = pd.DataFrame([
            {"Resolution": "640x640", "Average_FPS": 7.6, "Average_Latency_ms": 110.0, "Observation": "Legacy"}
        ])
        mock_read_csv.return_value = existing_legacy_data

        save_summary(416, "yolov8n.pt", "FP32", 14.2, 65.0, "New legacy entry")

        self.assertTrue(mock_to_csv.called)
        called_df = mock_to_csv.call_args[0][0]

        # It should have 2 entries (first migrated to YOLOv8n, FP32; second is new)
        self.assertEqual(len(called_df), 2)

        # Check migrated legacy entry
        self.assertEqual(called_df.iloc[0]["Model"], "YOLOv8n")
        self.assertEqual(called_df.iloc[0]["Precision"], "FP32")
        self.assertEqual(called_df.iloc[0]["Resolution"], "640x640")

        # Check new entry
        self.assertEqual(called_df.iloc[1]["Resolution"], "416x416")
        self.assertEqual(called_df.iloc[1]["Model"], "YOLOv8n")
        self.assertEqual(called_df.iloc[1]["Precision"], "FP32")

if __name__ == '__main__':
    unittest.main()
