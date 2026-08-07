import os
import sys
import unittest
from unittest.mock import MagicMock, patch, mock_open

# Insert experiments directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

# Import functions to test from utils
from utils import normalize_model_name, format_resolution, save_summary, benchmark_model

class TestUtils(unittest.TestCase):

    def test_normalize_model_name(self):
        self.assertEqual(normalize_model_name("yolov8n.pt"), "YOLOv8n")
        self.assertEqual(normalize_model_name("yolov8s"), "YOLOv8s")
        self.assertEqual(normalize_model_name("/path/to/yolov8m.pt"), "YOLOv8m")
        self.assertEqual(normalize_model_name("custom_model"), "custom_model")
        self.assertEqual(normalize_model_name("YOLOv8n.pt"), "YOLOv8n")

    def test_format_resolution(self):
        self.assertEqual(format_resolution(640), "640x640")
        self.assertEqual(format_resolution("416"), "416x416")
        self.assertEqual(format_resolution("640x640"), "640x640")

    @patch('utils.pd.read_csv')
    @patch('utils.os.path.exists')
    @patch('utils.os.makedirs')
    @patch.object(pd.DataFrame, 'to_csv', autospec=True)
    def test_save_summary_new_file(self, mock_to_csv, mock_makedirs, mock_exists, mock_read_csv):
        mock_exists.return_value = False

        # Call save_summary
        save_summary(640, "yolov8n.pt", "FP32", 15.5, 65.2, "New entry")

        # Verify makedirs was called
        mock_makedirs.assert_called_once()

        # Verify to_csv was called on a DataFrame instance
        self.assertTrue(mock_to_csv.called)
        # Extract the DataFrame instance passed as the first argument
        called_df = mock_to_csv.call_args[0][0]
        self.assertEqual(len(called_df), 1)
        self.assertEqual(called_df.iloc[0]["Resolution"], "640x640")
        self.assertEqual(called_df.iloc[0]["Model"], "YOLOv8n")
        self.assertEqual(called_df.iloc[0]["Precision"], "FP32")
        self.assertEqual(called_df.iloc[0]["Average_FPS"], 15.5)
        self.assertEqual(called_df.iloc[0]["Average_Latency_ms"], 65.2)

    @patch('utils.pd.read_csv')
    @patch('utils.os.path.exists')
    @patch('utils.os.makedirs')
    @patch.object(pd.DataFrame, 'to_csv', autospec=True)
    def test_save_summary_update_entry(self, mock_to_csv, mock_makedirs, mock_exists, mock_read_csv):
        mock_exists.return_value = True
        # Mock existing summary containing the same entry
        existing_df = pd.DataFrame([
            {
                "Resolution": "640x640",
                "Model": "YOLOv8n",
                "Precision": "FP32",
                "Average_FPS": 10.0,
                "Average_Latency_ms": 100.0,
                "Observation": "Old entry"
            }
        ])
        mock_read_csv.return_value = existing_df

        # Call save_summary with new metrics
        save_summary(640, "yolov8n.pt", "FP32", 20.0, 50.0, "Updated entry")

        self.assertTrue(mock_to_csv.called)
        called_df = mock_to_csv.call_args[0][0]
        self.assertEqual(len(called_df), 1)
        self.assertEqual(called_df.iloc[0]["Average_FPS"], 20.0)
        self.assertEqual(called_df.iloc[0]["Average_Latency_ms"], 50.0)
        self.assertEqual(called_df.iloc[0]["Observation"], "Updated entry")

    @patch('utils.pd.read_csv')
    @patch('utils.os.path.exists')
    @patch('utils.os.makedirs')
    @patch.object(pd.DataFrame, 'to_csv', autospec=True)
    def test_save_summary_schema_migration(self, mock_to_csv, mock_makedirs, mock_exists, mock_read_csv):
        mock_exists.return_value = True
        # Mock existing summary with missing Model and Precision columns
        legacy_df = pd.DataFrame([
            {
                "Resolution": "640x640",
                "Average_FPS": 7.6,
                "Average_Latency_ms": 110.0,
                "Observation": "Higher detection quality"
            }
        ])
        mock_read_csv.return_value = legacy_df

        # Save a new independent entry
        save_summary(416, "yolov8n.pt", "FP32", 14.2, 65.0, "Faster Inference")

        self.assertTrue(mock_to_csv.called)
        called_df = mock_to_csv.call_args[0][0]

        # Verify that legacy row got model 'YOLOv8n' and precision 'FP32'
        legacy_row = called_df[called_df["Resolution"] == "640x640"].iloc[0]
        self.assertEqual(legacy_row["Model"], "YOLOv8n")
        self.assertEqual(legacy_row["Precision"], "FP32")
        self.assertEqual(legacy_row["Average_FPS"], 7.6)

        # Verify new row is also there
        new_row = called_df[called_df["Resolution"] == "416x416"].iloc[0]
        self.assertEqual(new_row["Model"], "YOLOv8n")
        self.assertEqual(new_row["Precision"], "FP32")
        self.assertEqual(new_row["Average_FPS"], 14.2)

    @patch('utils.YOLO')
    @patch('utils.cv2.VideoCapture')
    @patch('utils.torch.cuda.is_available')
    def test_benchmark_model_synthetic_fallback(self, mock_cuda_is_available, mock_video_capture, mock_yolo_class):
        mock_cuda_is_available.return_value = False

        # Mock webcam to fail to open
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap

        # Mock YOLO model call
        mock_yolo_instance = MagicMock()
        mock_yolo_class.return_value = mock_yolo_instance

        # Call benchmark_model
        avg_latency, fps, actual_half = benchmark_model("yolov8n.pt", 640, half=False, num_frames=10)

        # Verify YOLO was called for warmup (5) + inference (10) = 15 times
        self.assertEqual(mock_yolo_instance.call_count, 15)
        self.assertFalse(actual_half)
        self.assertGreater(fps, 0)
        self.assertGreater(avg_latency, 0)

    @patch('utils.YOLO')
    @patch('utils.torch.cuda.is_available')
    def test_benchmark_model_division_by_zero_hardening(self, mock_cuda_is_available, mock_yolo_class):
        mock_cuda_is_available.return_value = False

        # Mock YOLO model instance
        mock_yolo_instance = MagicMock()
        mock_yolo_class.return_value = mock_yolo_instance

        # Force num_frames to 0 to test division-by-zero hardening
        avg_latency, fps, actual_half = benchmark_model("yolov8n.pt", 640, half=False, num_frames=0)

        self.assertEqual(avg_latency, 0.0)
        self.assertEqual(fps, 0.0)
        self.assertFalse(actual_half)

if __name__ == '__main__':
    unittest.main()
