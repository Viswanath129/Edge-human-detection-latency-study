import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Ensure the experiments directory is in the import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import torch
from utils import save_summary, benchmark_model

class TestSaveSummary(unittest.TestCase):

    @patch('utils.os.makedirs')
    @patch('utils.os.path.exists')
    @patch('utils.pd.read_csv')
    @patch.object(pd.DataFrame, 'to_csv', autospec=True)
    def test_save_summary_new_file(self, mock_to_csv, mock_read_csv, mock_exists, mock_makedirs):
        # Setup: CSV file does not exist
        mock_exists.return_value = False

        # Run save_summary
        save_summary("640x640", "yolov8n", "FP32", 10.5, 95.2, "Test new file")

        # Verify that to_csv was called once with the correct data
        mock_to_csv.assert_called_once()
        saved_df = mock_to_csv.call_args[0][0] if mock_to_csv.call_args else None
        self.assertIsNotNone(saved_df)
        self.assertIsInstance(saved_df, pd.DataFrame)
        self.assertEqual(len(saved_df), 1)
        self.assertEqual(saved_df.iloc[0]['Resolution'], "640x640")
        self.assertEqual(saved_df.iloc[0]['Model'], "YOLOv8n")
        self.assertEqual(saved_df.iloc[0]['Precision'], "FP32")
        self.assertEqual(saved_df.iloc[0]['Average_FPS'], 10.5)
        self.assertEqual(saved_df.iloc[0]['Average_Latency_ms'], 95.2)

    @patch('utils.os.makedirs')
    @patch('utils.os.path.exists')
    @patch('utils.pd.read_csv')
    @patch.object(pd.DataFrame, 'to_csv', autospec=True)
    def test_save_summary_existing_update(self, mock_to_csv, mock_read_csv, mock_exists, mock_makedirs):
        # Setup: CSV file exists with matching row
        mock_exists.return_value = True
        existing_df = pd.DataFrame([{
            "Resolution": "640x640",
            "Model": "YOLOv8n",
            "Precision": "FP32",
            "Average_FPS": 8.0,
            "Average_Latency_ms": 120.0,
            "Observation": "Old observation"
        }])
        mock_read_csv.return_value = existing_df

        # Run save_summary (update values)
        save_summary("640x640", "yolov8n", "FP32", 12.0, 80.0, "Updated observation")

        # Verify that the row was updated, not appended
        mock_to_csv.assert_called_once()
        saved_df = mock_to_csv.call_args[0][0]
        self.assertIsInstance(saved_df, pd.DataFrame)
        self.assertEqual(len(saved_df), 1)
        self.assertEqual(saved_df.iloc[0]['Average_FPS'], 12.0)
        self.assertEqual(saved_df.iloc[0]['Average_Latency_ms'], 80.0)
        self.assertEqual(saved_df.iloc[0]['Observation'], "Updated observation")

    @patch('utils.os.makedirs')
    @patch('utils.os.path.exists')
    @patch('utils.pd.read_csv')
    @patch.object(pd.DataFrame, 'to_csv', autospec=True)
    def test_save_summary_existing_append(self, mock_to_csv, mock_read_csv, mock_exists, mock_makedirs):
        # Setup: CSV file exists but with a non-matching row
        mock_exists.return_value = True
        existing_df = pd.DataFrame([{
            "Resolution": "416x416",
            "Model": "YOLOv8n",
            "Precision": "FP32",
            "Average_FPS": 20.0,
            "Average_Latency_ms": 50.0,
            "Observation": "Fast inference"
        }])
        mock_read_csv.return_value = existing_df

        # Run save_summary (should append)
        save_summary("640x640", "yolov8n", "FP32", 10.0, 100.0, "Lightweight edge model")

        # Verify that a new row was appended (now 2 rows)
        mock_to_csv.assert_called_once()
        saved_df = mock_to_csv.call_args[0][0]
        self.assertIsInstance(saved_df, pd.DataFrame)
        self.assertEqual(len(saved_df), 2)
        self.assertEqual(saved_df.iloc[1]['Resolution'], "640x640")
        self.assertEqual(saved_df.iloc[1]['Average_FPS'], 10.0)

    @patch('utils.os.makedirs')
    @patch('utils.os.path.exists')
    @patch('utils.pd.read_csv')
    @patch.object(pd.DataFrame, 'to_csv', autospec=True)
    def test_save_summary_missing_columns(self, mock_to_csv, mock_read_csv, mock_exists, mock_makedirs):
        # Setup: CSV file exists but is missing 'Model' or 'Precision' (legacy schema)
        mock_exists.return_value = True
        existing_df = pd.DataFrame([{
            "Resolution": "640x640",
            "Average_FPS": 7.6,
            "Average_Latency_ms": 110.0,
            "Observation": "Legacy data"
        }])
        mock_read_csv.return_value = existing_df

        # Run save_summary
        save_summary("640x640", "yolov8n", "FP32", 10.0, 100.0, "Recovered entry")

        # Verify that schema migration is handled by overwriting with the correct schema
        mock_to_csv.assert_called_once()
        saved_df = mock_to_csv.call_args[0][0]
        self.assertIsInstance(saved_df, pd.DataFrame)
        self.assertEqual(len(saved_df), 1)
        self.assertIn("Model", saved_df.columns)
        self.assertIn("Precision", saved_df.columns)
        self.assertEqual(saved_df.iloc[0]['Model'], "YOLOv8n")


class TestBenchmarkModel(unittest.TestCase):

    @patch('utils.YOLO')
    @patch('utils.torch.cuda.is_available')
    @patch('utils.cv2.VideoCapture')
    def test_benchmark_model_synthetic(self, mock_video_capture, mock_cuda_available, mock_yolo):
        # Setup mocks
        mock_cuda_available.return_value = False

        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap_instance

        mock_yolo_instance = MagicMock()
        mock_yolo.return_value = mock_yolo_instance

        # Run benchmark_model with synthetic enabled
        with patch.dict(os.environ, {"FORCE_SYNTHETIC": "true"}):
            avg_latency, fps, actual_half = benchmark_model("yolov8n.pt", imgsz=64, half=False, num_frames=5)

        # Assertions
        self.assertGreater(fps, 0)
        self.assertGreater(avg_latency, 0)
        self.assertFalse(actual_half)
        # Verify the model was called for 5 warmup frames + 5 benchmark frames
        self.assertEqual(mock_yolo_instance.call_count, 10)

    @patch('utils.YOLO')
    @patch('utils.torch.cuda.is_available')
    def test_benchmark_model_cpu_fp16_warning(self, mock_cuda_available, mock_yolo):
        # Setup: CUDA not available, but FP16 requested
        mock_cuda_available.return_value = False
        mock_yolo_instance = MagicMock()
        mock_yolo.return_value = mock_yolo_instance

        # Run benchmark_model
        with patch.dict(os.environ, {"FORCE_SYNTHETIC": "true"}):
            benchmark_model("yolov8n.pt", imgsz=64, half=True, num_frames=2)

        self.assertTrue(mock_yolo_instance.called)

    @patch('utils.YOLO')
    @patch('utils.torch.cuda.is_available')
    @patch('utils.torch.cuda.synchronize')
    def test_benchmark_model_gpu_sync(self, mock_sync, mock_cuda_available, mock_yolo):
        # Setup: CUDA is available
        mock_cuda_available.return_value = True
        mock_yolo_instance = MagicMock()
        mock_yolo.return_value = mock_yolo_instance

        # Run benchmark_model
        with patch.dict(os.environ, {"FORCE_SYNTHETIC": "true"}):
            benchmark_model("yolov8n.pt", imgsz=64, half=False, num_frames=3)

        self.assertTrue(mock_sync.called)
        self.assertGreaterEqual(mock_sync.call_count, 6)

if __name__ == '__main__':
    unittest.main()
