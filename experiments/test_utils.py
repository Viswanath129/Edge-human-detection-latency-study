import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import os
import sys

# Insert experiments dir to sys.path to resolve local utility imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import save_summary, benchmark_model

class TestUtils(unittest.TestCase):

    @patch('os.path.exists')
    @patch('pandas.read_csv')
    @patch.object(pd.DataFrame, 'to_csv', autospec=True)
    @patch('os.makedirs')
    def test_save_summary_new_file(self, mock_makedirs, mock_to_csv, mock_read_csv, mock_exists):
        # Scenario: summary.csv does not exist, creating a brand new summary table
        mock_exists.return_value = False

        save_summary(640, "yolov8n", "FP32", 10.5, 95.0, "New Run")

        # Verify to_csv was called
        self.assertTrue(mock_to_csv.called)
        # Capture the dataframe passed to to_csv as the first positional argument (self)
        called_df = mock_to_csv.call_args[0][0]
        self.assertEqual(len(called_df), 1)
        self.assertEqual(called_df.iloc[0]['Resolution'], '640x640')
        self.assertEqual(called_df.iloc[0]['Model'], 'YOLOv8n')
        self.assertEqual(called_df.iloc[0]['Precision'], 'FP32')
        self.assertEqual(called_df.iloc[0]['Average_FPS'], 10.5)
        self.assertEqual(called_df.iloc[0]['Average_Latency_ms'], 95.0)

    @patch('os.path.exists')
    @patch('pandas.read_csv')
    @patch.object(pd.DataFrame, 'to_csv', autospec=True)
    @patch('os.makedirs')
    def test_save_summary_update_existing(self, mock_makedirs, mock_to_csv, mock_read_csv, mock_exists):
        # Scenario: summary.csv exists, and we are updating an existing entry
        mock_exists.return_value = True

        # Create mock existing data
        existing_df = pd.DataFrame({
            'Resolution': ['640x640', '416x416'],
            'Model': ['YOLOv8n', 'YOLOv8n'],
            'Precision': ['FP32', 'FP32'],
            'Average_FPS': [8.0, 15.0],
            'Average_Latency_ms': [120.0, 60.0],
            'Observation': ['Old Obs 1', 'Old Obs 2']
        })
        mock_read_csv.return_value = existing_df

        # Update YOLOv8n, FP32, 640x640
        save_summary(640, "yolov8n", "FP32", 9.0, 111.0, "Updated Obs")

        self.assertTrue(mock_to_csv.called)
        called_df = mock_to_csv.call_args[0][0]
        self.assertEqual(len(called_df), 2)
        # Check that index 0 is updated
        self.assertEqual(called_df.iloc[0]['Average_FPS'], 9.0)
        self.assertEqual(called_df.iloc[0]['Average_Latency_ms'], 111.0)
        self.assertEqual(called_df.iloc[0]['Observation'], "Updated Obs")
        # Index 1 remains unchanged
        self.assertEqual(called_df.iloc[1]['Average_FPS'], 15.0)

    @patch('os.path.exists')
    @patch('pandas.read_csv')
    @patch.object(pd.DataFrame, 'to_csv', autospec=True)
    @patch('os.makedirs')
    def test_save_summary_legacy_migration(self, mock_makedirs, mock_to_csv, mock_read_csv, mock_exists):
        # Scenario: summary.csv exists but lacks 'Model' and 'Precision' columns
        mock_exists.return_value = True

        legacy_df = pd.DataFrame({
            'Resolution': ['640x640', '416x416'],
            'Average_FPS': [7.6, 14.2],
            'Average_Latency_ms': [110.0, 65.0],
            'Observation': ['Higher quality', 'Faster']
        })
        mock_read_csv.return_value = legacy_df

        # Add another entry (e.g. YOLOv8s)
        save_summary(640, "yolov8s", "FP32", 5.0, 200.0, "Larger model")

        self.assertTrue(mock_to_csv.called)
        called_df = mock_to_csv.call_args[0][0]

        # The legacy rows must be populated with YOLOv8n and FP32
        self.assertEqual(called_df.iloc[0]['Model'], 'YOLOv8n')
        self.assertEqual(called_df.iloc[0]['Precision'], 'FP32')
        self.assertEqual(called_df.iloc[1]['Model'], 'YOLOv8n')
        self.assertEqual(called_df.iloc[1]['Precision'], 'FP32')

        # The new row is appended
        self.assertEqual(called_df.iloc[2]['Model'], 'YOLOv8s')
        self.assertEqual(called_df.iloc[2]['Precision'], 'FP32')
        self.assertEqual(called_df.iloc[2]['Average_FPS'], 5.0)

    @patch('utils.YOLO')
    @patch.dict(os.environ, {"FORCE_SYNTHETIC": "true"})
    def test_benchmark_model_synthetic_fallback(self, mock_yolo):
        # Scenario: FORCE_SYNTHETIC=true, verify we don't open webcam and return performance numbers

        # Mock YOLO model
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance
        # Mocking model inference call to return a list of results
        mock_model_instance.return_value = [MagicMock()]

        avg_latency, fps, actual_half = benchmark_model("yolov8n", 640, "FP32", num_frames=5)

        # Verify model was called with synthetic frame
        self.assertEqual(mock_model_instance.call_count, 10)  # 5 warmup + 5 main
        self.assertGreater(fps, 0)
        self.assertGreater(avg_latency, 0)
        self.assertFalse(actual_half)

    @patch('utils.YOLO')
    @patch.dict(os.environ, {"FORCE_SYNTHETIC": "true"})
    def test_benchmark_model_division_by_zero_hardening(self, mock_yolo):
        # Scenario: No frames are successfully processed, must return (0.0, 0.0, actual_half)

        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance
        # Make model inference raise RuntimeError to simulate failed processing / empty latencies
        mock_model_instance.side_effect = RuntimeError("Inference failed")

        avg_latency, fps, actual_half = benchmark_model("yolov8n", 640, "FP32", num_frames=5)

        self.assertEqual(avg_latency, 0.0)
        self.assertEqual(fps, 0.0)
        self.assertFalse(actual_half)

if __name__ == '__main__':
    unittest.main()
