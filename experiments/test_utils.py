import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Add experiments/ directory to sys.path to import utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import save_summary, benchmark_model

class TestUtils(unittest.TestCase):

    @patch('os.path.exists')
    @patch('pandas.read_csv')
    @patch.object(pd.DataFrame, 'to_csv', autospec=True)
    def test_save_summary_new(self, mock_to_csv, mock_read_csv, mock_exists):
        """
        Tests save_summary when summary.csv does not exist.
        """
        mock_exists.return_value = False

        # Run save_summary
        save_summary(640, "yolov8n", "FP32", 15.0, 66.0, "Test New Row")

        # Verify to_csv was called
        self.assertTrue(mock_to_csv.called)

        # Get the DataFrame instance that called to_csv
        df_passed = mock_to_csv.call_args[0][0]
        self.assertEqual(len(df_passed), 1)
        self.assertEqual(df_passed.iloc[0]['Resolution'], '640x640')
        self.assertEqual(df_passed.iloc[0]['Model'], 'YOLOv8n')
        self.assertEqual(df_passed.iloc[0]['Precision'], 'FP32')
        self.assertEqual(df_passed.iloc[0]['Average_FPS'], 15.0)
        self.assertEqual(df_passed.iloc[0]['Average_Latency_ms'], 66.0)
        self.assertEqual(df_passed.iloc[0]['Observation'], "Test New Row")

    @patch('os.path.exists')
    @patch('pandas.read_csv')
    @patch.object(pd.DataFrame, 'to_csv', autospec=True)
    def test_save_summary_migration(self, mock_to_csv, mock_read_csv, mock_exists):
        """
        Tests save_summary when existing summary.csv is missing Model and Precision columns (legacy schema).
        """
        mock_exists.return_value = True

        # Create a df without Model and Precision
        legacy_df = pd.DataFrame({
            "Resolution": ["640x640"],
            "Average_FPS": [7.6],
            "Average_Latency_ms": [110.0],
            "Observation": ["Legacy observation"]
        })
        mock_read_csv.return_value = legacy_df

        # Save a new summary entry (which should also trigger schema migration on the existing one)
        save_summary(416, "yolov8s", "FP16", 15.0, 60.0, "New observation")

        self.assertTrue(mock_to_csv.called)
        df_passed = mock_to_csv.call_args[0][0]

        # Should have updated/migrated columns
        self.assertIn("Model", df_passed.columns)
        self.assertIn("Precision", df_passed.columns)

        # Existing row should be migrated
        self.assertEqual(df_passed.iloc[0]['Model'], 'YOLOv8n')
        self.assertEqual(df_passed.iloc[0]['Precision'], 'FP32')

        # New row should be appended correctly
        self.assertEqual(df_passed.iloc[1]['Resolution'], '416x416')
        self.assertEqual(df_passed.iloc[1]['Model'], 'YOLOv8s')
        self.assertEqual(df_passed.iloc[1]['Precision'], 'FP16')

    @patch('os.path.exists')
    @patch('pandas.read_csv')
    @patch.object(pd.DataFrame, 'to_csv', autospec=True)
    def test_save_summary_update_existing(self, mock_to_csv, mock_read_csv, mock_exists):
        """
        Tests save_summary when the same Resolution, Model, and Precision already exist.
        It should update the entry in-place instead of appending.
        """
        mock_exists.return_value = True

        existing_df = pd.DataFrame({
            "Resolution": ["640x640", "416x416"],
            "Model": ["YOLOv8n", "YOLOv8n"],
            "Precision": ["FP32", "FP32"],
            "Average_FPS": [9.0, 19.0],
            "Average_Latency_ms": [110.0, 52.0],
            "Observation": ["Old Obs 1", "Old Obs 2"]
        })
        mock_read_csv.return_value = existing_df

        # Update the 640x640 YOLOv8n FP32 entry
        save_summary(640, "yolov8n", "FP32", 9.5, 105.0, "Updated Obs 1")

        df_passed = mock_to_csv.call_args[0][0]
        self.assertEqual(len(df_passed), 2)
        # Check first entry updated in place
        self.assertEqual(df_passed.iloc[0]['Average_FPS'], 9.5)
        self.assertEqual(df_passed.iloc[0]['Average_Latency_ms'], 105.0)
        self.assertEqual(df_passed.iloc[0]['Observation'], "Updated Obs 1")
        # Check second entry untouched
        self.assertEqual(df_passed.iloc[1]['Average_FPS'], 19.0)

    @patch('utils.YOLO')
    @patch('cv2.VideoCapture')
    @patch('os.environ.get')
    def test_benchmark_model_synthetic(self, mock_env_get, mock_video_capture, mock_yolo_class):
        """
        Tests benchmark_model with synthetic option enabled.
        """
        mock_env_get.return_value = "true"
        mock_model_instance = MagicMock()
        mock_yolo_class.return_value = mock_model_instance

        # Mocking inference results
        mock_result = MagicMock()
        mock_model_instance.return_value = [mock_result]

        avg_latency, fps, actual_half = benchmark_model("yolov8n.pt", 640, "FP32", num_frames=10)

        self.assertFalse(mock_video_capture.called)
        self.assertGreater(fps, 0)
        self.assertFalse(actual_half)

    @patch('utils.YOLO')
    def test_benchmark_model_zero_frames(self, mock_yolo_class):
        """
        Tests benchmark_model when num_frames is 0.
        """
        avg_latency, fps, actual_half = benchmark_model("yolov8n.pt", 640, "FP32", num_frames=0)
        self.assertEqual(avg_latency, 0.0)
        self.assertEqual(fps, 0.0)
        self.assertFalse(actual_half)

    @patch('utils.YOLO')
    @patch('cv2.VideoCapture')
    def test_benchmark_model_webcam_fail(self, mock_video_capture, mock_yolo_class):
        """
        Tests benchmark_model when webcam fails to open, verifying synthetic fallback.
        """
        mock_model_instance = MagicMock()
        mock_yolo_class.return_value = mock_model_instance

        # Webcam fails to open
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap

        avg_latency, fps, actual_half = benchmark_model("yolov8n.pt", 640, "FP32", num_frames=5)
        self.assertTrue(mock_video_capture.called)
        # Should still run successfully using synthetic fallback
        self.assertGreater(fps, 0)

if __name__ == '__main__':
    unittest.main()
