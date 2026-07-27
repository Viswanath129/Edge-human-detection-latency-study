import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

# Insert containing directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import save_summary, benchmark_model

class TestUtils(unittest.TestCase):
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('pandas.read_csv')
    @patch.object(pd.DataFrame, 'to_csv', autospec=True)
    def test_save_summary_new_file(self, mock_to_csv, mock_read_csv, mock_makedirs, mock_exists):
        # Scenario 1: summary.csv does not exist, creating new file
        mock_exists.return_value = False

        save_summary(640, 'yolov8n.pt', 'FP32', 15.5, 65.0, 'Test observation')

        # Verify makedirs and to_csv were called
        mock_makedirs.assert_called_once()
        self.assertTrue(mock_to_csv.called)

        # Capture the dataframe passed to to_csv
        # autospec=True means mock_to_csv(self_df, path, ...)
        called_df = mock_to_csv.call_args[0][0]
        self.assertEqual(len(called_df), 1)
        self.assertEqual(called_df.iloc[0]['Resolution'], '640x640')
        self.assertEqual(called_df.iloc[0]['Model'], 'YOLOv8n')
        self.assertEqual(called_df.iloc[0]['Precision'], 'FP32')
        self.assertEqual(called_df.iloc[0]['Average_FPS'], 15.5)
        self.assertEqual(called_df.iloc[0]['Average_Latency_ms'], 65.0)

    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('pandas.read_csv')
    @patch.object(pd.DataFrame, 'to_csv', autospec=True)
    def test_save_summary_existing_update(self, mock_to_csv, mock_read_csv, mock_makedirs, mock_exists):
        # Scenario 2: summary.csv exists, with existing entries. Update matching entry.
        mock_exists.return_value = True

        # Return a mock df with matching entries
        existing_df = pd.DataFrame({
            'Resolution': ['640x640', '416x416'],
            'Model': ['YOLOv8n', 'YOLOv8n'],
            'Precision': ['FP32', 'FP32'],
            'Average_FPS': [10.0, 20.0],
            'Average_Latency_ms': [100.0, 50.0],
            'Observation': ['Old', 'Old']
        })
        mock_read_csv.return_value = existing_df

        save_summary(640, 'yolov8n.pt', 'FP32', 12.0, 83.3, 'Updated observation')

        called_df = mock_to_csv.call_args[0][0]
        self.assertEqual(len(called_df), 2)
        # First row (the 640x640 one) should be updated
        row640 = called_df[called_df['Resolution'] == '640x640'].iloc[0]
        self.assertEqual(row640['Average_FPS'], 12.0)
        self.assertEqual(row640['Average_Latency_ms'], 83.3)
        self.assertEqual(row640['Observation'], 'Updated observation')

    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('pandas.read_csv')
    @patch.object(pd.DataFrame, 'to_csv', autospec=True)
    def test_save_summary_legacy_migration(self, mock_to_csv, mock_read_csv, mock_makedirs, mock_exists):
        # Scenario 3: summary.csv exists but lacks Model and Precision columns
        mock_exists.return_value = True

        existing_df = pd.DataFrame({
            'Resolution': ['640x640'],
            'Average_FPS': [10.0],
            'Average_Latency_ms': [100.0],
            'Observation': ['Old']
        })
        mock_read_csv.return_value = existing_df

        save_summary(416, 'yolov8n.pt', 'FP32', 20.0, 50.0, 'New row')

        called_df = mock_to_csv.call_args[0][0]
        # Columns Model and Precision must have been populated with defaults
        row640 = called_df[called_df['Resolution'] == '640x640'].iloc[0]
        self.assertEqual(row640['Model'], 'YOLOv8n')
        self.assertEqual(row640['Precision'], 'FP32')

    @patch.dict(os.environ, {"FORCE_SYNTHETIC": "true"})
    @patch('cv2.VideoCapture')
    @patch('utils.YOLO')
    def test_benchmark_model_synthetic(self, mock_yolo, mock_videocapture):
        # Mock YOLO instance
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance

        # Run benchmark with small num_frames
        avg_latency, fps, actual_half = benchmark_model('yolov8n.pt', 640, 'FP32', num_frames=5)

        # Verify results
        self.assertIsInstance(avg_latency, float)
        self.assertIsInstance(fps, float)
        self.assertFalse(actual_half)
        # Check that YOLO was called
        self.assertTrue(mock_model_instance.called)

if __name__ == '__main__':
    unittest.main()
