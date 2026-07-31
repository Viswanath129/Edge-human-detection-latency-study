import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
import sys

# Ensure experiments directory is in the PYTHONPATH/sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import save_summary, benchmark_model

class TestUtils(unittest.TestCase):

    @patch('os.path.exists')
    @patch('pandas.read_csv')
    @patch.object(pd.DataFrame, 'to_csv', autospec=True)
    def test_save_summary_new_file(self, mock_to_csv, mock_read_csv, mock_exists):
        """
        Test save_summary when the summary.csv does not exist yet.
        """
        mock_exists.return_value = False

        # Invoke save_summary
        save_summary(640, "yolov8n.pt", "FP32", 15.5, 64.5, "Test New Obs")

        # Verify to_csv was called on the DataFrame
        self.assertTrue(mock_to_csv.called)

        # First positional argument to the mock should be the DataFrame instance itself
        df_instance = mock_to_csv.call_args[0][0]
        self.assertIsInstance(df_instance, pd.DataFrame)
        self.assertEqual(len(df_instance), 1)
        self.assertEqual(df_instance.iloc[0]["Resolution"], "640x640")
        self.assertEqual(df_instance.iloc[0]["Model"], "YOLOv8n")
        self.assertEqual(df_instance.iloc[0]["Precision"], "FP32")
        self.assertEqual(df_instance.iloc[0]["Average_FPS"], 15.5)
        self.assertEqual(df_instance.iloc[0]["Average_Latency_ms"], 64.5)
        self.assertEqual(df_instance.iloc[0]["Observation"], "Test New Obs")

    @patch('os.path.exists')
    @patch('pandas.read_csv')
    @patch.object(pd.DataFrame, 'to_csv', autospec=True)
    def test_save_summary_existing_update(self, mock_to_csv, mock_read_csv, mock_exists):
        """
        Test save_summary when matching entry exists in summary.csv and is updated.
        """
        mock_exists.return_value = True
        existing_df = pd.DataFrame([
            {
                "Resolution": "640x640",
                "Model": "YOLOv8n",
                "Precision": "FP32",
                "Average_FPS": 10.0,
                "Average_Latency_ms": 100.0,
                "Observation": "Old Obs"
            }
        ])
        mock_read_csv.return_value = existing_df

        save_summary(640, "yolov8n", "FP32", 12.0, 80.0, "Updated Obs")

        self.assertTrue(mock_to_csv.called)
        df_instance = mock_to_csv.call_args[0][0]
        self.assertEqual(len(df_instance), 1)  # updated, not appended
        self.assertEqual(df_instance.iloc[0]["Average_FPS"], 12.0)
        self.assertEqual(df_instance.iloc[0]["Average_Latency_ms"], 80.0)
        self.assertEqual(df_instance.iloc[0]["Observation"], "Updated Obs")

    @patch('os.path.exists')
    @patch('pandas.read_csv')
    @patch.object(pd.DataFrame, 'to_csv', autospec=True)
    def test_save_summary_existing_append(self, mock_to_csv, mock_read_csv, mock_exists):
        """
        Test save_summary when a non-matching entry exists and a new row is appended.
        """
        mock_exists.return_value = True
        existing_df = pd.DataFrame([
            {
                "Resolution": "640x640",
                "Model": "YOLOv8n",
                "Precision": "FP32",
                "Average_FPS": 10.0,
                "Average_Latency_ms": 100.0,
                "Observation": "Old Obs"
            }
        ])
        mock_read_csv.return_value = existing_df

        save_summary(416, "yolov8n", "FP32", 20.0, 50.0, "New Row Obs")

        self.assertTrue(mock_to_csv.called)
        df_instance = mock_to_csv.call_args[0][0]
        self.assertEqual(len(df_instance), 2)  # appended, so 2 rows
        row_416 = df_instance[df_instance["Resolution"] == "416x416"].iloc[0]
        self.assertEqual(row_416["Model"], "YOLOv8n")
        self.assertEqual(row_416["Average_FPS"], 20.0)
        self.assertEqual(row_416["Average_Latency_ms"], 50.0)

    @patch('os.path.exists')
    @patch('pandas.read_csv')
    @patch.object(pd.DataFrame, 'to_csv', autospec=True)
    def test_save_summary_schema_migration(self, mock_to_csv, mock_read_csv, mock_exists):
        """
        Test save_summary schema migration when existing CSV is missing Model or Precision columns.
        """
        mock_exists.return_value = True
        # Missing Model and Precision columns
        existing_df = pd.DataFrame([
            {
                "Resolution": "640x640",
                "Average_FPS": 7.6,
                "Average_Latency_ms": 110.0,
                "Observation": "Legacy Obs"
            }
        ])
        mock_read_csv.return_value = existing_df

        save_summary(416, "yolov8n", "FP32", 14.2, 65.0, "New Obs")

        self.assertTrue(mock_to_csv.called)
        df_instance = mock_to_csv.call_args[0][0]

        # Legacy row must have defaults populated
        row_640 = df_instance[df_instance["Resolution"] == "640x640"].iloc[0]
        self.assertEqual(row_640["Model"], "YOLOv8n")
        self.assertEqual(row_640["Precision"], "FP32")

        # New row must be appended correctly
        row_416 = df_instance[df_instance["Resolution"] == "416x416"].iloc[0]
        self.assertEqual(row_416["Model"], "YOLOv8n")
        self.assertEqual(row_416["Precision"], "FP32")
        self.assertEqual(row_416["Average_FPS"], 14.2)

    @patch('utils.YOLO')
    def test_benchmark_model_synthetic(self, mock_yolo):
        """
        Test benchmark_model runs successfully with synthetic frames when forced.
        """
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance

        with patch.dict(os.environ, {"FORCE_SYNTHETIC": "true"}):
            avg_lat, fps, actual_half = benchmark_model("yolov8n.pt", 640, "FP32", num_frames=5)

        self.assertTrue(mock_model_instance.called)
        self.assertIsInstance(avg_lat, float)
        self.assertIsInstance(fps, float)
        self.assertFalse(actual_half)

    @patch('utils.YOLO')
    def test_benchmark_model_division_by_zero_hardening(self, mock_yolo):
        """
        Test benchmark_model returns (0.0, 0.0, actual_half) safely if no frames can be processed.
        """
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance

        # Force synthetic to fail or have 0 frames by overriding num_frames to 0
        with patch.dict(os.environ, {"FORCE_SYNTHETIC": "true"}):
            avg_lat, fps, actual_half = benchmark_model("yolov8n.pt", 640, "FP32", num_frames=0)

        self.assertEqual(avg_lat, 0.0)
        self.assertEqual(fps, 0.0)
        self.assertFalse(actual_half)

if __name__ == "__main__":
    unittest.main()
