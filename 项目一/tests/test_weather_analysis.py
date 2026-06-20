import tempfile
import unittest
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import weather_analysis  # noqa: E402


class WeatherAnalysisTests(unittest.TestCase):
    def test_read_weather_csv_normalizes_columns_and_datetime(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "weather.csv"
            csv_path.write_text(
                "StationID,StationCode,StationName,DataTime,WindAvgSpeed,Temperature,City\n"
                "id1,54629,青岛站,2019-01-01 00:00,3.2,-4.6,青岛\n",
                encoding="utf-8",
            )

            df = weather_analysis.read_weather_csv(csv_path)
            normalized = weather_analysis.normalize_weather_frame(df)

            self.assertEqual(list(normalized.columns), [
                "StationID",
                "StationCode",
                "StationName",
                "DataTime",
                "WindAvgSpeed",
                "Temperature",
                "City",
            ])
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(normalized["DataTime"]))
            self.assertEqual(normalized.loc[0, "StationName"], "青岛站")

    def test_summarize_weather_frame_computes_basic_stats(self):
        df = pd.DataFrame(
            {
                "StationCode": ["54629", "54629", "54658"],
                "DataTime": pd.to_datetime(["2019-01-01 00:00", "2019-01-01 01:00", "2019-01-02 00:00"]),
                "WindAvgSpeed": [1.0, 3.0, 5.0],
                "Temperature": [-4.0, -2.0, 1.0],
                "City": ["青岛", "青岛", "烟台"],
            }
        )

        summary = weather_analysis.summarize_weather_frame(df)

        self.assertEqual(summary["row_count"], 3)
        self.assertEqual(summary["station_count"], 2)
        self.assertEqual(summary["city_count"], 2)
        self.assertAlmostEqual(summary["numeric_summary"]["WindAvgSpeed"]["mean"], 3.0)
        self.assertAlmostEqual(summary["numeric_summary"]["Temperature"]["max"], 1.0)

    def test_analyze_weather_directory_collects_all_csv_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a.csv").write_text(
                "StationID,StationCode,StationName,DataTime,WindAvgSpeed,Temperature,City\n"
                "id1,54629,青岛站,2019-01-01 00:00,3.2,-4.6,青岛\n",
                encoding="utf-8",
            )
            (root / "b.csv").write_text(
                "StationID,StationCode,StationName,DataTime,WindAvgSpeed,Temperature,City\n"
                "id2,54658,烟台站,2019-01-01 01:00,5.2,-2.6,烟台\n",
                encoding="utf-8",
            )

            results = weather_analysis.analyze_weather_directory(root)

            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["summary"]["row_count"], 1)

    def test_generate_weather_report_writes_summary_and_figures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "201901.csv").write_text(
                "StationID,StationCode,StationName,DataTime,WindAvgSpeed,Temperature,City\n"
                "id1,54629,青岛站,2019-01-01 00:00,3.2,-4.6,青岛\n"
                "id2,54629,青岛站,2019-01-01 01:00,4.2,-3.6,青岛\n"
                "id3,54658,烟台站,2019-01-01 00:00,5.2,-2.6,烟台\n",
                encoding="utf-8",
            )
            output_dir = root / "out"

            outputs = weather_analysis.generate_weather_report(data_dir, output_dir)

            self.assertTrue(outputs["summary_json"].exists())
            self.assertTrue(outputs["city_summary_csv"].exists())
            self.assertTrue(outputs["temperature_trend_png"].exists())
            self.assertTrue(outputs["correlation_heatmap_png"].exists())

    def test_generate_weather_report_handles_missing_numeric_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "minimal.csv").write_text(
                "StationID,StationCode,StationName,DataTime,City\n"
                "id1,54629,青岛站,2019-01-01 00:00,青岛\n",
                encoding="utf-8",
            )
            output_dir = root / "out"

            outputs = weather_analysis.generate_weather_report(data_dir, output_dir)

            self.assertTrue(outputs["summary_json"].exists())
            self.assertTrue(outputs["city_summary_csv"].exists())


if __name__ == "__main__":
    unittest.main()
