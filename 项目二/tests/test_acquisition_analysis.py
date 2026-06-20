import tempfile
import unittest
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import acquisition_analysis  # noqa: E402


class AcquisitionAnalysisTests(unittest.TestCase):
    def test_read_timeseries_csv_parses_timestamp_and_strips_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "agc.csv"
            csv_path.write_text(
                "Server, Tagname, Value, Timestamp, Questionable, Annotated, Substituted\n"
                "10.141.43.1, 04:AGC01CCS, 166.1271, 2021/8/26 07:10:18, False, Not Annotated, False\n",
                encoding="utf-8",
            )

            df = acquisition_analysis.read_timeseries_csv(csv_path)

            self.assertEqual(list(df.columns), [
                "Server",
                "Tagname",
                "Value",
                "Timestamp",
                "Questionable",
                "Annotated",
                "Substituted",
            ])
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["Timestamp"]))
            self.assertAlmostEqual(df.loc[0, "Value"], 166.1271)

    def test_classify_metric_from_filename_recognizes_agc_and_load(self):
        self.assertEqual(acquisition_analysis.classify_metric_from_filename("AGC控制指令 361.csv"), "agc")
        self.assertEqual(acquisition_analysis.classify_metric_from_filename("#4发电机有功功率362.csv"), "load")

    def test_summarize_timeseries_frame_reports_bounds(self):
        df = pd.DataFrame(
            {
                "Server": ["10.141.43.1", "10.141.43.1"],
                "Tagname": ["04:AGC01CCS", "04:AGC01CCS"],
                "Value": [166.1, 166.5],
                "Timestamp": pd.to_datetime(["2021-08-26 07:10:18", "2021-08-26 07:20:18"]),
                "Questionable": [False, False],
                "Annotated": ["Not Annotated", "Not Annotated"],
                "Substituted": [False, False],
            }
        )

        summary = acquisition_analysis.summarize_timeseries_frame(df)

        self.assertEqual(summary["row_count"], 2)
        self.assertEqual(summary["tag_count"], 1)
        self.assertEqual(summary["start_time"], pd.Timestamp("2021-08-26 07:10:18"))
        self.assertEqual(summary["end_time"], pd.Timestamp("2021-08-26 07:20:18"))
        self.assertAlmostEqual(summary["value_summary"]["mean"], 166.3)

    def test_analyze_acquisition_directory_adds_metric_types(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "AGC控制指令 361.csv").write_text(
                "Server, Tagname, Value, Timestamp, Questionable, Annotated, Substituted\n"
                "10.141.43.1, 04:AGC01CCS, 166.1271, 2021/8/26 07:10:18, False, Not Annotated, False\n",
                encoding="utf-8",
            )

            results = acquisition_analysis.analyze_acquisition_directory(root)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["metric_type"], "agc")

    def test_generate_acquisition_report_creates_summary_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "AGC控制指令 361.csv").write_text(
                "Server, Tagname, Value, Timestamp, Questionable, Annotated, Substituted\n"
                "10.141.43.1, 04:AGC01CCS, 166.1271, 2021/8/26 07:10:18, False, Not Annotated, False\n"
                "10.141.43.1, 04:AGC01CCS, 166.2271, 2021/8/26 07:20:18, False, Not Annotated, False\n",
                encoding="utf-8",
            )
            (data_dir / "#4发电机有功功率362.csv").write_text(
                "Server, Tagname, Value, Timestamp, Questionable, Annotated, Substituted\n"
                "10.141.43.1, 04:4UGM4, 165.1475, 2021/8/26 07:10:18, False, Not Annotated, False\n"
                "10.141.43.1, 04:4UGM4, 165.3475, 2021/8/26 07:20:18, False, Not Annotated, False\n",
                encoding="utf-8",
            )
            output_dir = root / "out"

            outputs = acquisition_analysis.generate_acquisition_report(data_dir, output_dir)

            self.assertTrue(outputs["summary_json"].exists())
            self.assertTrue(outputs["metric_summary_csv"].exists())
            self.assertTrue(outputs["top_metrics_png"].exists())
            self.assertTrue(outputs["lag_correlation_csv"].exists())

    def test_generate_acquisition_report_handles_single_file_without_directory_expansion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_file = root / "AGC控制指令 361.csv"
            data_file.write_text(
                "Server, Tagname, Value, Timestamp, Questionable, Annotated, Substituted\n"
                "10.141.43.1, 04:AGC01CCS, 166.1271, 2021/8/26 07:10:18, False, Not Annotated, False\n",
                encoding="utf-8",
            )
            output_dir = root / "out"

            outputs = acquisition_analysis.analyze_acquisition_file_report(data_file, output_dir)

            self.assertTrue(outputs["summary_json"].exists())
            self.assertTrue(outputs["metric_summary_csv"].exists())


if __name__ == "__main__":
    unittest.main()
