from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def read_timeseries_csv(path: str | Path) -> pd.DataFrame:
    """Read a process timeseries CSV and normalize its columns."""
    path = Path(path)
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="gbk")

    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    if "Value" in df.columns:
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    return df


def classify_metric_from_filename(filename: str) -> str:
    """Classify a CSV file by its filename for a coarse analysis bucket."""
    name = Path(filename).name
    lowered = name.lower()
    if "AGC" in name or "agc" in lowered or "负荷指令" in name:
        return "agc"
    if "发电机有功功率" in name or "有功功率" in name or "机组实际负荷" in name or "机组负荷" in name:
        return "load"
    if "二次风" in name:
        return "secondary_air"
    if "一次风" in name:
        return "primary_air"
    if "煤粉浓度" in name:
        return "coal_concentration"
    if "阀" in name or "调门" in name:
        return "valve"
    return "other"


def normalize_timeseries_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw acquisition dataframe into a standard schema."""
    frame = df.copy()
    frame.columns = [str(col).strip() for col in frame.columns]
    if "Timestamp" in frame.columns:
        frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="coerce")
    if "Value" in frame.columns:
        frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")
    return frame


def summarize_timeseries_frame(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize a normalized acquisition timeseries table."""
    frame = normalize_timeseries_frame(df)

    value_summary: dict[str, float] = {}
    if "Value" in frame.columns:
        desc = frame["Value"].describe()
        value_summary = {key: float(value) for key, value in desc.items() if pd.notna(value)}

    return {
        "row_count": int(len(frame)),
        "tag_count": int(frame["Tagname"].nunique()) if "Tagname" in frame.columns else 0,
        "start_time": frame["Timestamp"].min() if "Timestamp" in frame.columns else None,
        "end_time": frame["Timestamp"].max() if "Timestamp" in frame.columns else None,
        "value_summary": value_summary,
    }


def analyze_timeseries_file(path: str | Path) -> dict[str, Any]:
    """Convenience wrapper for one process CSV file."""
    return summarize_timeseries_frame(read_timeseries_csv(path))


def analyze_acquisition_directory(directory: str | Path) -> list[dict[str, Any]]:
    """Analyze all process CSV files in a directory."""
    directory = Path(directory)
    results: list[dict[str, Any]] = []
    for csv_path in sorted(directory.rglob("*.csv")):
        results.append(
            {
                "file": str(csv_path),
                "metric_type": classify_metric_from_filename(csv_path.name),
                "summary": analyze_timeseries_file(csv_path),
            }
        )
    return results


def save_summary(summary: Any, output_path: str | Path) -> None:
    """Persist a summary dictionary as formatted JSON."""
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)


def collect_acquisition_tables(data_dir: str | Path) -> pd.DataFrame:
    """Load all acquisition CSV files into one table with metadata."""
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(Path(data_dir).rglob("*.csv")):
        frame = read_timeseries_csv(csv_path)
        frame["SourceFile"] = csv_path.name
        frame["MetricType"] = classify_metric_from_filename(csv_path.name)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_metric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rows and basic value stats by metric type."""
    if df.empty or "MetricType" not in df.columns:
        return pd.DataFrame()

    grouped = (
        df.groupby("MetricType", dropna=False)
        .agg(
            row_count=("Value", "size"),
            value_mean=("Value", "mean"),
            value_min=("Value", "min"),
            value_max=("Value", "max"),
            timestamp_start=("Timestamp", "min"),
            timestamp_end=("Timestamp", "max"),
        )
        .reset_index()
    )
    return grouped.sort_values("row_count", ascending=False)


def compute_lag_correlation(df: pd.DataFrame, filename_a: str, filename_b: str, max_lag: int = 12) -> pd.DataFrame:
    """Compute lag correlation between two files if both are present."""
    if df.empty:
        return pd.DataFrame()

    a = df[df["SourceFile"] == filename_a].sort_values("Timestamp")
    b = df[df["SourceFile"] == filename_b].sort_values("Timestamp")
    if a.empty or b.empty:
        return pd.DataFrame()

    merged = pd.merge_asof(
        a[["Timestamp", "Value"]],
        b[["Timestamp", "Value"]],
        on="Timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("5min"),
        suffixes=("_a", "_b"),
    ).dropna()
    if merged.empty:
        return pd.DataFrame()

    rows = []
    for lag in range(-max_lag, max_lag + 1):
        shifted = merged.copy()
        shifted["Value_b"] = shifted["Value_b"].shift(lag)
        usable = shifted[["Value_a", "Value_b"]].dropna()
        corr = usable["Value_a"].corr(usable["Value_b"]) if len(usable) >= 2 else float("nan")
        rows.append({"lag": lag, "correlation": corr})
    return pd.DataFrame(rows)


def plot_top_metrics(metric_summary: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot the most common metric categories."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if metric_summary.empty:
        output_path.touch()
        return output_path

    plot_df = metric_summary.head(10)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot_df, x="MetricType", y="row_count", color="#4c72b0")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_lag_correlation(lag_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot lag correlation curve."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if lag_df.empty:
        output_path.touch()
        return output_path

    plt.figure(figsize=(10, 4))
    sns.lineplot(data=lag_df, x="lag", y="correlation", marker="o")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def generate_acquisition_report(data_dir: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Generate report artifacts for acquisition raw data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = collect_acquisition_tables(data_dir)
    overall_summary = {
        "row_count": int(len(table)),
        "file_count": int(table["SourceFile"].nunique()) if "SourceFile" in table.columns else 0,
        "metric_type_count": int(table["MetricType"].nunique()) if "MetricType" in table.columns else 0,
        "start_time": table["Timestamp"].min() if "Timestamp" in table.columns else None,
        "end_time": table["Timestamp"].max() if "Timestamp" in table.columns else None,
    }

    metric_summary = build_metric_summary(table)
    agc_candidates: list[str] = []
    load_candidates: list[str] = []
    if "MetricType" in table.columns and "SourceFile" in table.columns:
        agc_candidates = table.loc[table["MetricType"] == "agc", "SourceFile"].dropna().astype(str).tolist()
        load_candidates = table.loc[table["MetricType"] == "load", "SourceFile"].dropna().astype(str).tolist()
    preferred_agc = agc_candidates[0] if agc_candidates else "AGC控制指令 361.csv"
    preferred_load = load_candidates[0] if load_candidates else "#4发电机有功功率362.csv"
    lag_df = compute_lag_correlation(
        table,
        preferred_agc,
        preferred_load,
        max_lag=12,
    )

    summary_json = output_dir / "acquisition_summary.json"
    metric_summary_csv = output_dir / "metric_summary.csv"
    top_metrics_png = output_dir / "top_metrics.png"
    lag_correlation_csv = output_dir / "lag_correlation.csv"
    lag_correlation_png = output_dir / "lag_correlation.png"

    save_summary(overall_summary, summary_json)
    metric_summary.to_csv(metric_summary_csv, index=False, encoding="utf-8-sig")
    lag_df.to_csv(lag_correlation_csv, index=False, encoding="utf-8-sig")
    plot_top_metrics(metric_summary, top_metrics_png)
    plot_lag_correlation(lag_df, lag_correlation_png)

    return {
        "summary_json": summary_json,
        "metric_summary_csv": metric_summary_csv,
        "top_metrics_png": top_metrics_png,
        "lag_correlation_csv": lag_correlation_csv,
        "lag_correlation_png": lag_correlation_png,
    }


def analyze_acquisition_file_report(path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Convenience wrapper for generating artifacts from one file or directory."""
    path = Path(path)
    if path.is_dir():
        return generate_acquisition_report(path, output_dir)
    return generate_acquisition_report(path.parent, output_dir)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Analyze acquisition raw data")
    parser.add_argument("input", help="Path to a process CSV file or directory")
    parser.add_argument("--output", help="Optional path to save summary JSON")
    parser.add_argument("--report-dir", help="Generate a full report into this directory")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if args.report_dir:
        result = analyze_acquisition_file_report(input_path, args.report_dir)
    elif input_path.is_dir():
        result = analyze_acquisition_directory(input_path)
    else:
        result = analyze_timeseries_file(input_path)
    print(result)

    if args.output:
        save_summary(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
