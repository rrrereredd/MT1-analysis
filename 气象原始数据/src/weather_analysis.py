from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_DATETIME_COLUMN = "DataTime"


def read_weather_csv(path: str | Path) -> pd.DataFrame:
    """Read a weather CSV file with a forgiving encoding fallback."""
    path = Path(path)
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def normalize_weather_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Strip column whitespace and parse the main datetime column."""
    normalized = df.copy()
    normalized.columns = [str(col).strip() for col in normalized.columns]

    if DEFAULT_DATETIME_COLUMN in normalized.columns:
        normalized[DEFAULT_DATETIME_COLUMN] = pd.to_datetime(
            normalized[DEFAULT_DATETIME_COLUMN], errors="coerce"
        )

    return normalized


def summarize_weather_frame(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize a normalized weather table for reporting."""
    normalized = normalize_weather_frame(df)

    numeric_columns = normalized.select_dtypes(include="number").columns.tolist()
    numeric_summary: dict[str, dict[str, float]] = {}
    if numeric_columns:
        desc = normalized[numeric_columns].describe().to_dict()
        for column, stats in desc.items():
            numeric_summary[column] = {
                key: float(value) for key, value in stats.items() if pd.notna(value)
            }

    return {
        "row_count": int(len(normalized)),
        "station_count": int(normalized["StationCode"].nunique()) if "StationCode" in normalized else 0,
        "city_count": int(normalized["City"].nunique()) if "City" in normalized else 0,
        "start_time": normalized[DEFAULT_DATETIME_COLUMN].min() if DEFAULT_DATETIME_COLUMN in normalized else None,
        "end_time": normalized[DEFAULT_DATETIME_COLUMN].max() if DEFAULT_DATETIME_COLUMN in normalized else None,
        "numeric_summary": numeric_summary,
    }


def analyze_weather_file(path: str | Path) -> dict[str, Any]:
    """Convenience wrapper for a single weather CSV file."""
    return summarize_weather_frame(read_weather_csv(path))


def analyze_weather_directory(directory: str | Path) -> list[dict[str, Any]]:
    """Analyze all CSV files in a weather data directory."""
    directory = Path(directory)
    results: list[dict[str, Any]] = []
    for csv_path in sorted(directory.rglob("*.csv")):
        results.append(
            {
                "file": str(csv_path),
                "summary": analyze_weather_file(csv_path),
            }
        )
    return results


def save_summary(summary: Any, output_path: str | Path) -> None:
    """Persist a summary payload as formatted JSON."""
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)


def collect_weather_tables(data_dir: str | Path) -> pd.DataFrame:
    """Load all weather CSV files under a directory into a single table."""
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(Path(data_dir).rglob("*.csv")):
        frame = read_weather_csv(csv_path)
        frame = normalize_weather_frame(frame)
        frame["SourceFile"] = csv_path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_city_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate simple city-level weather statistics."""
    if df.empty or "City" not in df.columns:
        return pd.DataFrame()

    numeric_cols = [col for col in ["WindAvgSpeed", "Temperature", "Humidity", "Visibility", "Rain"] if col in df.columns]
    if not numeric_cols:
        return df[["City"]].drop_duplicates().reset_index(drop=True)
    agg_map: dict[str, list[str]] = {col: ["mean", "min", "max"] for col in numeric_cols}
    summary = df.groupby("City", dropna=False).agg(agg_map)
    summary.columns = [f"{left}_{right}" for left, right in summary.columns]
    return summary.reset_index()


def build_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Create a correlation matrix from available numeric weather fields."""
    numeric_cols = [col for col in ["WindAvgSpeed", "WindMaxSpeed", "Temperature", "Humidity", "Visibility", "Rain"] if col in df.columns]
    if len(numeric_cols) < 2:
        return pd.DataFrame()
    return df[numeric_cols].corr(numeric_only=True)


def plot_temperature_trend(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot the average temperature trend by day."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty or DEFAULT_DATETIME_COLUMN not in df.columns or "Temperature" not in df.columns:
        output_path.touch()
        return output_path

    plot_df = df.dropna(subset=[DEFAULT_DATETIME_COLUMN]).copy()
    plot_df["Day"] = plot_df[DEFAULT_DATETIME_COLUMN].dt.date
    daily = plot_df.groupby("Day", as_index=False)["Temperature"].mean()

    plt.figure(figsize=(10, 4))
    sns.lineplot(data=daily, x="Day", y="Temperature", marker="o")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_correlation_heatmap(corr: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot a correlation heatmap."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if corr.empty:
        output_path.touch()
        return output_path

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def generate_weather_report(data_dir: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Generate report artifacts for a weather data directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = collect_weather_tables(data_dir)
    overall_summary = summarize_weather_frame(table) if not table.empty else {
        "row_count": 0,
        "station_count": 0,
        "city_count": 0,
        "start_time": None,
        "end_time": None,
        "numeric_summary": {},
    }
    city_summary = build_city_summary(table)
    corr = build_correlation_matrix(table)

    summary_json = output_dir / "weather_summary.json"
    city_summary_csv = output_dir / "weather_city_summary.csv"
    temperature_trend_png = output_dir / "temperature_trend.png"
    correlation_heatmap_png = output_dir / "correlation_heatmap.png"

    save_summary(overall_summary, summary_json)
    city_summary.to_csv(city_summary_csv, index=False, encoding="utf-8-sig")
    plot_temperature_trend(table, temperature_trend_png)
    plot_correlation_heatmap(corr, correlation_heatmap_png)

    return {
        "summary_json": summary_json,
        "city_summary_csv": city_summary_csv,
        "temperature_trend_png": temperature_trend_png,
        "correlation_heatmap_png": correlation_heatmap_png,
    }


def analyze_weather_file_report(path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Convenience wrapper for generating artifacts from one file or directory."""
    path = Path(path)
    if path.is_dir():
        return generate_weather_report(path, output_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = normalize_weather_frame(read_weather_csv(path))
    overall_summary = summarize_weather_frame(table)
    city_summary = build_city_summary(table)
    corr = build_correlation_matrix(table)

    summary_json = output_dir / "weather_summary.json"
    city_summary_csv = output_dir / "weather_city_summary.csv"
    temperature_trend_png = output_dir / "temperature_trend.png"
    correlation_heatmap_png = output_dir / "correlation_heatmap.png"

    save_summary(overall_summary, summary_json)
    city_summary.to_csv(city_summary_csv, index=False, encoding="utf-8-sig")
    plot_temperature_trend(table, temperature_trend_png)
    plot_correlation_heatmap(corr, correlation_heatmap_png)

    return {
        "summary_json": summary_json,
        "city_summary_csv": city_summary_csv,
        "temperature_trend_png": temperature_trend_png,
        "correlation_heatmap_png": correlation_heatmap_png,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Analyze weather raw data")
    parser.add_argument("input", help="Path to a weather CSV file or directory")
    parser.add_argument("--output", help="Optional path to save summary JSON")
    parser.add_argument("--report-dir", help="Generate a full report into this directory")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if args.report_dir:
        result = analyze_weather_file_report(input_path, args.report_dir)
    elif input_path.is_dir():
        result = analyze_weather_directory(input_path)
    else:
        result = analyze_weather_file(input_path)
    print(result)

    if args.output:
        save_summary(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
