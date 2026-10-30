"""Render summary tables and metric charts from evaluation result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def load_results(result_dir: Path) -> pd.DataFrame:
    """Load per-configuration summary JSON files into a data frame."""
    rows: list[dict[str, Any]] = []
    for path in sorted(result_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"configuration": data["configuration"]["name"], **data["summary"]})
    return pd.DataFrame(rows)


def create_plots(result_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Save a CSV comparison table and a chart for available metrics."""
    frame = load_results(result_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "metrics.csv", index=False)
    available = [
        metric
        for metric in ("accuracy", "citation_precision", "citation_recall", "hallucinated_url_rate")
        if metric in frame and frame[metric].notna().any()
    ]
    if available:
        axis = frame.set_index("configuration")[available].plot(kind="bar", figsize=(10, 5))
        axis.set_ylim(0, 1)
        axis.set_ylabel("Score")
        axis.set_title("CiteCheck ablation results")
        axis.figure.tight_layout()
        axis.figure.savefig(output_dir / "metrics.png", dpi=160)
        plt.close(axis.figure)
    return frame


def main() -> None:
    """Create comparison outputs from a result directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    print(create_plots(args.results, args.output).to_string(index=False))


if __name__ == "__main__":
    main()
