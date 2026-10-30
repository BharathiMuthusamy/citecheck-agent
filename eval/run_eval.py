"""Run the research question dataset under one or more ablation settings."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from citecheck.agent import run_agent
from citecheck.config import Settings
from eval.ablations import CONFIGURATIONS
from eval.metrics import aggregate_metrics, citation_metrics

ROOT = Path(__file__).resolve().parents[1]


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Read JSONL evaluation prompts and validate required fields."""
    entries = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if any(not entry.get("id") or not entry.get("question") for entry in entries):
        raise ValueError("Every evaluation row must contain id and question")
    return entries


def run_evaluation(
    dataset: list[dict[str, Any]],
    configurations: list[dict[str, Any]],
    *,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Run each prompt/configuration and save raw outputs and aggregates."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for configuration in configurations:
        rows: list[dict[str, Any]] = []
        settings = Settings(
            retrieval_enabled=configuration["retrieval_enabled"],
            verifier_enabled=configuration["verifier_enabled"],
        )
        for entry in dataset:
            answer = run_agent(entry["question"], settings)
            sources = {source.id: source for source in answer.sources}
            citations = [
                {
                    "url": sources[source_id].url,
                    "supports_claim": point.verified,
                }
                for section in answer.sections
                for point in section.points
                for source_id in point.source_ids
                if source_id in sources
            ]
            fetched_urls = {source.url for source in answer.sources}
            metrics = citation_metrics(
                citations, fetched_urls, set(entry["gold_urls"]) if "gold_urls" in entry else None
            )
            gold_answer = entry.get("gold_answer")
            accuracy_value = None
            if gold_answer is not None:
                normalized = " ".join(
                    [answer.summary]
                    + [
                        point.text
                        for section in answer.sections
                        for point in section.points
                    ]
                ).casefold()
                accuracy_value = all(
                    str(fact).casefold() in normalized
                    for fact in gold_answer.get("required_facts", [])
                )
            rows.append(
                {
                    "id": entry["id"],
                    "question": entry["question"],
                    "answer": answer.model_dump(mode="json"),
                    "accuracy": accuracy_value,
                    **metrics,
                }
            )
        summary = aggregate_metrics(rows)
        result = {
            "configuration": configuration,
            "timestamp": datetime.now(UTC).isoformat(),
            "summary": summary,
            "results": rows,
        }
        (output_dir / f"{configuration['name']}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        results.append(result)
    return results


def main() -> None:
    """Parse command-line options and run the selected evaluation grid."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "eval" / "dataset.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--config", choices=[item["name"] for item in CONFIGURATIONS], action="append"
    )
    args = parser.parse_args()
    selected = [item for item in CONFIGURATIONS if not args.config or item["name"] in args.config]
    summaries = run_evaluation(load_dataset(args.dataset), selected, output_dir=args.output)
    for result in summaries:
        print(f"{result['configuration']['name']}: {json.dumps(result['summary'])}")


if __name__ == "__main__":
    main()
