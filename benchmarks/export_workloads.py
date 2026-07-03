"""Export context-budget datasets as release-lab workload JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_budget_lab.datasets import BenchmarkTask, DATASET_FILES, load_dataset


DEFAULT_EXPORT_DATASETS = ["public_ai_policy_v0", "synthetic_agent_memory_v0"]
V2_LONG_CONTEXT_EXPORT_DATASETS = [
    "public_ai_policy_v2_h32",
    "synthetic_agent_memory_v2_h32",
    "synthetic_agent_memory_abstain_v2_h32",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DEFAULT_EXPORT_DATASETS))
    parser.add_argument("--out-dir", type=Path, default=Path("exports"))
    args = parser.parse_args(argv)

    written = export_workloads(_split_csv(args.datasets), out_dir=args.out_dir)
    for path in written:
        print(path)
    return 0


def export_workloads(dataset_ids: list[str], *, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for dataset_id in dataset_ids:
        if dataset_id not in DATASET_FILES:
            raise ValueError(f"unknown dataset_id: {dataset_id}")
        path = out_dir / f"{dataset_id}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for task in load_dataset(dataset_id):
                handle.write(json.dumps(task_to_workload_record(task), sort_keys=True) + "\n")
        written.append(path)
    return written


def task_to_workload_record(task: BenchmarkTask) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "dataset_id": task.dataset_id,
        "task_type": task.task_type,
        "query": task.query,
        "context": [
            {
                "document_id": document.document_id,
                "title": document.title,
                "text": document.text,
                "source_url": document.source_url,
            }
            for document in task.context
        ],
        "expected_answer": task.expected_answer,
        "expected_citations": list(task.expected_citations),
        "expected_facts": list(task.expected_facts),
        "answer_schema": task.answer_schema,
        "metadata": dict(task.metadata),
    }


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
