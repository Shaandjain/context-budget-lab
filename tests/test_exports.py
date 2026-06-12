from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.export_workloads import DEFAULT_EXPORT_DATASETS, export_workloads, task_to_workload_record
from context_budget_lab.datasets import load_dataset


REQUIRED_FIELDS = {
    "task_id": str,
    "dataset_id": str,
    "task_type": str,
    "query": str,
    "context": list,
    "expected_answer": str,
    "expected_citations": list,
    "expected_facts": list,
    "answer_schema": str,
    "metadata": dict,
}
REQUIRED_CONTEXT_FIELDS = {
    "document_id": str,
    "title": str,
    "text": str,
    "source_url": str,
}


def test_task_to_workload_record_matches_schema() -> None:
    task = load_dataset("public_ai_policy_v0")[0]
    record = task_to_workload_record(task)

    validate_export_record(record)
    assert record["task_id"] == task.task_id
    assert record["expected_facts"] == list(task.expected_facts)


def test_export_workloads_writes_jsonl_files(tmp_path: Path) -> None:
    written = export_workloads(list(DEFAULT_EXPORT_DATASETS), out_dir=tmp_path)

    assert {path.name for path in written} == {
        "public_ai_policy_v0.jsonl",
        "synthetic_agent_memory_v0.jsonl",
    }
    for path in written:
        records = _read_jsonl(path)
        assert len(records) == 20
        for record in records:
            validate_export_record(record)


def test_committed_exports_are_valid_if_present() -> None:
    export_dir = Path("exports")
    if not export_dir.exists():
        return

    task_ids: set[str] = set()
    for dataset_id in DEFAULT_EXPORT_DATASETS:
        path = export_dir / f"{dataset_id}.jsonl"
        assert path.exists()
        records = _read_jsonl(path)
        assert len(records) == 20
        for record in records:
            validate_export_record(record)
            assert record["dataset_id"] == dataset_id
            assert record["task_id"] not in task_ids
            task_ids.add(record["task_id"])


def validate_export_record(record: dict[str, Any]) -> None:
    for field, expected_type in REQUIRED_FIELDS.items():
        assert field in record
        assert isinstance(record[field], expected_type)
        if expected_type in {str, dict} or field in {"context", "expected_facts"}:
            assert record[field]

    assert record["task_type"] in {"citation_qa", "memory_qa", "structured_extraction"}
    assert record["answer_schema"] == "answer_with_citations"
    assert record["metadata"]["source_type"] in {"public", "synthetic"}
    assert record["metadata"]["difficulty"] in {"toy", "easy", "medium", "hard"}
    assert isinstance(record["metadata"]["tags"], list)
    assert record["task_id"].startswith(record["dataset_id"] + "/")

    context_ids = set()
    for document in record["context"]:
        assert isinstance(document, dict)
        for field, expected_type in REQUIRED_CONTEXT_FIELDS.items():
            assert field in document
            assert isinstance(document[field], expected_type)
            assert document[field]
        assert document["document_id"] not in context_ids
        context_ids.add(document["document_id"])

    assert set(record["expected_citations"]).issubset(context_ids)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
