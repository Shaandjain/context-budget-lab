from __future__ import annotations

import pytest

from context_budget_lab.datasets import (
    DatasetValidationError,
    available_datasets,
    load_dataset,
    task_to_strategy_input,
    validate_task_object,
)


def test_loads_public_ai_policy_dataset() -> None:
    tasks = load_dataset("public_ai_policy_toy")

    assert len(tasks) == 3
    assert "public_ai_policy_toy" in available_datasets()
    assert {task.dataset_id for task in tasks} == {"public_ai_policy_toy"}
    assert {task.metadata["source_type"] for task in tasks} == {"public"}

    for task in tasks:
        assert task.context
        assert task.expected_citations
        assert all(url.startswith("https://") for url in task.metadata["source_urls"])
        context_ids = {document.document_id for document in task.context}
        assert set(task.expected_citations).issubset(context_ids)


def test_loads_synthetic_agent_memory_dataset() -> None:
    tasks = load_dataset("synthetic_agent_memory_toy")

    assert len(tasks) == 3
    assert "synthetic_agent_memory_toy" in available_datasets()
    assert {task.dataset_id for task in tasks} == {"synthetic_agent_memory_toy"}
    assert {task.metadata["source_type"] for task in tasks} == {"synthetic"}

    for task in tasks:
        assert task.context
        assert task.expected_citations
        assert all(url.startswith("https://") for url in task.metadata["source_urls"])


def test_task_to_strategy_input_adapts_loader_contract() -> None:
    task = load_dataset("public_ai_policy_toy")[0]

    adapted = task_to_strategy_input(task)

    assert adapted["task_id"] == task.task_id
    assert adapted["question"] == task.query
    assert adapted["sources"][0]["id"] == task.context[0].document_id
    assert adapted["gold_source_ids"] == list(task.expected_citations)
    assert adapted["gold_answer_keywords"]
    assert adapted["summary"]
    assert adapted["structured_facts"][0]["source_id"] == task.context[0].document_id


def test_validate_task_object_rejects_missing_required_fields() -> None:
    with pytest.raises(DatasetValidationError, match="missing required field"):
        validate_task_object(
            {
                "task_id": "broken-task",
                "dataset_id": "public_ai_policy_toy",
            }
        )


def test_validate_task_object_rejects_unknown_citation_id() -> None:
    raw = {
        "task_id": "broken-citation",
        "dataset_id": "public_ai_policy_toy",
        "task_type": "citation_qa",
        "query": "What should be cited?",
        "context": [
            {
                "document_id": "doc-a",
                "title": "Doc A",
                "text": "A short context document.",
                "source_url": "https://example.com/doc-a",
            }
        ],
        "expected_answer": "Cite the known document.",
        "expected_citations": ["missing-doc"],
        "metadata": {
            "source_type": "public",
            "source_urls": ["https://example.com/doc-a"],
        },
    }

    with pytest.raises(DatasetValidationError, match="not present in context"):
        validate_task_object(raw)
