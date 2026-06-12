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
    tasks = load_dataset("public_ai_policy_v0")

    assert len(tasks) == 20
    assert "public_ai_policy_v0" in available_datasets()
    assert {task.dataset_id for task in tasks} == {"public_ai_policy_v0"}
    assert {task.metadata["source_type"] for task in tasks} == {"public"}
    assert _difficulty_counts(tasks) == {"easy": 6, "medium": 10, "hard": 4}

    for task in tasks:
        assert task.context
        assert task.answer_schema == "answer_with_citations"
        assert task.expected_facts
        assert all(url.startswith("https://") for url in task.metadata["source_urls"])
        context_ids = {document.document_id for document in task.context}
        assert set(task.expected_citations).issubset(context_ids)
        assert len(task.context) >= 3
        assert _valid_distractors(task)


def test_loads_synthetic_agent_memory_dataset() -> None:
    tasks = load_dataset("synthetic_agent_memory_v0")

    assert len(tasks) == 20
    assert "synthetic_agent_memory_v0" in available_datasets()
    assert {task.dataset_id for task in tasks} == {"synthetic_agent_memory_v0"}
    assert {task.metadata["source_type"] for task in tasks} == {"synthetic"}
    assert _difficulty_counts(tasks) == {"easy": 6, "medium": 10, "hard": 4}

    for task in tasks:
        assert task.context
        assert task.answer_schema == "answer_with_citations"
        assert task.expected_facts
        assert all(url.startswith("https://") for url in task.metadata["source_urls"])
        assert len(task.context) >= 3
        assert _valid_distractors(task)

    abstention_tasks = [task for task in tasks if "abstention" in task.metadata["tags"]]
    assert len(abstention_tasks) >= 5
    assert all(not task.expected_citations for task in abstention_tasks)
    assert all(
        any("memory does not contain" == fact.lower() for fact in task.expected_facts)
        for task in abstention_tasks
    )


def test_dataset_task_ids_are_unique_across_registered_datasets() -> None:
    seen: set[str] = set()
    for dataset_id in available_datasets():
        for task in load_dataset(dataset_id):
            assert task.task_id not in seen
            seen.add(task.task_id)


def test_task_to_strategy_input_adapts_loader_contract() -> None:
    task = load_dataset("public_ai_policy_v0")[0]

    adapted = task_to_strategy_input(task)

    assert adapted["task_id"] == task.task_id
    assert adapted["question"] == task.query
    assert adapted["sources"][0]["id"] == task.context[0].document_id
    assert adapted["gold_source_ids"] == list(task.expected_citations)
    assert adapted["gold_answer_keywords"] == list(task.expected_facts)
    assert adapted["answer_schema"] == "answer_with_citations"
    assert adapted["summary"]
    assert adapted["structured_facts"][0]["source_id"] == task.context[0].document_id


def test_validate_task_object_rejects_missing_required_fields() -> None:
    with pytest.raises(DatasetValidationError, match="missing required field"):
        validate_task_object(
            {
                "task_id": "broken-task",
                "dataset_id": "public_ai_policy_v0",
            }
        )


def test_validate_task_object_rejects_unknown_citation_id() -> None:
    raw = {
        "task_id": "broken-citation",
        "dataset_id": "public_ai_policy_v0",
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
        "expected_facts": ["known document"],
        "answer_schema": "answer_with_citations",
        "metadata": {
            "source_type": "public",
            "source_urls": ["https://example.com/doc-a"],
            "difficulty": "easy",
            "tags": ["test"],
        },
    }

    with pytest.raises(DatasetValidationError, match="not present in context"):
        validate_task_object(raw)


def test_validate_task_object_rejects_distractor_that_is_expected_citation() -> None:
    raw = {
        "task_id": "broken-distractor",
        "dataset_id": "public_ai_policy_v0",
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
        "expected_citations": ["doc-a"],
        "expected_facts": ["known document"],
        "answer_schema": "answer_with_citations",
        "metadata": {
            "source_type": "public",
            "source_urls": ["https://example.com/doc-a"],
            "difficulty": "easy",
            "tags": ["test"],
            "distractor_document_ids": ["doc-a"],
        },
    }

    with pytest.raises(DatasetValidationError, match="expected citation"):
        validate_task_object(raw)


def _difficulty_counts(tasks: list[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        difficulty = task.metadata["difficulty"]  # type: ignore[attr-defined]
        counts[difficulty] = counts.get(difficulty, 0) + 1
    return counts


def _valid_distractors(task: object) -> bool:
    context_ids = {document.document_id for document in task.context}  # type: ignore[attr-defined]
    expected_citations = set(task.expected_citations)  # type: ignore[attr-defined]
    distractors = set(task.metadata.get("distractor_document_ids", []))  # type: ignore[attr-defined]
    return bool(distractors) and distractors.issubset(context_ids) and not distractors.intersection(expected_citations)
