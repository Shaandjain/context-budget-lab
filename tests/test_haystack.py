from __future__ import annotations

from context_budget_lab.datasets import BenchmarkTask, ContextDocument, load_dataset
from context_budget_lab.haystack import (
    HAYSTACK_SIZES,
    build_haystack_dataset,
    build_haystack_task,
)


def _doc(doc_id: str) -> ContextDocument:
    return ContextDocument(
        document_id=doc_id,
        title=f"title {doc_id}",
        text=f"text {doc_id}",
        source_url="https://example.test/" + doc_id,
    )


def _task(task_id: str, context: list[ContextDocument], distractors: list[str]) -> BenchmarkTask:
    return BenchmarkTask(
        task_id=task_id,
        dataset_id="lane_v0",
        task_type="citation_qa",
        query="q",
        context=tuple(context),
        expected_answer="a",
        expected_citations=("gold-1",),
        expected_facts=("fact",),
        answer_schema="answer_with_citations",
        metadata={"distractor_document_ids": distractors},
    )


def _fixture_tasks() -> list[BenchmarkTask]:
    # Task A has one gold doc; the rest of the corpus supplies distractors.
    task_a = _task("lane_v0/a", [_doc("gold-1"), _doc("near-1")], ["near-1"])
    others = [
        _task(f"lane_v0/{i}", [_doc(f"d{i}-1"), _doc(f"d{i}-2")], [f"d{i}-1", f"d{i}-2"])
        for i in range(10)
    ]
    return [task_a] + others


def test_gold_doc_always_present_and_target_size_met() -> None:
    tasks = _fixture_tasks()
    record = build_haystack_task(
        tasks[0], tasks, target_size=8, seed=7, new_dataset_id="lane_v2_h8"
    )

    ids = [doc["document_id"] for doc in record["context"]]
    assert "gold-1" in ids  # gold survives
    assert len(ids) == 8  # padded to target
    assert record["dataset_id"] == "lane_v2_h8"
    assert record["task_id"] == "lane_v2_h8/a"
    assert record["metadata"]["base_task_id"] == "lane_v0/a"
    assert record["metadata"]["haystack_size"] == 8


def test_seeded_generation_is_deterministic() -> None:
    tasks = _fixture_tasks()
    first = build_haystack_task(tasks[0], tasks, 8, seed=7, new_dataset_id="lane_v2_h8")
    second = build_haystack_task(tasks[0], tasks, 8, seed=7, new_dataset_id="lane_v2_h8")
    assert first == second


def test_different_seed_changes_distractor_selection() -> None:
    tasks = _fixture_tasks()
    a = build_haystack_task(tasks[0], tasks, 8, seed=7, new_dataset_id="lane_v2_h8")
    b = build_haystack_task(tasks[0], tasks, 8, seed=999, new_dataset_id="lane_v2_h8")
    assert a != b  # negative control: padding must depend on the seed


def test_does_not_mutate_source_task() -> None:
    tasks = _fixture_tasks()
    before = tuple(doc.document_id for doc in tasks[0].context)
    build_haystack_task(tasks[0], tasks, 8, seed=7, new_dataset_id="lane_v2_h8")
    after = tuple(doc.document_id for doc in tasks[0].context)
    assert before == after  # original record untouched


def test_size_floors_at_gold_count_when_pool_too_small() -> None:
    # Negative case: ask for more docs than gold + available distractors.
    tasks = _fixture_tasks()
    record = build_haystack_task(
        tasks[0], tasks[:1], target_size=8, seed=7, new_dataset_id="lane_v2_h8"
    )
    ids = [doc["document_id"] for doc in record["context"]]
    # No distractor pool (only the task itself) -> just the gold doc remains.
    assert ids == ["gold-1"]
    assert record["metadata"]["haystack_size"] == 1


def test_real_v2_datasets_load_and_preserve_gold() -> None:
    for stem in ("public_ai_policy_v2", "synthetic_agent_memory_v2"):
        for size in HAYSTACK_SIZES:
            dataset_id = f"{stem}_h{size}"
            variant = load_dataset(dataset_id)
            assert len(variant) == 20
            for task in variant:
                context_ids = {doc.document_id for doc in task.context}
                assert set(task.expected_citations).issubset(context_ids)


def test_build_dataset_uses_cross_lane_pool_for_h32() -> None:
    policy = load_dataset("public_ai_policy_v0")
    memory = load_dataset("synthetic_agent_memory_v0")
    pool = policy + memory
    records = build_haystack_dataset(
        policy, 32, seed=1, new_dataset_id="public_ai_policy_v2_h32", pool_tasks=pool
    )
    assert all(len(r["context"]) == 32 for r in records)
