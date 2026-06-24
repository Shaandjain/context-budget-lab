from __future__ import annotations

from context_budget_lab.datasets import load_dataset, task_to_strategy_input, validate_task_object
from context_budget_lab.scoring import score_answer
from benchmarks.build_abstention import ABSTAIN_DATASET_ID, NEW_TASKS


def test_abstain_suite_has_about_twenty_tasks() -> None:
    tasks = load_dataset(ABSTAIN_DATASET_ID)
    assert len(tasks) == 6 + len(NEW_TASKS) == 20


def test_every_task_expects_abstention_and_has_no_gold_citation() -> None:
    for task in load_dataset(ABSTAIN_DATASET_ID):
        tags = {t.lower() for t in task.metadata.get("tags", [])}
        assert "abstention" in tags
        assert task.expected_citations == ()
        assert "memory does not contain" in task.expected_facts


def test_correct_abstention_scores_true_fabrication_scores_false() -> None:
    task = load_dataset(ABSTAIN_DATASET_ID)[6]  # an authored pure-absence task
    strategy_input = task_to_strategy_input(task)

    good = score_answer(strategy_input, "full_context", "The memory does not contain this; not recorded.")
    bad = score_answer(strategy_input, "full_context", "It is 123 Main Street, definitely.")

    assert good["abstain_expected"] is True
    assert good["scores"]["abstain_correct"] is True
    assert bad["scores"]["abstain_correct"] is False  # negative case


def test_haystack_variants_preserve_abstention_under_pressure() -> None:
    for size in (2, 8, 32):
        variant = load_dataset(f"{ABSTAIN_DATASET_ID}_h{size}")
        assert len(variant) == 20
        assert all(len(t.context) == size for t in variant)
        for task in variant:
            # Padding with distractors must not introduce a gold citation or
            # drop the abstention expectation.
            assert task.expected_citations == ()
            assert "abstention" in {t.lower() for t in task.metadata.get("tags", [])}


def test_new_task_records_validate_and_reuse_existing_docs() -> None:
    memory_doc_ids = {
        d.document_id for t in load_dataset("synthetic_agent_memory_v0") for d in t.context
    }
    for local_id, query, topic, third, domain, doc_ids in NEW_TASKS:
        assert len(doc_ids) == 3
        assert set(doc_ids).issubset(memory_doc_ids)  # distractors are real docs
        assert query.endswith("?")
        assert topic and third and domain


def test_migrated_ids_and_new_ids_are_all_distinct() -> None:
    ids = [t.task_id for t in load_dataset(ABSTAIN_DATASET_ID)]
    assert len(ids) == len(set(ids))
