from __future__ import annotations

import pytest

from context_budget_lab.scoring import (
    detect_abstention,
    extract_citation_tokens,
    extract_cited_source_ids,
    score_answer,
)


@pytest.fixture
def sample_task() -> dict[str, object]:
    return {
        "task_id": "task-1",
        "question": "Who wrote notes on the Analytical Engine?",
        "sources": [
            {"id": "S1", "text": "Ada Lovelace wrote notes on the Analytical Engine."},
            {"id": "S2", "text": "Grace Hopper worked on compilers."},
        ],
        "expected_facts": ["Ada Lovelace", "Analytical Engine"],
        "answer_schema": "answer_with_citations",
        "gold_source_ids": ["S1"],
        "metadata": {"dataset": "synthetic-history"},
    }


def test_score_answer_computes_fact_citation_schema_and_evidence_scores(sample_task: dict[str, object]) -> None:
    strategy_output = {
        "strategy": "rag_topk",
        "source_ids": ["S1"],
        "metadata": {"top_k": 1},
    }

    result = score_answer(
        sample_task,
        strategy_output,
        "Ada Lovelace wrote notes on the Analytical Engine. [S1]",
    )

    assert result["task_id"] == "task-1"
    assert result["strategy"] == "rag_topk"
    assert result["scores"]["fact_coverage"] == 1.0
    assert result["scores"]["citation_precision"] == 1.0
    assert result["scores"]["citation_recall"] == 1.0
    assert result["scores"]["evidence_recall"] == 1.0
    assert result["scores"]["schema_ok"] is True
    assert result["scores"]["abstain_correct"] is None
    assert result["matched_expected_facts"] == ["Ada Lovelace", "Analytical Engine"]
    assert result["cited_source_ids"] == ["S1"]
    assert result["unknown_citation_ids"] == []
    assert result["metadata"]["task_metadata"] == {"dataset": "synthetic-history"}
    assert result["metadata"]["strategy_metadata"] == {"top_k": 1}


def test_score_answer_handles_partial_answer_and_wrong_evidence(sample_task: dict[str, object]) -> None:
    strategy_output = {
        "strategy": "rag_topk",
        "source_ids": ["S2"],
        "metadata": {"top_k": 1},
    }

    result = score_answer(sample_task, strategy_output, "Ada Lovelace was involved. [S2]")

    assert result["scores"]["fact_coverage"] == 0.5
    assert result["scores"]["citation_precision"] == 0.0
    assert result["scores"]["citation_recall"] == 0.0
    assert result["scores"]["evidence_recall"] == 0.0
    assert result["matched_expected_facts"] == ["Ada Lovelace"]
    assert result["matched_gold_source_ids"] == []


def test_score_answer_penalizes_distractor_citations(sample_task: dict[str, object]) -> None:
    strategy_output = {
        "strategy": "full_context",
        "source_ids": ["S1", "S2"],
        "metadata": {},
    }

    result = score_answer(
        sample_task,
        strategy_output,
        "Ada Lovelace wrote notes on the Analytical Engine. [S1] [S2]",
    )

    assert result["scores"]["fact_coverage"] == 1.0
    assert result["scores"]["citation_precision"] == 0.5
    assert result["scores"]["citation_recall"] == 1.0
    assert result["scores"]["schema_ok"] is True


def test_score_answer_flags_fabricated_citation_even_with_right_facts(sample_task: dict[str, object]) -> None:
    result = score_answer(
        sample_task,
        {"strategy": "full_context", "source_ids": ["S1", "S2"]},
        "Ada Lovelace wrote notes on the Analytical Engine. [S999]",
    )

    assert result["scores"]["fact_coverage"] == 1.0
    assert result["scores"]["citation_precision"] == 0.0
    assert result["scores"]["citation_recall"] == 0.0
    assert result["scores"]["schema_ok"] is False
    assert result["unknown_citation_ids"] == ["S999"]
    assert result["schema_errors"] == ["unknown citation id(s): S999"]


def test_score_answer_returns_none_for_missing_gold_fields() -> None:
    task = {
        "task_id": "ungolded",
        "question": "What happened?",
        "sources": [{"id": "S1", "text": "Something happened."}],
    }

    result = score_answer(task, {"strategy": "full_context", "source_ids": ["S1"]}, "Something happened. [S1]")

    assert result["scores"]["fact_coverage"] is None
    assert result["scores"]["citation_precision"] == 0.0
    assert result["scores"]["citation_recall"] is None
    assert result["scores"]["evidence_recall"] is None
    assert result["scores"]["schema_ok"] is None


def test_score_answer_accepts_strategy_name_without_metadata(sample_task: dict[str, object]) -> None:
    result = score_answer(sample_task, "summary_memory", "Ada Lovelace wrote about the Analytical Engine.")

    assert result["strategy"] == "summary_memory"
    assert result["selected_source_ids"] == []
    assert result["scores"]["fact_coverage"] == 1.0
    assert result["scores"]["citation_precision"] == 0.0
    assert result["scores"]["citation_recall"] == 0.0
    assert result["scores"]["evidence_recall"] is None
    assert result["scores"]["schema_ok"] is False
    assert result["schema_errors"] == ["missing citation"]


def test_extract_cited_source_ids_accepts_common_citation_shapes() -> None:
    answer = "Ada is supported by [S1], 【S2】, (S3), and source S4."

    assert extract_cited_source_ids(answer, ["S1", "S2", "S3", "S4", "S5"]) == ["S1", "S2", "S3", "S4"]


def test_extract_citation_tokens_returns_fabricated_ids() -> None:
    answer = "The claim cites [S1], [S999], and source S2."

    assert extract_citation_tokens(answer) == ["S1", "S999", "S2"]


def test_correct_abstention_scores_true() -> None:
    task = {
        "task_id": "memory-missing",
        "question": "What is the user's favorite coffee shop?",
        "sources": [{"id": "M1", "text": "The memory only says the user likes concise updates."}],
        "expected_facts": ["memory does not contain", "favorite coffee shop"],
        "answer_schema": "answer_with_citations",
        "gold_source_ids": [],
        "metadata": {"tags": ["abstention"]},
    }

    result = score_answer(task, {"strategy": "full_context", "source_ids": ["M1"]}, "The memory does not contain the user's favorite coffee shop.")

    assert result["abstained"] is True
    assert result["abstain_expected"] is True
    assert result["scores"]["abstain_correct"] is True
    assert result["scores"]["schema_ok"] is True


def test_incorrect_abstention_scores_false(sample_task: dict[str, object]) -> None:
    result = score_answer(sample_task, {"strategy": "full_context", "source_ids": ["S1"]}, "I cannot determine this from the supplied context.")

    assert result["abstained"] is True
    assert result["abstain_expected"] is False
    assert result["scores"]["abstain_correct"] is False


def test_detect_abstention_handles_common_shapes() -> None:
    assert detect_abstention("The memory does not contain that preference.") is True
    assert detect_abstention("Ada Lovelace wrote the notes.") is False
