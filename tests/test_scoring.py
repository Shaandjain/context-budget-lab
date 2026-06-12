from __future__ import annotations

import pytest

from context_budget_lab.scoring import extract_cited_source_ids, score_answer


@pytest.fixture
def sample_task() -> dict[str, object]:
    return {
        "task_id": "task-1",
        "question": "Who wrote notes on the Analytical Engine?",
        "sources": [
            {"id": "S1", "text": "Ada Lovelace wrote notes on the Analytical Engine."},
            {"id": "S2", "text": "Grace Hopper worked on compilers."},
        ],
        "gold_answer_keywords": ["Ada Lovelace", "Analytical Engine"],
        "gold_source_ids": ["S1"],
        "metadata": {"dataset": "synthetic-history"},
    }


def test_score_answer_computes_keyword_citation_and_evidence_scores(sample_task: dict[str, object]) -> None:
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
    assert result["scores"]["answer_keyword_score"] == 1.0
    assert result["scores"]["citation_source_score"] == 1.0
    assert result["scores"]["evidence_recall"] == 1.0
    assert result["scores"]["overall_score"] == 1.0
    assert result["matched_answer_keywords"] == ["Ada Lovelace", "Analytical Engine"]
    assert result["cited_source_ids"] == ["S1"]
    assert result["metadata"]["task_metadata"] == {"dataset": "synthetic-history"}
    assert result["metadata"]["strategy_metadata"] == {"top_k": 1}


def test_score_answer_handles_partial_answer_and_wrong_evidence(sample_task: dict[str, object]) -> None:
    strategy_output = {
        "strategy": "rag_topk",
        "source_ids": ["S2"],
        "metadata": {"top_k": 1},
    }

    result = score_answer(sample_task, strategy_output, "Ada Lovelace was involved. [S2]")

    assert result["scores"]["answer_keyword_score"] == 0.5
    assert result["scores"]["citation_source_score"] == 0.0
    assert result["scores"]["evidence_recall"] == 0.0
    assert result["matched_answer_keywords"] == ["Ada Lovelace"]
    assert result["matched_gold_source_ids"] == []


def test_score_answer_returns_none_for_missing_gold_fields() -> None:
    task = {
        "task_id": "ungolded",
        "question": "What happened?",
        "sources": [{"id": "S1", "text": "Something happened."}],
    }

    result = score_answer(task, {"strategy": "full_context", "source_ids": ["S1"]}, "Something happened. [S1]")

    assert result["scores"]["answer_keyword_score"] is None
    assert result["scores"]["citation_source_score"] is None
    assert result["scores"]["evidence_recall"] is None
    assert result["scores"]["overall_score"] is None


def test_score_answer_accepts_strategy_name_without_metadata(sample_task: dict[str, object]) -> None:
    result = score_answer(sample_task, "summary_memory", "Ada Lovelace wrote about the Analytical Engine.")

    assert result["strategy"] == "summary_memory"
    assert result["selected_source_ids"] == []
    assert result["scores"]["answer_keyword_score"] == 1.0
    assert result["scores"]["citation_source_score"] == 0.0
    assert result["scores"]["evidence_recall"] is None


def test_extract_cited_source_ids_accepts_common_citation_shapes() -> None:
    answer = "Ada is supported by [S1], 【S2】, (S3), and source S4."

    assert extract_cited_source_ids(answer, ["S1", "S2", "S3", "S4", "S5"]) == ["S1", "S2", "S3", "S4"]
