from __future__ import annotations

import pytest

from context_budget_lab.strategies import (
    STRATEGIES,
    build_all_strategy_prompts,
    build_strategy_prompt,
    full_context,
    prefix_cache_friendly,
    rag_topk,
    structured_memory,
    summary_memory,
)


@pytest.fixture
def sample_task() -> dict[str, object]:
    return {
        "task_id": "task-1",
        "question": "Who wrote notes on the Analytical Engine?",
        "sources": [
            {
                "id": "S1",
                "title": "Computing history",
                "text": "Ada Lovelace wrote notes on the Analytical Engine, including Bernoulli numbers.",
            },
            {
                "id": "S2",
                "title": "Programming languages",
                "text": "Grace Hopper worked on compilers and helped popularize COBOL.",
            },
            {
                "id": "S3",
                "title": "Geography",
                "text": "Paris is the capital of France.",
            },
        ],
        "summary": "Ada Lovelace wrote the Analytical Engine notes. Source S1.",
        "structured_facts": [
            {"claim": "Ada Lovelace wrote notes on the Analytical Engine.", "source_id": "S1"},
            {"claim": "Grace Hopper worked on COBOL.", "source_id": "S2"},
        ],
        "gold_answer_keywords": ["Ada Lovelace", "Analytical Engine"],
        "gold_source_ids": ["S1"],
    }


def test_full_context_includes_all_sources(sample_task: dict[str, object]) -> None:
    result = full_context(sample_task)

    assert result["strategy"] == "full_context"
    assert result["source_ids"] == ["S1", "S2", "S3"]
    assert "[S1]" in result["prompt"]
    assert "[S2]" in result["prompt"]
    assert "[S3]" in result["prompt"]


def test_rag_topk_uses_lexical_overlap(sample_task: dict[str, object]) -> None:
    result = rag_topk(sample_task, top_k=1)

    assert result["strategy"] == "rag_topk"
    assert result["source_ids"] == ["S1"]
    assert "Ada Lovelace" in result["prompt"]
    assert "Grace Hopper" not in result["prompt"]


def test_summary_memory_builds_summary_only_prompt(sample_task: dict[str, object]) -> None:
    result = summary_memory(sample_task)

    assert result["strategy"] == "summary_memory"
    assert result["source_ids"] == ["S1"]
    assert "Summary Memory" in result["prompt"]
    assert "Ada Lovelace wrote the Analytical Engine notes" in result["prompt"]
    assert "including Bernoulli numbers" not in result["prompt"]


def test_structured_memory_formats_facts_and_preserves_fact_sources(sample_task: dict[str, object]) -> None:
    result = structured_memory(sample_task)

    assert result["strategy"] == "structured_memory"
    assert result["source_ids"] == ["S1", "S2"]
    assert "Structured Facts" in result["prompt"]
    assert "Ada Lovelace wrote notes on the Analytical Engine" in result["prompt"]


def test_prefix_cache_friendly_keeps_stable_header_before_payload(sample_task: dict[str, object]) -> None:
    result = prefix_cache_friendly(sample_task, top_k=1)

    assert result["strategy"] == "prefix_cache_friendly"
    assert result["source_ids"] == ["S1"]
    assert result["prompt"].startswith("You are running the context-budget-lab benchmark.")
    assert result["prompt"].index("Payload:") < result["prompt"].index("Question:")
    assert result["metadata"]["cache_prefix"] == "context_budget_lab.v1"


def test_strategy_dispatch_covers_all_registered_strategies(sample_task: dict[str, object]) -> None:
    assert list(STRATEGIES) == [
        "full_context",
        "rag_topk",
        "summary_memory",
        "structured_memory",
        "prefix_cache_friendly",
    ]

    results = build_all_strategy_prompts(sample_task, top_k=2)

    assert [result["strategy"] for result in results] == list(STRATEGIES)
    assert build_strategy_prompt(sample_task, "rag_topk", top_k=1)["source_ids"] == ["S1"]


def test_unknown_strategy_raises(sample_task: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        build_strategy_prompt(sample_task, "made_up")
