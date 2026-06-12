from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from context_budget_lab.prompts import (
    Task,
    Source,
    PREFIX_CACHE_ABSTAIN_HEADER,
    build_prefix_cache_prompt,
    build_standard_prompt,
    format_sources,
    format_structured_facts,
    format_summary,
    question,
    source_id,
    source_ids,
    source_text,
    source_title,
    sources,
)

StrategyResult = dict[str, Any]

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def full_context(task: Task) -> StrategyResult:
    selected = sources(task)
    return _result(
        task=task,
        strategy="full_context",
        prompt=build_standard_prompt(task, "Full Context", format_sources(selected)),
        selected_sources=selected,
        metadata={"retrieval": "all_sources", "source_count": len(selected)},
    )


def rag_topk(task: Task, *, top_k: int = 3) -> StrategyResult:
    selected = rank_sources(task)[:top_k]
    return _result(
        task=task,
        strategy="rag_topk",
        prompt=build_standard_prompt(task, f"Top {top_k} Retrieved Sources", format_sources(selected)),
        selected_sources=selected,
        metadata={"retrieval": "lexical_overlap", "top_k": top_k, "source_count": len(selected)},
    )


def summary_memory(task: Task) -> StrategyResult:
    body = format_summary(task)
    known_ids = source_ids(sources(task))
    return {
        "task_id": _task_id(task),
        "strategy": "summary_memory",
        "question": question(task),
        "prompt": build_standard_prompt(task, "Summary Memory", body),
        "source_ids": _ids_mentioned_in_text(body, known_ids),
        "metadata": {"memory": "summary", "source_count": 0},
    }


def structured_memory(task: Task) -> StrategyResult:
    facts = task.get("structured_facts")
    body = format_structured_facts(facts)
    ids = _fact_source_ids(facts, source_ids(sources(task)))
    return {
        "task_id": _task_id(task),
        "strategy": "structured_memory",
        "question": question(task),
        "prompt": build_standard_prompt(task, "Structured Facts", body),
        "source_ids": ids,
        "metadata": {"memory": "structured_facts", "source_count": 0},
    }


def prefix_cache_friendly(task: Task, *, top_k: int = 3) -> StrategyResult:
    selected = rank_sources(task)[:top_k]
    return _result(
        task=task,
        strategy="prefix_cache_friendly",
        prompt=build_prefix_cache_prompt(task, format_sources(selected)),
        selected_sources=selected,
        metadata={
            "retrieval": "lexical_overlap",
            "top_k": top_k,
            "source_count": len(selected),
            "cache_prefix": "context_budget_lab.v1",
        },
    )


def prefix_cache_abstain(task: Task, *, top_k: int = 3) -> StrategyResult:
    selected = rank_sources(task)[:top_k]
    return _result(
        task=task,
        strategy="prefix_cache_abstain",
        prompt=build_prefix_cache_prompt(task, format_sources(selected), header=PREFIX_CACHE_ABSTAIN_HEADER),
        selected_sources=selected,
        metadata={
            "retrieval": "lexical_overlap",
            "top_k": top_k,
            "source_count": len(selected),
            "cache_prefix": "context_budget_lab.v1.abstain",
            "variant": "explicit_insufficient_context_instruction",
        },
    )


def build_strategy_prompt(task: Task, strategy: str, *, top_k: int = 3) -> StrategyResult:
    if strategy == "full_context":
        return full_context(task)
    if strategy == "rag_topk":
        return rag_topk(task, top_k=top_k)
    if strategy == "summary_memory":
        return summary_memory(task)
    if strategy == "structured_memory":
        return structured_memory(task)
    if strategy == "prefix_cache_friendly":
        return prefix_cache_friendly(task, top_k=top_k)
    if strategy == "prefix_cache_abstain":
        return prefix_cache_abstain(task, top_k=top_k)
    raise ValueError(f"unknown strategy: {strategy}")


def build_all_strategy_prompts(task: Task, *, top_k: int = 3) -> list[StrategyResult]:
    return [build_strategy_prompt(task, name, top_k=top_k) for name in STRATEGIES]


def list_strategy_names() -> list[str]:
    return list(STRATEGIES)


def rank_sources(task: Task) -> list[Source]:
    query_terms = _tokens(question(task))
    scored: list[tuple[int, int, Source]] = []
    for index, source in enumerate(sources(task)):
        haystack = " ".join([source_title(source), source_text(source)])
        scored.append((_lexical_score(query_terms, haystack), index, source))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [source for _, _, source in scored]


def _result(
    *,
    task: Task,
    strategy: str,
    prompt: str,
    selected_sources: Sequence[Source],
    metadata: Mapping[str, Any],
) -> StrategyResult:
    return {
        "task_id": _task_id(task),
        "strategy": strategy,
        "question": question(task),
        "prompt": prompt,
        "source_ids": source_ids(selected_sources),
        "metadata": dict(metadata),
    }


def _tokens(text: str) -> list[str]:
    return [token for token in _WORD_RE.findall(text.lower()) if token not in _STOPWORDS]


def _lexical_score(query_terms: Sequence[str], text: str) -> int:
    if not query_terms:
        return 0
    text_counts = Counter(_tokens(text))
    return sum(min(count, text_counts[term]) for term, count in Counter(query_terms).items())


def _task_id(task: Task) -> str:
    for key in ("task_id", "id"):
        value = task.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _ids_mentioned_in_text(text: str, known_ids: Sequence[str]) -> list[str]:
    lowered = text.lower()
    return [sid for sid in known_ids if sid.lower() in lowered]


def _fact_source_ids(value: Any, known_ids: Sequence[str]) -> list[str]:
    explicit = _collect_source_ids(value)
    if explicit:
        known_lookup = {sid.lower(): sid for sid in known_ids}
        return [known_lookup.get(sid.lower(), sid) for sid in explicit]
    return _ids_mentioned_in_text(format_structured_facts(value), known_ids)


def _collect_source_ids(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"source_id", "source_ids", "gold_source_id", "gold_source_ids"}:
                found.extend(_string_values(item))
            else:
                found.extend(_collect_source_ids(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            found.extend(_collect_source_ids(item))
    return list(dict.fromkeys(found))


def _string_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


STRATEGIES: dict[str, Callable[..., StrategyResult]] = {
    "full_context": full_context,
    "rag_topk": rag_topk,
    "summary_memory": summary_memory,
    "structured_memory": structured_memory,
    "prefix_cache_friendly": prefix_cache_friendly,
    "prefix_cache_abstain": prefix_cache_abstain,
}
