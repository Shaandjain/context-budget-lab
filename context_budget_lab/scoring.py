from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from context_budget_lab.prompts import Task, source_id, sources

ScoreResult = dict[str, Any]


def score_answer(task: Task, strategy_output: Mapping[str, Any] | str, answer: str) -> ScoreResult:
    strategy_name = strategy_output if isinstance(strategy_output, str) else str(strategy_output.get("strategy", ""))
    selected_source_ids = [] if isinstance(strategy_output, str) else _string_list(strategy_output.get("source_ids", []))
    strategy_metadata = {} if isinstance(strategy_output, str) else dict(strategy_output.get("metadata", {}))

    gold_keywords = _string_list(task.get("gold_answer_keywords", []))
    gold_source_ids = _string_list(task.get("gold_source_ids", []))
    known_source_ids = _known_source_ids(task, gold_source_ids)

    matched_keywords = _matched_keywords(answer, gold_keywords)
    cited_source_ids = extract_cited_source_ids(answer, known_source_ids)
    matched_gold_citations = _intersection(cited_source_ids, gold_source_ids)
    matched_gold_evidence = _intersection(selected_source_ids, gold_source_ids)

    answer_keyword_score = _ratio(len(matched_keywords), len(gold_keywords))
    citation_source_score = _ratio(len(matched_gold_citations), len(gold_source_ids))
    evidence_recall = _ratio(len(matched_gold_evidence), len(gold_source_ids)) if selected_source_ids else None

    component_scores = [
        score
        for score in (answer_keyword_score, citation_source_score, evidence_recall)
        if score is not None
    ]
    overall_score = sum(component_scores) / len(component_scores) if component_scores else None

    return {
        "task_id": _task_id(task),
        "strategy": strategy_name,
        "scores": {
            "answer_keyword_score": answer_keyword_score,
            "citation_source_score": citation_source_score,
            "evidence_recall": evidence_recall,
            "overall_score": overall_score,
        },
        "matched_answer_keywords": matched_keywords,
        "cited_source_ids": cited_source_ids,
        "matched_gold_source_ids": matched_gold_citations,
        "selected_source_ids": selected_source_ids,
        "gold_source_ids": gold_source_ids,
        "metadata": {
            "task_metadata": dict(task.get("metadata", {})) if isinstance(task.get("metadata", {}), Mapping) else {},
            "strategy_metadata": strategy_metadata,
        },
    }


def extract_cited_source_ids(answer: str, known_source_ids: Sequence[str]) -> list[str]:
    found: list[str] = []
    for sid in known_source_ids:
        escaped = re.escape(sid)
        patterns = (
            rf"\[{escaped}\]",
            rf"【{escaped}】",
            rf"\({escaped}\)",
            rf"\bsource\s+{escaped}\b",
        )
        if any(re.search(pattern, answer, flags=re.IGNORECASE) for pattern in patterns):
            found.append(sid)
    return found


def _matched_keywords(answer: str, gold_keywords: Sequence[str]) -> list[str]:
    normalized_answer = _normalize(answer)
    matches: list[str] = []
    for keyword in gold_keywords:
        normalized_keyword = _normalize(keyword)
        if not normalized_keyword:
            continue
        if _contains_keyword(normalized_answer, normalized_keyword):
            matches.append(keyword)
    return matches


def _contains_keyword(normalized_answer: str, normalized_keyword: str) -> bool:
    if " " not in normalized_keyword and re.fullmatch(r"[a-z0-9_]+", normalized_keyword):
        return re.search(rf"\b{re.escape(normalized_keyword)}\b", normalized_answer) is not None
    return normalized_keyword in normalized_answer


def _normalize(text: str) -> str:
    lowered = text.lower()
    collapsed = re.sub(r"[^a-z0-9_]+", " ", lowered)
    return re.sub(r"\s+", " ", collapsed).strip()


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _intersection(values: Sequence[str], gold_values: Sequence[str]) -> list[str]:
    gold_lookup = {value.lower(): value for value in gold_values}
    matches: list[str] = []
    for value in values:
        match = gold_lookup.get(value.lower())
        if match is not None and match not in matches:
            matches.append(match)
    return matches


def _known_source_ids(task: Task, gold_source_ids: Sequence[str]) -> list[str]:
    ids = [source_id(source, index) for index, source in enumerate(sources(task))]
    ids.extend(gold_source_ids)
    return list(dict.fromkeys(ids))


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _task_id(task: Task) -> str:
    for key in ("task_id", "id"):
        value = task.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""
