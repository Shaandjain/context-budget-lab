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

    expected_facts = _string_list(task.get("expected_facts", task.get("gold_answer_keywords", [])))
    gold_source_ids = _string_list(task.get("gold_source_ids", []))
    known_source_ids = _known_source_ids(task, gold_source_ids)

    matched_facts = _matched_keywords(answer, expected_facts)
    citation_tokens = extract_citation_tokens(answer)
    cited_source_ids = extract_cited_source_ids(answer, known_source_ids)
    unknown_citation_ids = _unknown_citation_ids(citation_tokens, known_source_ids)
    matched_gold_citations = _intersection(cited_source_ids, gold_source_ids)
    matched_gold_evidence = _intersection(selected_source_ids, gold_source_ids)

    fact_coverage = _ratio(len(matched_facts), len(expected_facts))
    citation_precision = _citation_precision(
        matched_gold_citations=matched_gold_citations,
        citation_tokens=citation_tokens,
        gold_source_ids=gold_source_ids,
    )
    citation_recall = _ratio(len(matched_gold_citations), len(gold_source_ids))
    evidence_recall = _ratio(len(matched_gold_evidence), len(gold_source_ids)) if selected_source_ids else None
    schema_ok, schema_errors = _schema_check(
        answer_schema=str(task.get("answer_schema", "")),
        citation_tokens=citation_tokens,
        unknown_citation_ids=unknown_citation_ids,
        expected_citation_count=len(gold_source_ids),
    )
    abstained = detect_abstention(answer)
    abstain_expected = _expects_abstention(task, gold_source_ids)
    abstain_correct = _abstain_correct(abstained=abstained, expected=abstain_expected)

    return {
        "task_id": _task_id(task),
        "strategy": strategy_name,
        "scores": {
            "fact_coverage": fact_coverage,
            "citation_precision": citation_precision,
            "citation_recall": citation_recall,
            "evidence_recall": evidence_recall,
            "schema_ok": schema_ok,
            "abstain_correct": abstain_correct,
        },
        "matched_expected_facts": matched_facts,
        "cited_source_ids": cited_source_ids,
        "unknown_citation_ids": unknown_citation_ids,
        "matched_gold_source_ids": matched_gold_citations,
        "selected_source_ids": selected_source_ids,
        "gold_source_ids": gold_source_ids,
        "schema_errors": schema_errors,
        "abstained": abstained,
        "abstain_expected": abstain_expected,
        "metadata": {
            "task_metadata": dict(task.get("metadata", {})) if isinstance(task.get("metadata", {}), Mapping) else {},
            "strategy_metadata": strategy_metadata,
        },
    }


def extract_citation_tokens(answer: str) -> list[str]:
    patterns = (
        r"\[([A-Za-z][A-Za-z0-9_.:/-]*)\]",
        r"【([A-Za-z][A-Za-z0-9_.:/-]*)】",
        r"\(([A-Za-z][A-Za-z0-9_.:/-]*)\)",
        r"\bsource\s+([A-Za-z][A-Za-z0-9_.:/-]*)\b",
    )
    matches: list[tuple[int, str]] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, answer, flags=re.IGNORECASE):
            token = match.group(1).strip(".,:;")
            normalized = token.lower()
            if token and normalized not in seen:
                matches.append((match.start(), token))
                seen.add(normalized)
    matches.sort(key=lambda item: item[0])
    return [token for _, token in matches]


def extract_cited_source_ids(answer: str, known_source_ids: Sequence[str]) -> list[str]:
    known_lookup = {sid.lower(): sid for sid in known_source_ids}
    found: list[str] = []
    for token in extract_citation_tokens(answer):
        known = known_lookup.get(token.lower())
        if known is not None and known not in found:
            found.append(known)

    return found


def detect_abstention(answer: str) -> bool:
    normalized = _normalize(answer)
    phrases = (
        "memory does not contain",
        "memory doesn t contain",
        "does not contain",
        "not recorded",
        "not in memory",
        "insufficient evidence",
        "cannot determine",
        "can t determine",
        "do not know",
        "don t know",
        "not enough information",
    )
    return any(phrase in normalized for phrase in phrases)


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


def _citation_precision(
    *,
    matched_gold_citations: Sequence[str],
    citation_tokens: Sequence[str],
    gold_source_ids: Sequence[str],
) -> float | None:
    if not citation_tokens:
        return None if not gold_source_ids else 0.0
    return len(matched_gold_citations) / len(citation_tokens)


def _schema_check(
    *,
    answer_schema: str,
    citation_tokens: Sequence[str],
    unknown_citation_ids: Sequence[str],
    expected_citation_count: int,
) -> tuple[bool | None, list[str]]:
    if not answer_schema:
        return None, []
    if answer_schema != "answer_with_citations":
        return False, [f"unsupported answer_schema: {answer_schema}"]

    errors: list[str] = []
    if unknown_citation_ids:
        errors.append("unknown citation id(s): " + ", ".join(unknown_citation_ids))
    if expected_citation_count > 0 and not citation_tokens:
        errors.append("missing citation")
    return not errors, errors


def _unknown_citation_ids(citation_tokens: Sequence[str], known_source_ids: Sequence[str]) -> list[str]:
    known = {sid.lower() for sid in known_source_ids}
    return [token for token in citation_tokens if token.lower() not in known]


def _expects_abstention(task: Task, gold_source_ids: Sequence[str]) -> bool:
    metadata = task.get("metadata", {})
    tags = _string_list(metadata.get("tags", [])) if isinstance(metadata, Mapping) else []
    facts = _string_list(task.get("expected_facts", task.get("gold_answer_keywords", [])))
    return (
        "abstention" in {tag.lower() for tag in tags}
        or (not gold_source_ids and any(fact.lower() == "memory does not contain" for fact in facts))
    )


def _abstain_correct(*, abstained: bool, expected: bool) -> bool | None:
    if expected:
        return abstained
    if abstained:
        return False
    return None


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
