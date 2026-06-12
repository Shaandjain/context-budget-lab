from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

Task = Mapping[str, Any]
Source = Mapping[str, Any]

STANDARD_INSTRUCTIONS = (
    "Answer the question using only the provided context. "
    "Cite source ids in square brackets when source-backed evidence is used. "
    "If the context is insufficient, say what is missing."
)

PREFIX_CACHE_HEADER = "\n".join(
    [
        "You are running the context-budget-lab benchmark.",
        "Rules:",
        "- Use only the request-specific evidence below.",
        "- Cite source ids in square brackets when source-backed evidence is used.",
        "- Keep the answer concise and explicit.",
        "",
        "The request-specific payload begins after this line.",
    ]
)


def question(task: Task) -> str:
    value = task.get("question", "")
    return str(value).strip()


def sources(task: Task) -> list[Source]:
    raw_sources = task.get("sources", [])
    if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)):
        return []
    return [source for source in raw_sources if isinstance(source, Mapping)]


def source_id(source: Source, index: int) -> str:
    for key in ("id", "source_id", "doc_id"):
        value = source.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"source_{index + 1}"


def source_text(source: Source) -> str:
    for key in ("text", "content", "body", "excerpt"):
        value = source.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def source_title(source: Source) -> str:
    for key in ("title", "name"):
        value = source.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def source_ids(raw_sources: Sequence[Source]) -> list[str]:
    return [source_id(source, index) for index, source in enumerate(raw_sources)]


def format_sources(raw_sources: Sequence[Source], max_chars_per_source: int = 1_800) -> str:
    if not raw_sources:
        return "No source text provided."

    blocks: list[str] = []
    for index, source in enumerate(raw_sources):
        sid = source_id(source, index)
        title = source_title(source)
        text = source_text(source) or "(empty source)"
        if len(text) > max_chars_per_source:
            text = text[: max_chars_per_source - 3].rstrip() + "..."

        heading = f"[{sid}]"
        if title:
            heading = f"{heading} {title}"
        blocks.append(f"{heading}\n{text}")
    return "\n\n".join(blocks)


def format_summary(task: Task) -> str:
    summary = task.get("summary", "")
    text = str(summary).strip()
    return text if text else "No summary memory provided."


def format_structured_facts(value: Any) -> str:
    if value is None:
        return "No structured facts provided."
    if isinstance(value, str):
        return value.strip() or "No structured facts provided."
    if isinstance(value, Mapping):
        return "\n".join(f"- {key}: {_compact(value_item)}" for key, value_item in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        lines = [_compact(item) for item in value]
        return "\n".join(f"- {line}" for line in lines) if lines else "No structured facts provided."
    return _compact(value)


def build_standard_prompt(
    task: Task,
    evidence_heading: str,
    evidence_body: str,
    *,
    instructions: str = STANDARD_INSTRUCTIONS,
) -> str:
    return "\n\n".join(
        [
            instructions,
            f"Question:\n{question(task)}",
            f"{evidence_heading}:\n{evidence_body}",
            "Answer:",
        ]
    )


def build_prefix_cache_prompt(task: Task, evidence_body: str) -> str:
    return "\n\n".join(
        [
            PREFIX_CACHE_HEADER,
            "Payload:",
            f"Question:\n{question(task)}",
            f"Evidence:\n{evidence_body}",
            "Answer:",
        ]
    )


def _compact(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    except TypeError:
        return str(value)
