from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping


DATASET_DIR = Path(__file__).resolve().parents[1] / "datasets"
DATASET_FILES: Mapping[str, str] = {
    "public_ai_policy_toy": "public_ai_policy_toy.jsonl",
    "synthetic_agent_memory_toy": "synthetic_agent_memory_toy.jsonl",
}

_REQUIRED_TASK_FIELDS = {
    "task_id",
    "dataset_id",
    "task_type",
    "query",
    "context",
    "expected_answer",
    "expected_citations",
    "metadata",
}
_REQUIRED_CONTEXT_FIELDS = {"document_id", "title", "text", "source_url"}
_KEYWORD_STOPWORDS = {
    "about",
    "against",
    "answer",
    "around",
    "before",
    "being",
    "between",
    "cite",
    "from",
    "into",
    "rather",
    "should",
    "than",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "use",
    "with",
    "which",
}


class DatasetValidationError(ValueError):
    """Raised when a dataset id, JSONL line, or task object is invalid."""


@dataclass(frozen=True)
class ContextDocument:
    document_id: str
    title: str
    text: str
    source_url: str


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    dataset_id: str
    task_type: str
    query: str
    context: tuple[ContextDocument, ...]
    expected_answer: str
    expected_citations: tuple[str, ...]
    metadata: Mapping[str, Any]


def task_to_strategy_input(task: BenchmarkTask) -> dict[str, Any]:
    """Adapt the validated JSONL task into the prompt/scoring contract."""

    sources = [
        {
            "id": document.document_id,
            "title": document.title,
            "text": document.text,
            "source_url": document.source_url,
        }
        for document in task.context
    ]
    return {
        "task_id": task.task_id,
        "id": task.task_id,
        "dataset_id": task.dataset_id,
        "task_type": task.task_type,
        "question": task.query,
        "sources": sources,
        "summary": _summary_memory(task),
        "structured_facts": _structured_facts(task),
        "gold_answer_keywords": _gold_keywords(task.expected_answer),
        "gold_source_ids": list(task.expected_citations),
        "expected_answer": task.expected_answer,
        "metadata": dict(task.metadata),
    }


def available_datasets() -> tuple[str, ...]:
    return tuple(DATASET_FILES)


def dataset_path(dataset_id: str, data_dir: Path | None = None) -> Path:
    try:
        filename = DATASET_FILES[dataset_id]
    except KeyError as exc:
        known = ", ".join(available_datasets())
        raise DatasetValidationError(
            f"Unknown dataset_id {dataset_id!r}; expected one of: {known}"
        ) from exc

    return (data_dir or DATASET_DIR) / filename


def load_dataset(dataset_id: str, data_dir: Path | None = None) -> list[BenchmarkTask]:
    path = dataset_path(dataset_id, data_dir=data_dir)
    tasks: list[BenchmarkTask] = []
    seen_task_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(raw, Mapping):
                raise DatasetValidationError(
                    f"{path}:{line_number}: task must be a JSON object"
                )

            task = validate_task_object(raw, expected_dataset_id=dataset_id)
            if task.task_id in seen_task_ids:
                raise DatasetValidationError(
                    f"{path}:{line_number}: duplicate task_id {task.task_id!r}"
                )
            seen_task_ids.add(task.task_id)
            tasks.append(task)

    if not tasks:
        raise DatasetValidationError(f"{path}: dataset is empty")
    return tasks


def validate_task_object(
    raw: Mapping[str, Any], expected_dataset_id: str | None = None
) -> BenchmarkTask:
    missing = sorted(_REQUIRED_TASK_FIELDS.difference(raw))
    if missing:
        raise DatasetValidationError(
            f"Task object is missing required field(s): {', '.join(missing)}"
        )

    task_id = _require_str(raw, "task_id")
    dataset_id = _require_str(raw, "dataset_id")
    if expected_dataset_id is not None and dataset_id != expected_dataset_id:
        raise DatasetValidationError(
            f"Task {task_id!r} has dataset_id {dataset_id!r}; "
            f"expected {expected_dataset_id!r}"
        )

    context = _validate_context(raw["context"], task_id=task_id)
    citation_ids = _validate_str_list(
        raw["expected_citations"], field="expected_citations", task_id=task_id
    )
    context_ids = {document.document_id for document in context}
    missing_citations = sorted(set(citation_ids).difference(context_ids))
    if missing_citations:
        raise DatasetValidationError(
            f"Task {task_id!r} cites document id(s) not present in context: "
            f"{', '.join(missing_citations)}"
        )

    metadata = raw["metadata"]
    if not isinstance(metadata, Mapping):
        raise DatasetValidationError(f"Task {task_id!r} field 'metadata' must be an object")

    source_urls = _validate_str_list(
        metadata.get("source_urls"), field="metadata.source_urls", task_id=task_id
    )
    if not source_urls:
        raise DatasetValidationError(
            f"Task {task_id!r} field 'metadata.source_urls' must not be empty"
        )

    source_type = metadata.get("source_type")
    if source_type not in {"public", "synthetic"}:
        raise DatasetValidationError(
            f"Task {task_id!r} field 'metadata.source_type' must be "
            "'public' or 'synthetic'"
        )

    return BenchmarkTask(
        task_id=task_id,
        dataset_id=dataset_id,
        task_type=_require_str(raw, "task_type"),
        query=_require_str(raw, "query"),
        context=context,
        expected_answer=_require_str(raw, "expected_answer"),
        expected_citations=tuple(citation_ids),
        metadata=dict(metadata),
    )


def _validate_context(raw_context: Any, *, task_id: str) -> tuple[ContextDocument, ...]:
    if not isinstance(raw_context, list):
        raise DatasetValidationError(f"Task {task_id!r} field 'context' must be a list")
    if not raw_context:
        raise DatasetValidationError(f"Task {task_id!r} field 'context' must not be empty")

    documents: list[ContextDocument] = []
    seen_document_ids: set[str] = set()
    for index, raw_document in enumerate(raw_context):
        if not isinstance(raw_document, Mapping):
            raise DatasetValidationError(
                f"Task {task_id!r} context[{index}] must be an object"
            )
        missing = sorted(_REQUIRED_CONTEXT_FIELDS.difference(raw_document))
        if missing:
            raise DatasetValidationError(
                f"Task {task_id!r} context[{index}] is missing field(s): "
                f"{', '.join(missing)}"
            )

        document = ContextDocument(
            document_id=_require_str(raw_document, "document_id"),
            title=_require_str(raw_document, "title"),
            text=_require_str(raw_document, "text"),
            source_url=_require_str(raw_document, "source_url"),
        )
        if document.document_id in seen_document_ids:
            raise DatasetValidationError(
                f"Task {task_id!r} has duplicate context document_id "
                f"{document.document_id!r}"
            )
        seen_document_ids.add(document.document_id)
        documents.append(document)

    return tuple(documents)


def _validate_str_list(raw: Any, *, field: str, task_id: str) -> list[str]:
    if not isinstance(raw, list):
        raise DatasetValidationError(f"Task {task_id!r} field {field!r} must be a list")
    values: list[str] = []
    for index, value in enumerate(raw):
        if not isinstance(value, str) or not value.strip():
            raise DatasetValidationError(
                f"Task {task_id!r} field {field!r}[{index}] must be a non-empty string"
            )
        values.append(value)
    return values


def _require_str(raw: Mapping[str, Any], field: str) -> str:
    value = raw[field]
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"Field {field!r} must be a non-empty string")
    return value


def _summary_memory(task: BenchmarkTask) -> str:
    parts: list[str] = []
    for document in task.context:
        first_sentence = document.text.split(".")[0].strip()
        parts.append(f"{first_sentence}. Source {document.document_id}.")
    return " ".join(parts)


def _structured_facts(task: BenchmarkTask) -> list[dict[str, str]]:
    facts: list[dict[str, str]] = []
    for document in task.context:
        first_sentence = document.text.split(".")[0].strip()
        facts.append({"claim": first_sentence, "source_id": document.document_id})
    return facts


def _gold_keywords(expected_answer: str) -> list[str]:
    phrases: list[str] = []
    for token in re.findall(r"[A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*)*", expected_answer):
        if len(token) > 2 and token.lower() not in _KEYWORD_STOPWORDS:
            phrases.append(token)

    words = [
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z0-9-]{4,}", expected_answer)
        if word.lower() not in _KEYWORD_STOPWORDS
    ]
    for word in words[:6]:
        phrases.append(word)
    return list(dict.fromkeys(phrases[:8]))


__all__ = [
    "BenchmarkTask",
    "ContextDocument",
    "DATASET_FILES",
    "DatasetValidationError",
    "available_datasets",
    "dataset_path",
    "load_dataset",
    "task_to_strategy_input",
    "validate_task_object",
]
