"""Build {2, 8, 32}-doc haystack variants of the v0 tasks (A13, v2).

For each base task we keep every gold document and pad with distractor
documents sampled from *other* tasks in the same lane until the context
reaches a target size. Sampling is seeded per (base task, size) so the
generated datasets are byte-reproducible from this script.

Gold documents are always retained: if a task already has more gold docs
than the target size, the variant keeps all of them and the actual size
exceeds the target floor. That is the honest behaviour and is recorded in
``metadata.haystack_size``.

Original v0 records are never mutated; variants get new ``dataset_id`` and
``task_id`` values suffixed with ``_h{N}``.
"""

from __future__ import annotations

import random
from typing import Any, Mapping

from context_budget_lab.datasets import BenchmarkTask, ContextDocument


HAYSTACK_SIZES = (2, 8, 32)


def _doc_to_raw(document: ContextDocument) -> dict[str, str]:
    return {
        "document_id": document.document_id,
        "title": document.title,
        "text": document.text,
        "source_url": document.source_url,
    }


def _gold_document_ids(task: BenchmarkTask) -> set[str]:
    """Documents that must always survive: everything not flagged a distractor."""

    metadata = task.metadata if isinstance(task.metadata, Mapping) else {}
    distractors = set(metadata.get("distractor_document_ids", []) or [])
    return {doc.document_id for doc in task.context if doc.document_id not in distractors}


def _distractor_pool(
    pool_tasks: list[BenchmarkTask], current_task_id: str
) -> list[ContextDocument]:
    """All docs from the other tasks in ``pool_tasks``, deduped by id, stable order.

    ``pool_tasks`` may span lanes — the v0 corpus has only ~30 unique docs per
    lane, too few for a true 32-doc haystack, so distractors are drawn from
    both lanes. Document ids are lane-prefixed (``nist-*`` vs ``mem-*``) so
    there is no id collision; distractors stay non-answering for the query.
    """

    pool: list[ContextDocument] = []
    seen: set[str] = set()
    for task in pool_tasks:
        if task.task_id == current_task_id:
            continue
        for doc in task.context:
            if doc.document_id in seen:
                continue
            seen.add(doc.document_id)
            pool.append(doc)
    return pool


def build_haystack_task(
    task: BenchmarkTask,
    pool_tasks: list[BenchmarkTask],
    target_size: int,
    *,
    seed: int,
    new_dataset_id: str,
) -> dict[str, Any]:
    """Build one haystack variant of ``task`` at ``target_size`` documents.

    ``pool_tasks`` is the set of tasks distractors are sampled from (may span
    lanes); gold docs and the task itself come from ``task``.
    """

    gold_ids = _gold_document_ids(task)
    gold_docs = [doc for doc in task.context if doc.document_id in gold_ids]

    present_ids = {doc.document_id for doc in task.context}
    pool = [
        doc
        for doc in _distractor_pool(pool_tasks, task.task_id)
        if doc.document_id not in present_ids
    ]

    wanted_distractors = max(0, target_size - len(gold_docs))
    rng = random.Random(f"{seed}:{task.task_id}:{target_size}")
    sample_n = min(wanted_distractors, len(pool))
    distractors = rng.sample(pool, sample_n) if sample_n else []

    documents = gold_docs + distractors
    # Seeded shuffle so gold docs are not always first (guards against
    # position bias in full_context / rag_topk ranking).
    order_rng = random.Random(f"{seed}:order:{task.task_id}:{target_size}")
    order_rng.shuffle(documents)

    base_metadata = dict(task.metadata) if isinstance(task.metadata, Mapping) else {}
    distractor_ids = [doc.document_id for doc in distractors]
    metadata = {
        **base_metadata,
        "distractor_document_ids": distractor_ids,
        "haystack_size": len(documents),
        "haystack_target": target_size,
        "base_task_id": task.task_id,
        "base_dataset_id": task.dataset_id,
    }

    base_local_id = task.task_id.split("/", 1)[-1]
    return {
        "task_id": f"{new_dataset_id}/{base_local_id}",
        "dataset_id": new_dataset_id,
        "task_type": task.task_type,
        "query": task.query,
        "context": [_doc_to_raw(doc) for doc in documents],
        "expected_answer": task.expected_answer,
        "expected_citations": list(task.expected_citations),
        "expected_facts": list(task.expected_facts),
        "answer_schema": task.answer_schema,
        "metadata": metadata,
    }


def build_haystack_dataset(
    tasks: list[BenchmarkTask],
    target_size: int,
    *,
    seed: int,
    new_dataset_id: str,
    pool_tasks: list[BenchmarkTask] | None = None,
) -> list[dict[str, Any]]:
    """Build every task in ``tasks`` at ``target_size``.

    ``pool_tasks`` is where distractors are sampled from; it defaults to
    ``tasks`` (same-lane) but A13 passes the union of both lanes so h32 can
    reach a true 32 documents.
    """

    pool = pool_tasks if pool_tasks is not None else tasks
    return [
        build_haystack_task(
            task,
            pool,
            target_size,
            seed=seed,
            new_dataset_id=new_dataset_id,
        )
        for task in tasks
    ]
