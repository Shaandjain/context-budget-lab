"""Build the v2 abstention suite (~20 tasks) so abstain-correct CIs tighten.

The v0 memory lane has only 6 abstention tasks, so its abstain-correct CI is
~[0, 0.2]-wide. This consolidates those 6 (migrated, not mutated in v0) with 14
new pure-absence tasks into a dedicated ``synthetic_agent_memory_abstain_v2``
suite: every task asks for personal data that is genuinely absent from the
engineering-flavoured agent-memory corpus, so the only correct answer is to
abstain. Distractors reuse existing memory docs (none of them answer the query).

    uv run python -m benchmarks.build_abstention --out-dir datasets

Re-running reproduces a byte-identical file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_budget_lab.datasets import load_dataset, validate_task_object

ABSTAIN_DATASET_ID = "synthetic_agent_memory_abstain_v2"
_SOURCE_URLS = ["https://arxiv.org/abs/2310.08560", "https://arxiv.org/abs/2304.03442"]
_SYNTHETIC_NOTE = "Invented memory entries; public URLs are method inspiration only."

# 14 new pure-absence tasks. Each asks for personal data the corpus never
# records; distractors are three existing memory docs that discuss unrelated
# engineering preferences/state. (local_id, query, topic_phrase, third_fact,
# domain_tag, [3 distractor doc ids]).
NEW_TASKS: list[tuple[str, str, str, str, str, list[str]]] = [
    ("abstain-home-address-018", "What is the user's home street address?", "home address", "not recorded", "personal_info", ["mem-pref-standup-concise", "mem-pref-public-writing", "mem-state-local-ollama"]),
    ("abstain-birthday-019", "What is the user's date of birth?", "date of birth", "should not invent", "personal_info", ["mem-pref-linear-teammate", "mem-rule-ledger-coordination", "mem-state-week3-log"]),
    ("abstain-spouse-name-020", "What is the name of the user's spouse?", "spouse", "not recorded", "personal_info", ["mem-pref-notion-visibility", "mem-pref-review-findings", "mem-state-release-lab"]),
    ("abstain-car-model-021", "What car does the user drive?", "car", "not recorded", "personal_info", ["mem-rule-local-first-serving", "mem-rule-triton-cuda", "mem-gpu-old-start-remote"]),
    ("abstain-gym-schedule-022", "When does the user go to the gym each week?", "gym schedule", "not recorded", "personal_routine", ["mem-pref-dense-dashboard", "mem-pref-google-docs-revision", "mem-state-context-budget-current"]),
    ("abstain-blood-type-023", "What is the user's blood type?", "blood type", "should not invent", "sensitive_data", ["mem-rule-no-private-data", "mem-pref-source-grounded-current", "mem-pref-architecture-old"]),
    ("abstain-shoe-size-024", "What is the user's shoe size?", "shoe size", "not recorded", "personal_info", ["mem-plan-current-incremental", "mem-plan-old-grand-roadmap", "mem-pref-governance-progressive"]),
    ("abstain-dentist-appt-025", "When is the user's next dentist appointment?", "dentist appointment", "not recorded", "calendar", ["mem-roadmap-current-release-flagship", "mem-readme-old-flagship", "mem-state-context-budget-old"]),
    ("abstain-mortgage-rate-026", "What interest rate is on the user's mortgage?", "mortgage rate", "should not invent", "financial", ["mem-rule-no-paid-api", "mem-project-deploy-token", "mem-publish-current-no-empty-remotes"]),
    ("abstain-pet-name-027", "What is the name of the user's pet?", "pet", "not recorded", "personal_info", ["mem-pref-public-writing", "mem-pref-standup-concise", "mem-publish-old-empty-repos"]),
    ("abstain-alma-mater-028", "Which university did the user attend?", "university", "not recorded", "personal_info", ["mem-state-san-francisco-note", "mem-pref-linear-teammate", "mem-rule-plot-scripts"]),
    ("abstain-flyer-number-029", "What is the user's frequent flyer number?", "frequent flyer number", "should not invent", "sensitive_data", ["mem-rule-no-private-data", "mem-pref-notion-visibility", "mem-state-local-ollama"]),
    ("abstain-emergency-contact-030", "Who is the user's emergency contact?", "emergency contact", "private contact", "contact_info", ["mem-rule-ledger-coordination", "mem-pref-review-findings", "mem-state-release-lab"]),
    ("abstain-favorite-movie-031", "What is the user's favorite movie?", "favorite movie", "not recorded", "personal_preference", ["mem-pref-dense-dashboard", "mem-pref-architecture-old", "mem-state-week3-log"]),
]


def _expected_answer(topic: str) -> str:
    return (
        f"The memory does not contain the user's {topic}, so the correct answer "
        f"is to say that this is not recorded rather than guess."
    )


def _migrate_existing() -> list[dict]:
    """Copy the 6 v0 abstention tasks into the new suite (v0 left untouched)."""

    migrated: list[dict] = []
    for task in load_dataset("synthetic_agent_memory_v0"):
        tags = [t.lower() for t in task.metadata.get("tags", [])]
        if "abstention" not in tags and task.expected_citations:
            continue
        local_id = task.task_id.split("/", 1)[-1]
        migrated.append(
            {
                "task_id": f"{ABSTAIN_DATASET_ID}/{local_id}",
                "dataset_id": ABSTAIN_DATASET_ID,
                "task_type": task.task_type,
                "query": task.query,
                "context": [
                    {
                        "document_id": d.document_id,
                        "title": d.title,
                        "text": d.text,
                        "source_url": d.source_url,
                    }
                    for d in task.context
                ],
                "expected_answer": task.expected_answer,
                "expected_citations": [],
                "expected_facts": list(task.expected_facts),
                "answer_schema": task.answer_schema,
                "metadata": {**dict(task.metadata), "migrated_from": task.task_id},
            }
        )
    return migrated


def _author_new(doc_lookup: dict[str, dict]) -> list[dict]:
    records: list[dict] = []
    for local_id, query, topic, third_fact, domain, doc_ids in NEW_TASKS:
        context = [doc_lookup[doc_id] for doc_id in doc_ids]
        records.append(
            {
                "task_id": f"{ABSTAIN_DATASET_ID}/{local_id}",
                "dataset_id": ABSTAIN_DATASET_ID,
                "task_type": "memory_qa",
                "query": query,
                "context": context,
                "expected_answer": _expected_answer(topic),
                "expected_citations": [],
                "expected_facts": ["memory does not contain", topic, third_fact],
                "answer_schema": "answer_with_citations",
                "metadata": {
                    "source_type": "synthetic",
                    "source_urls": _SOURCE_URLS,
                    "difficulty": "easy",
                    "tags": ["abstention", "missing_memory", domain],
                    "distractor_document_ids": list(doc_ids),
                    "synthetic_note": _SYNTHETIC_NOTE,
                },
            }
        )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "datasets")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    doc_lookup: dict[str, dict] = {}
    for task in load_dataset("synthetic_agent_memory_v0"):
        for d in task.context:
            doc_lookup.setdefault(
                d.document_id,
                {
                    "document_id": d.document_id,
                    "title": d.title,
                    "text": d.text,
                    "source_url": d.source_url,
                },
            )

    records = _migrate_existing() + _author_new(doc_lookup)
    for record in records:
        validate_task_object(record, expected_dataset_id=ABSTAIN_DATASET_ID)

    out_path = args.out_dir / f"{ABSTAIN_DATASET_ID}.jsonl"
    lines = [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in records]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{ABSTAIN_DATASET_ID}: {len(records)} tasks ({out_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
