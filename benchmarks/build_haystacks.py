"""Generate the v2 haystack datasets ({2, 8, 32} docs) from the v0 lanes.

Writes new JSONL files next to the v0 datasets; the v0 files are read-only
inputs and are never modified. Each generated file validates against the
same loader as the hand-written datasets.

    uv run python -m benchmarks.build_haystacks --out-dir datasets

Re-running with the same --seed reproduces byte-identical files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_budget_lab.datasets import (
    load_dataset,
    validate_task_object,
)
from context_budget_lab.haystack import HAYSTACK_SIZES, build_haystack_dataset


# Base lanes -> v2 dataset-id stem. Variants are "<stem>_h{N}".
# The abstain suite is itself a v2 source (built by build_abstention.py) and is
# also swept across haystack sizes so abstention can be tested under context
# pressure; run build_abstention.py before this script.
BASE_LANES: dict[str, str] = {
    "public_ai_policy_v0": "public_ai_policy_v2",
    "synthetic_agent_memory_v0": "synthetic_agent_memory_v2",
    "synthetic_agent_memory_abstain_v2": "synthetic_agent_memory_abstain_v2",
}


def variant_dataset_id(stem: str, size: int) -> str:
    return f"{stem}_h{size}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "datasets")
    parser.add_argument("--seed", type=int, default=20260614)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    lanes = {base_id: load_dataset(base_id) for base_id in BASE_LANES}
    # Distractors are pooled across all lanes so h32 can reach a true 32 docs;
    # the v0 corpus has only ~30 unique docs per lane (see decisions.md, A13).
    pool_tasks = [task for tasks in lanes.values() for task in tasks]

    written: list[tuple[str, int]] = []
    for base_id, stem in BASE_LANES.items():
        tasks = lanes[base_id]
        for size in HAYSTACK_SIZES:
            dataset_id = variant_dataset_id(stem, size)
            records = build_haystack_dataset(
                tasks,
                size,
                seed=args.seed,
                new_dataset_id=dataset_id,
                pool_tasks=pool_tasks,
            )
            # Validate every record through the real loader contract before writing.
            for record in records:
                validate_task_object(record, expected_dataset_id=dataset_id)

            out_path = args.out_dir / f"{dataset_id}.jsonl"
            lines = [json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records]
            out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            written.append((dataset_id, len(records)))

    for dataset_id, count in written:
        print(f"{dataset_id}: {count} tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
