"""Compare the prefix-cache abstention variant against the memory-lane baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_budget_lab.metrics import read_trace_records, summarize_records


DEFAULT_MODELS = ["qwen2.5:3b", "qwen2.5:7b"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=Path("results/local-matrix"))
    parser.add_argument("--variant-root", type=Path, default=Path("results/abstention-variant"))
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/abstention_variant"))
    parser.add_argument("--dataset-id", default="synthetic_agent_memory_v0")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    args = parser.parse_args(argv)

    models = _split_csv(args.models)
    report = build_report(
        baseline_root=args.baseline_root,
        variant_root=args.variant_root,
        models=models,
        dataset_id=args.dataset_id,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out_dir / "summary.md").write_text(markdown_report(report), encoding="utf-8")
    print(args.out_dir)
    return 0


def build_report(
    *,
    baseline_root: Path,
    variant_root: Path,
    models: list[str],
    dataset_id: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for model in models:
        rows.append(
            _condition_row(
                root=baseline_root,
                model=model,
                strategy="prefix_cache_friendly",
                condition="baseline",
                dataset_id=dataset_id,
            )
        )
        rows.append(
            _condition_row(
                root=variant_root,
                model=model,
                strategy="prefix_cache_abstain",
                condition="abstain_variant",
                dataset_id=dataset_id,
            )
        )
    return {
        "dataset_id": dataset_id,
        "delta": "abstain_variant_minus_baseline",
        "rows": rows,
        "deltas": _delta_rows(rows),
    }


def markdown_report(report: dict[str, Any]) -> str:
    rows = report["rows"]
    deltas = report["deltas"]
    lines = [
        f'# Abstention Variant: {report["dataset_id"]}',
        "",
        "## Conditions",
        "",
        "| model | condition | strategy | n | errors | abstain correct | fact coverage | schema ok | useful answers | run |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    row["condition"],
                    row["strategy"],
                    str(row["n"]),
                    str(row["errors"]),
                    _fmt(row["avg_abstain_correct"]),
                    _fmt(row["avg_fact_coverage"]),
                    _fmt(row["avg_schema_ok"]),
                    str(row["useful_answers"]),
                    row["run_dir"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Deltas",
            "",
            "| model | delta abstain correct | delta fact coverage | delta schema ok | delta useful answers |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in deltas:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    _fmt_delta(row["delta_avg_abstain_correct"]),
                    _fmt_delta(row["delta_avg_fact_coverage"]),
                    _fmt_delta(row["delta_avg_schema_ok"]),
                    _fmt_delta(row["delta_useful_answers"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _condition_row(
    *,
    root: Path,
    model: str,
    strategy: str,
    condition: str,
    dataset_id: str,
) -> dict[str, Any]:
    run_dir = _latest_run_dir(root / _slug(model) / _slug(strategy))
    records = [record for record in read_trace_records(run_dir) if record.get("meta", {}).get("dataset_id") == dataset_id]
    if not records:
        raise RuntimeError(f"{run_dir} has no records for dataset {dataset_id}")
    summary = summarize_records(records)[0]
    return {
        "model": model,
        "condition": condition,
        "strategy": strategy,
        "dataset_id": dataset_id,
        "run_dir": str(run_dir),
        **summary,
    }


def _delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["model"], row["condition"]): row for row in rows}
    models = sorted({row["model"] for row in rows})
    deltas: list[dict[str, Any]] = []
    for model in models:
        baseline = by_key[(model, "baseline")]
        variant = by_key[(model, "abstain_variant")]
        deltas.append(
            {
                "model": model,
                "delta": "abstain_variant_minus_baseline",
                "delta_avg_abstain_correct": _subtract(variant["avg_abstain_correct"], baseline["avg_abstain_correct"]),
                "delta_avg_fact_coverage": _subtract(variant["avg_fact_coverage"], baseline["avg_fact_coverage"]),
                "delta_avg_schema_ok": _subtract(variant["avg_schema_ok"], baseline["avg_schema_ok"]),
                "delta_useful_answers": variant["useful_answers"] - baseline["useful_answers"],
            }
        )
    return deltas


def _latest_run_dir(path: Path) -> Path:
    run_dirs = sorted([child for child in path.iterdir() if child.is_dir()], key=lambda child: child.stat().st_mtime)
    if not run_dirs:
        raise RuntimeError(f"no run directories under {path}")
    return run_dirs[-1]


def _subtract(left: float | int | None, right: float | int | None) -> float | None:
    if left is None or right is None:
        return None
    return round(float(left) - float(right), 3)


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _fmt_delta(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value > 0 else ""
    if isinstance(value, float):
        return f"{sign}{value:.3f}"
    return f"{sign}{value}"


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value).strip("-")


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
