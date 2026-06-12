"""Paired task-level deltas from committed context-budget traces."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import random
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_budget_lab.metrics import read_trace_records


METRICS = ["fact_coverage", "citation_precision", "citation_recall", "abstain_correct"]
STRATEGY_ORDER = ["full_context", "rag_topk", "summary_memory", "structured_memory", "prefix_cache_friendly"]
MODEL_PAIR = ("qwen2.5:3b", "qwen2.5:7b")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/paired_deltas"))
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)

    report = build_report(args.results_root, resamples=args.resamples, seed=args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out_dir / "summary.md").write_text(markdown_report(report), encoding="utf-8")
    print(args.out_dir)
    return 0


def build_report(results_root: Path, *, resamples: int, seed: int) -> dict[str, Any]:
    arms = load_arms(results_root)
    rows: list[dict[str, Any]] = []
    rows.extend(strategy_pair_rows(arms, resamples=resamples, seed=seed))
    rows.extend(model_pair_rows(arms, resamples=resamples, seed=seed))
    return {
        "results_root": str(results_root),
        "metrics": METRICS,
        "resamples": resamples,
        "seed": seed,
        "rows": rows,
        "paired_only_separations": paired_only_separations(rows),
    }


def load_arms(results_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    arms: dict[tuple[str, str], dict[str, Any]] = {}
    for run_dir in find_run_dirs(results_root):
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        model = str(run["config"]["model"])
        strategy = str(run["config"]["strategies"][0])
        task_metrics: dict[str, dict[str, list[float]]] = {}
        for record in read_trace_records(run_dir):
            if record.get("error") is not None:
                continue
            task_id = str(record.get("meta", {}).get("task_id") or record.get("request_id", ""))
            metric_lists = task_metrics.setdefault(task_id, {metric: [] for metric in METRICS})
            for metric in METRICS:
                value = record.get("meta", {}).get(metric)
                if isinstance(value, bool):
                    metric_lists[metric].append(1.0 if value else 0.0)
                elif isinstance(value, int | float):
                    metric_lists[metric].append(float(value))
        arms[(model, strategy)] = {
            "model": model,
            "strategy": strategy,
            "run_dir": str(run_dir),
            "tasks": {
                task_id: {
                    metric: statistics.mean(values)
                    for metric, values in metric_lists.items()
                    if values
                }
                for task_id, metric_lists in task_metrics.items()
            },
        }
    return arms


def strategy_pair_rows(
    arms: dict[tuple[str, str], dict[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    models = sorted({model for model, _ in arms}, key=_model_sort_key)
    for model in models:
        strategies = [strategy for strategy in STRATEGY_ORDER if (model, strategy) in arms]
        for baseline_strategy, comparison_strategy in combinations(strategies, 2):
            baseline = arms[(model, baseline_strategy)]
            comparison = arms[(model, comparison_strategy)]
            rows.append(
                comparison_row(
                    comparison_type="strategy_pair",
                    baseline=baseline,
                    comparison=comparison,
                    resamples=resamples,
                    seed=seed,
                )
            )
    return rows


def model_pair_rows(
    arms: dict[tuple[str, str], dict[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline_model, comparison_model = MODEL_PAIR
    strategies = [strategy for strategy in STRATEGY_ORDER if (baseline_model, strategy) in arms and (comparison_model, strategy) in arms]
    for strategy in strategies:
        rows.append(
            comparison_row(
                comparison_type="model_pair",
                baseline=arms[(baseline_model, strategy)],
                comparison=arms[(comparison_model, strategy)],
                resamples=resamples,
                seed=seed,
            )
        )
    return rows


def comparison_row(
    *,
    comparison_type: str,
    baseline: dict[str, Any],
    comparison: dict[str, Any],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {}
    for metric in METRICS:
        paired = paired_metric(
            baseline["tasks"],
            comparison["tasks"],
            metric,
            resamples=resamples,
            seed=_metric_seed(seed, comparison_type, baseline["model"], baseline["strategy"], comparison["model"], comparison["strategy"], metric),
        )
        metrics[metric] = paired
    return {
        "comparison_type": comparison_type,
        "baseline_model": baseline["model"],
        "comparison_model": comparison["model"],
        "baseline_strategy": baseline["strategy"],
        "comparison_strategy": comparison["strategy"],
        "baseline_run_dir": baseline["run_dir"],
        "comparison_run_dir": comparison["run_dir"],
        "delta": "comparison_minus_baseline",
        "metrics": metrics,
    }


def paired_metric(
    baseline_tasks: dict[str, dict[str, float]],
    comparison_tasks: dict[str, dict[str, float]],
    metric: str,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    common_task_ids = sorted(
        task_id
        for task_id in set(baseline_tasks).intersection(comparison_tasks)
        if metric in baseline_tasks[task_id] and metric in comparison_tasks[task_id]
    )
    baseline_values = [baseline_tasks[task_id][metric] for task_id in common_task_ids]
    comparison_values = [comparison_tasks[task_id][metric] for task_id in common_task_ids]
    deltas = [right - left for left, right in zip(baseline_values, comparison_values, strict=True)]
    baseline_summary = summarize_values(baseline_values, resamples=resamples, seed=seed + 11)
    comparison_summary = summarize_values(comparison_values, resamples=resamples, seed=seed + 17)
    delta_summary = summarize_values(deltas, resamples=resamples, seed=seed + 23)
    per_arm_overlap = ci_overlap(baseline_summary, comparison_summary)
    paired_separation = excludes_zero(delta_summary)
    return {
        "n_tasks": len(common_task_ids),
        "baseline": baseline_summary,
        "comparison": comparison_summary,
        "paired_delta": delta_summary,
        "per_arm_ci_overlap": per_arm_overlap,
        "paired_separation": paired_separation,
        "paired_only_separation": bool(per_arm_overlap and paired_separation),
    }


def summarize_values(values: list[float], *, resamples: int, seed: int) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "value": None, "ci_low": None, "ci_high": None}
    value = statistics.mean(values)
    if len(values) == 1:
        rounded = round(value, 6)
        return {"n": 1, "value": rounded, "ci_low": rounded, "ci_high": rounded}
    rng = random.Random(seed)
    boot_values: list[float] = []
    for _ in range(resamples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        boot_values.append(statistics.mean(sample))
    boot_values.sort()
    low = boot_values[int(0.025 * (len(boot_values) - 1))]
    high = boot_values[int(0.975 * (len(boot_values) - 1))]
    return {"n": len(values), "value": round(value, 6), "ci_low": round(low, 6), "ci_high": round(high, 6)}


def ci_overlap(left: dict[str, float | int | None], right: dict[str, float | int | None]) -> bool | None:
    if left["ci_low"] is None or left["ci_high"] is None or right["ci_low"] is None or right["ci_high"] is None:
        return None
    return not (float(left["ci_high"]) < float(right["ci_low"]) or float(right["ci_high"]) < float(left["ci_low"]))


def excludes_zero(summary: dict[str, float | int | None]) -> bool:
    if summary["ci_low"] is None or summary["ci_high"] is None:
        return False
    return float(summary["ci_low"]) > 0.0 or float(summary["ci_high"]) < 0.0


def paired_only_separations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for row in rows:
        for metric, summary in row["metrics"].items():
            if summary["paired_only_separation"]:
                found.append(
                    {
                        "comparison_type": row["comparison_type"],
                        "baseline_model": row["baseline_model"],
                        "comparison_model": row["comparison_model"],
                        "baseline_strategy": row["baseline_strategy"],
                        "comparison_strategy": row["comparison_strategy"],
                        "metric": metric,
                        "n_tasks": summary["n_tasks"],
                        "paired_delta": summary["paired_delta"],
                    }
                )
    return found


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Paired Delta Reanalysis",
        "",
        "Deltas are comparison minus baseline. Each arm is first averaged per task across repeats, then bootstrapped over task-level deltas.",
        "",
        "## Paired-Only Separations",
        "",
    ]
    paired_only = report["paired_only_separations"]
    if paired_only:
        lines.extend(metric_rows_table(paired_only))
    else:
        lines.append("_No comparison had overlapping per-arm CIs but a paired CI excluding zero._")
    lines.extend(["", "## All Comparisons", ""])
    flat_rows: list[dict[str, Any]] = []
    for row in report["rows"]:
        for metric, summary in row["metrics"].items():
            flat_rows.append(
                {
                    "comparison_type": row["comparison_type"],
                    "baseline_model": row["baseline_model"],
                    "comparison_model": row["comparison_model"],
                    "baseline_strategy": row["baseline_strategy"],
                    "comparison_strategy": row["comparison_strategy"],
                    "metric": metric,
                    "n_tasks": summary["n_tasks"],
                    "baseline": summary["baseline"],
                    "comparison": summary["comparison"],
                    "paired_delta": summary["paired_delta"],
                    "per_arm_ci_overlap": summary["per_arm_ci_overlap"],
                    "paired_separation": summary["paired_separation"],
                    "paired_only_separation": summary["paired_only_separation"],
                }
            )
    lines.extend(comparison_table(flat_rows))
    return "\n".join(lines) + "\n"


def metric_rows_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| type | baseline | comparison | metric | n tasks | paired delta |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["comparison_type"],
                    _label(row["baseline_model"], row["baseline_strategy"]),
                    _label(row["comparison_model"], row["comparison_strategy"]),
                    row["metric"],
                    str(row["n_tasks"]),
                    _fmt_ci(row["paired_delta"]),
                ]
            )
            + " |"
        )
    return lines


def comparison_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| type | baseline | comparison | metric | n tasks | baseline | comparison | paired delta | per-arm overlap | paired separates | paired-only |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["comparison_type"],
                    _label(row["baseline_model"], row["baseline_strategy"]),
                    _label(row["comparison_model"], row["comparison_strategy"]),
                    row["metric"],
                    str(row["n_tasks"]),
                    _fmt_ci(row["baseline"]),
                    _fmt_ci(row["comparison"]),
                    _fmt_ci(row["paired_delta"]),
                    str(row["per_arm_ci_overlap"]),
                    str(row["paired_separation"]),
                    str(row["paired_only_separation"]),
                ]
            )
            + " |"
        )
    return lines


def find_run_dirs(results_root: Path) -> list[Path]:
    return sorted(path.parent for path in results_root.rglob("run.json") if (path.parent / "traces.jsonl").exists())


def _label(model: str, strategy: str) -> str:
    return f"{model} {strategy}"


def _fmt_ci(summary: dict[str, float | int | None]) -> str:
    value = summary["value"]
    if value is None:
        return "n/a"
    return f'{value:.3f} [{summary["ci_low"]:.3f}, {summary["ci_high"]:.3f}]'


def _metric_seed(seed: int, *parts: str) -> int:
    return seed + sum(ord(ch) for part in parts for ch in part)


def _model_sort_key(model: str) -> tuple[int, str]:
    try:
        size = model.rsplit(":", maxsplit=1)[1]
        if size.endswith("b"):
            return (int(float(size[:-1]) * 10), model)
    except (IndexError, ValueError):
        pass
    return (9999, model)


if __name__ == "__main__":
    raise SystemExit(main())
