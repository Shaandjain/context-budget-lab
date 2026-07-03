"""Analyze the A13 v2 haystack-size sweep.

The script expects clean one-condition run directories under a matrix root:

    results/v2-matrix/<model>/<strategy>/<run_id>/{run.json,traces.jsonl}

It reports completeness first. Hypothesis summaries are computed only from
successful trace rows, so capped/429 runs show up as incomplete instead of
silently becoming science evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_budget_lab.datasets import load_dataset
from context_budget_lab.metrics import read_trace_records


HAYSTACK_SIZES = [2, 8, 32]
STRATEGIES = ["full_context", "rag_topk", "summary_memory", "structured_memory"]
COMPRESSION_STRATEGIES = ["summary_memory", "structured_memory"]
SIZE_RE = re.compile(r"_h(?P<size>\d+)$")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/v2_sweep"))
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
    runs = load_runs(results_root)
    arms = load_arms(runs)
    h1_rows = h1_rag_vs_full_rows(arms, resamples=resamples, seed=seed)
    h3_rows = h3_compression_rows(arms, resamples=resamples, seed=seed)
    frontier_rows = frontier_rows_by_size(arms)
    return {
        "results_root": str(results_root),
        "resamples": resamples,
        "seed": seed,
        "completeness": [run["completeness"] for run in runs],
        "clean": all(run["completeness"]["complete"] for run in runs) and bool(runs),
        "h1": {
            "verdicts": h1_verdicts(h1_rows),
            "rows": h1_rows,
        },
        "h3": {
            "verdicts": h3_verdicts(h3_rows),
            "rows": h3_rows,
        },
        "frontier": frontier_rows,
    }


def load_runs(results_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for run_json in sorted(results_root.rglob("run.json")):
        run_dir = run_json.parent
        traces_path = run_dir / "traces.jsonl"
        if not traces_path.exists():
            continue
        run = json.loads(run_json.read_text(encoding="utf-8"))
        records = read_trace_records(run_dir)
        datasets = list(run["config"].get("datasets", []))
        repeats = int(run["config"].get("repeats", 1))
        limit = run["config"].get("limit")
        expected = expected_request_count(datasets, repeats=repeats, limit=limit)
        errors = sum(1 for record in records if record.get("error") is not None)
        model = str(run["config"]["model"])
        strategy = str(run["config"]["strategies"][0])
        runs.append(
            {
                "model": model,
                "strategy": strategy,
                "run_dir": str(run_dir),
                "records": records,
                "completeness": {
                    "model": model,
                    "strategy": strategy,
                    "run_dir": str(run_dir),
                    "expected_records": expected,
                    "observed_records": len(records),
                    "errors": errors,
                    "complete": len(records) == expected and errors == 0,
                },
            }
        )
    return runs


def expected_request_count(dataset_ids: list[str], *, repeats: int, limit: int | None) -> int:
    total_tasks = 0
    for dataset_id in dataset_ids:
        task_count = len(load_dataset(dataset_id))
        total_tasks += min(task_count, limit) if limit is not None else task_count
    return total_tasks * repeats


def load_arms(runs: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    arms: dict[tuple[str, str, int], dict[str, Any]] = {}
    for run in runs:
        model = run["model"]
        strategy = run["strategy"]
        task_metrics: dict[int, dict[str, dict[str, list[float]]]] = {}
        timing: dict[int, dict[str, list[float]]] = {}
        for record in run["records"]:
            if record.get("error") is not None:
                continue
            meta = record.get("meta", {})
            if not isinstance(meta, dict):
                continue
            dataset_id = str(meta.get("dataset_id", ""))
            size = haystack_size(dataset_id)
            if size is None:
                continue
            task_id = str(meta.get("task_id", record.get("request_id", "")))
            metric_lists = task_metrics.setdefault(size, {}).setdefault(task_id, {"fact_coverage": []})
            fact_coverage = meta.get("fact_coverage")
            if isinstance(fact_coverage, int | float):
                metric_lists["fact_coverage"].append(float(fact_coverage))
            timing_lists = timing.setdefault(size, {"latency_s": [], "input_tokens": []})
            latency_s = record.get("latency_s")
            input_tokens = record.get("input_tokens")
            if isinstance(latency_s, int | float):
                timing_lists["latency_s"].append(float(latency_s))
            if isinstance(input_tokens, int | float):
                timing_lists["input_tokens"].append(float(input_tokens))
        for size, task_values in task_metrics.items():
            task_means = {
                task_id: {
                    metric: statistics.mean(values)
                    for metric, values in metric_lists.items()
                    if values
                }
                for task_id, metric_lists in task_values.items()
            }
            arms[(model, strategy, size)] = {
                "model": model,
                "strategy": strategy,
                "haystack_size": size,
                "tasks": task_means,
                "latency_p50_s": percentile(timing.get(size, {}).get("latency_s", []), 0.5),
                "input_tokens_mean": mean_or_none(timing.get(size, {}).get("input_tokens", [])),
            }
    return arms


def h1_rag_vs_full_rows(
    arms: dict[tuple[str, str, int], dict[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in sorted({key[0] for key in arms}):
        for size in HAYSTACK_SIZES:
            baseline = arms.get((model, "full_context", size))
            comparison = arms.get((model, "rag_topk", size))
            if baseline is None or comparison is None:
                continue
            rows.append(
                paired_row(
                    model=model,
                    haystack_size=size,
                    baseline_strategy="full_context",
                    comparison_strategy="rag_topk",
                    baseline_tasks=baseline["tasks"],
                    comparison_tasks=comparison["tasks"],
                    resamples=resamples,
                    seed=seed,
                )
            )
    return rows


def h3_compression_rows(
    arms: dict[tuple[str, str, int], dict[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in sorted({key[0] for key in arms}):
        for compression_strategy in COMPRESSION_STRATEGIES:
            for size in HAYSTACK_SIZES:
                baseline = arms.get((model, "rag_topk", size))
                comparison = arms.get((model, compression_strategy, size))
                if baseline is None or comparison is None:
                    continue
                rows.append(
                    paired_row(
                        model=model,
                        haystack_size=size,
                        baseline_strategy="rag_topk",
                        comparison_strategy=compression_strategy,
                        baseline_tasks=baseline["tasks"],
                        comparison_tasks=comparison["tasks"],
                        resamples=resamples,
                        seed=seed,
                    )
                )
    return rows


def paired_row(
    *,
    model: str,
    haystack_size: int,
    baseline_strategy: str,
    comparison_strategy: str,
    baseline_tasks: dict[str, dict[str, float]],
    comparison_tasks: dict[str, dict[str, float]],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    common_task_ids = sorted(
        task_id
        for task_id in set(baseline_tasks).intersection(comparison_tasks)
        if "fact_coverage" in baseline_tasks[task_id] and "fact_coverage" in comparison_tasks[task_id]
    )
    deltas = [
        comparison_tasks[task_id]["fact_coverage"] - baseline_tasks[task_id]["fact_coverage"]
        for task_id in common_task_ids
    ]
    summary = summarize_values(
        deltas,
        resamples=resamples,
        seed=metric_seed(seed, model, str(haystack_size), baseline_strategy, comparison_strategy),
    )
    return {
        "model": model,
        "haystack_size": haystack_size,
        "baseline_strategy": baseline_strategy,
        "comparison_strategy": comparison_strategy,
        "metric": "fact_coverage",
        "delta": "comparison_minus_baseline",
        "n_tasks": len(common_task_ids),
        "paired_delta": summary,
    }


def frontier_rows_by_size(arms: dict[tuple[str, str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in sorted({key[0] for key in arms}):
        for size in HAYSTACK_SIZES:
            size_arms = [arm for key, arm in arms.items() if key[0] == model and key[2] == size]
            for arm in sorted(size_arms, key=lambda item: item["strategy"]):
                fact_values = [metrics["fact_coverage"] for metrics in arm["tasks"].values() if "fact_coverage" in metrics]
                rows.append(
                    {
                        "model": model,
                        "haystack_size": size,
                        "strategy": arm["strategy"],
                        "fact_coverage": round(statistics.mean(fact_values), 6) if fact_values else None,
                        "latency_p50_s": arm["latency_p50_s"],
                        "input_tokens_mean": arm["input_tokens_mean"],
                        "pareto_optimal": is_pareto_optimal(arm, size_arms),
                    }
                )
    return rows


def is_pareto_optimal(arm: dict[str, Any], candidates: list[dict[str, Any]]) -> bool:
    quality = arm_quality(arm)
    latency = arm["latency_p50_s"]
    tokens = arm["input_tokens_mean"]
    if quality is None or latency is None or tokens is None:
        return False
    for candidate in candidates:
        if candidate is arm:
            continue
        other_quality = arm_quality(candidate)
        other_latency = candidate["latency_p50_s"]
        other_tokens = candidate["input_tokens_mean"]
        if other_quality is None or other_latency is None or other_tokens is None:
            continue
        weakly_better = other_quality >= quality and other_latency <= latency and other_tokens <= tokens
        strictly_better = other_quality > quality or other_latency < latency or other_tokens < tokens
        if weakly_better and strictly_better:
            return False
    return True


def arm_quality(arm: dict[str, Any]) -> float | None:
    values = [metrics["fact_coverage"] for metrics in arm["tasks"].values() if "fact_coverage" in metrics]
    return statistics.mean(values) if values else None


def h1_verdicts(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    verdicts: list[dict[str, str]] = []
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)
    for model, model_rows in sorted(by_model.items()):
        by_size = {row["haystack_size"]: row["paired_delta"] for row in model_rows}
        if not all(size in by_size for size in HAYSTACK_SIZES):
            verdict = "blocked"
            rationale = "missing one or more haystack sizes"
        elif by_size[32]["ci_low"] is not None and float(by_size[32]["ci_low"]) > 0.0 and _nondecreasing([by_size[size]["value"] for size in HAYSTACK_SIZES]):
            verdict = "supported"
            rationale = "rag_topk advantage increases and h32 excludes zero"
        elif by_size[32]["ci_high"] is not None and float(by_size[32]["ci_high"]) < 0.0:
            verdict = "refuted"
            rationale = "h32 excludes zero in the opposite direction"
        else:
            verdict = "inconclusive"
            rationale = "h32 does not show a positive paired advantage with a CI excluding zero"
        verdicts.append({"model": model, "hypothesis": "H1", "verdict": verdict, "rationale": rationale})
    return verdicts


def h3_verdicts(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    verdicts: list[dict[str, str]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["model"], row["comparison_strategy"]), []).append(row)
    for (model, strategy), strategy_rows in sorted(grouped.items()):
        by_size = {row["haystack_size"]: row["paired_delta"] for row in strategy_rows}
        if not all(size in by_size for size in HAYSTACK_SIZES):
            verdict = "blocked"
            rationale = "missing one or more haystack sizes"
        elif by_size[32]["ci_high"] is not None and float(by_size[32]["ci_high"]) < 0.0 and float(by_size[32]["value"]) < float(by_size[2]["value"]):
            verdict = "supported"
            rationale = f"{strategy} is farther below rag_topk at h32 than h2, with h32 CI below zero"
        elif by_size[32]["ci_low"] is not None and float(by_size[32]["ci_low"]) > 0.0:
            verdict = "refuted"
            rationale = f"{strategy} beats rag_topk at h32"
        else:
            verdict = "inconclusive"
            rationale = f"{strategy} does not show a clear h32 degradation relative to rag_topk"
        verdicts.append({"model": model, "hypothesis": "H3", "strategy": strategy, "verdict": verdict, "rationale": rationale})
    return verdicts


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# A13 v2 Haystack Sweep",
        "",
        f"Results root: `{report['results_root']}`.",
        f"Clean matrix: `{report['clean']}`.",
        "",
        "## Completeness",
        "",
        "| model | strategy | observed | expected | errors | complete | run |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["completeness"]:
        lines.append(
            f"| {row['model']} | {row['strategy']} | {row['observed_records']} | {row['expected_records']} | {row['errors']} | {row['complete']} | `{row['run_dir']}` |"
        )

    lines.extend(["", "## H1: RAG vs Full Context", ""])
    lines.extend(verdict_table(report["h1"]["verdicts"]))
    lines.extend(paired_delta_table(report["h1"]["rows"]))

    lines.extend(["", "## H3: Compression vs RAG", ""])
    lines.extend(verdict_table(report["h3"]["verdicts"]))
    lines.extend(paired_delta_table(report["h3"]["rows"]))

    lines.extend(["", "## Frontier", ""])
    lines.extend(frontier_table(report["frontier"]))
    return "\n".join(lines) + "\n"


def verdict_table(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return ["_No matched rows yet._", ""]
    keys = ["model", "hypothesis", "strategy", "verdict", "rationale"]
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row.get(key, "") for key in keys) + " |")
    lines.append("")
    return lines


def paired_delta_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["_No matched paired deltas yet._"]
    lines = [
        "| model | h | comparison | n tasks | fact coverage delta |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        comparison = f"{row['comparison_strategy']} - {row['baseline_strategy']}"
        lines.append(
            f"| {row['model']} | {row['haystack_size']} | {comparison} | {row['n_tasks']} | {format_ci(row['paired_delta'])} |"
        )
    return lines


def frontier_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["_No frontier rows yet._"]
    lines = [
        "| model | h | strategy | fact coverage | p50 latency s | mean input tokens | Pareto optimal |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    str(row["haystack_size"]),
                    row["strategy"],
                    format_number(row["fact_coverage"]),
                    format_number(row["latency_p50_s"]),
                    format_number(row["input_tokens_mean"]),
                    str(row["pareto_optimal"]),
                ]
            )
            + " |"
        )
    return lines


def haystack_size(dataset_id: str) -> int | None:
    match = SIZE_RE.search(dataset_id)
    if match is None:
        return None
    return int(match.group("size"))


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


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 6)


def mean_or_none(values: list[float]) -> float | None:
    return round(statistics.mean(values), 6) if values else None


def _nondecreasing(values: list[float | int | None]) -> bool:
    if any(value is None for value in values):
        return False
    return all(float(left) <= float(right) for left, right in zip(values, values[1:]))


def format_ci(summary: dict[str, float | int | None]) -> str:
    if summary["value"] is None:
        return "n/a"
    return f'{summary["value"]:.3f} [{summary["ci_low"]:.3f}, {summary["ci_high"]:.3f}]'


def format_number(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def metric_seed(seed: int, *parts: str) -> int:
    return seed + sum(ord(ch) for part in parts for ch in part)


if __name__ == "__main__":
    raise SystemExit(main())
