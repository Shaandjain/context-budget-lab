"""Build bootstrap summaries and a frontier SVG from committed trace dirs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_budget_lab.metrics import read_trace_records

MetricFn = Callable[[list[dict[str, Any]]], list[float]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/frontier"))
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)

    rows = summarize_matrix(args.results_root, resamples=args.resamples, seed=args.seed)
    deltas = compare_models(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out_dir / "summary.md").write_text(markdown_table(rows), encoding="utf-8")
    (args.out_dir / "model_deltas.json").write_text(json.dumps(deltas, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out_dir / "model_deltas.md").write_text(delta_markdown_table(deltas), encoding="utf-8")
    (args.out_dir / "frontier.svg").write_text(frontier_svg(rows), encoding="utf-8")
    print(args.out_dir)
    return 0


def summarize_matrix(results_root: Path, *, resamples: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in find_run_dirs(results_root):
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        records = read_trace_records(run_dir)
        ok_records = [record for record in records if record.get("error") is None]
        row = {
            "model": str(run["config"]["model"]),
            "strategy": str(run["config"]["strategies"][0]),
            "run_dir": str(run_dir),
            "n": len(records),
            "errors": len(records) - len(ok_records),
            "source": run["source"],
            "seed": run["config"]["seed"],
            "repeats": run["config"]["repeats"],
            "metrics": {},
        }
        for metric_name, values in metric_values(ok_records).items():
            row["metrics"][metric_name] = summarize_values(
                values,
                resamples=resamples,
                seed=_metric_seed(seed, row["model"], row["strategy"], metric_name),
                stat=metric_stat(metric_name),
            )
        rows.append(row)
    return sorted(rows, key=lambda item: (item["model"], item["strategy"]))


def find_run_dirs(results_root: Path) -> list[Path]:
    return sorted(path.parent for path in results_root.rglob("run.json") if (path.parent / "traces.jsonl").exists())


def compare_models(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    models = sorted({row["model"] for row in rows})
    if len(models) < 2:
        return []
    baseline_model, comparison_model = _comparison_pair(models)
    by_key = {(row["model"], row["strategy"]): row for row in rows}
    strategies = sorted(
        {
            row["strategy"]
            for row in rows
            if (baseline_model, row["strategy"]) in by_key and (comparison_model, row["strategy"]) in by_key
        }
    )
    deltas: list[dict[str, Any]] = []
    for strategy in strategies:
        baseline = by_key[(baseline_model, strategy)]
        comparison = by_key[(comparison_model, strategy)]
        metric_deltas: dict[str, dict[str, float | int | None]] = {}
        for metric_name in delta_metric_names():
            baseline_metric = baseline["metrics"][metric_name]
            comparison_metric = comparison["metrics"][metric_name]
            baseline_value = baseline_metric["value"]
            comparison_value = comparison_metric["value"]
            delta = None
            if baseline_value is not None and comparison_value is not None:
                delta = round(float(comparison_value) - float(baseline_value), 6)
            metric_deltas[metric_name] = {
                "baseline": baseline_value,
                "comparison": comparison_value,
                "delta": delta,
                "baseline_ci_low": baseline_metric["ci_low"],
                "baseline_ci_high": baseline_metric["ci_high"],
                "comparison_ci_low": comparison_metric["ci_low"],
                "comparison_ci_high": comparison_metric["ci_high"],
            }
        deltas.append(
            {
                "strategy": strategy,
                "baseline_model": baseline_model,
                "comparison_model": comparison_model,
                "delta": "comparison_minus_baseline",
                "baseline_run_dir": baseline["run_dir"],
                "comparison_run_dir": comparison["run_dir"],
                "metrics": metric_deltas,
            }
        )
    return deltas


def delta_metric_names() -> list[str]:
    return [
        "fact_coverage",
        "citation_precision",
        "citation_recall",
        "schema_ok",
        "abstain_correct",
        "latency_p50_s",
        "input_tokens_mean",
    ]


def _comparison_pair(models: list[str]) -> tuple[str, str]:
    if "qwen2.5:3b" in models and "qwen2.5:7b" in models:
        return "qwen2.5:3b", "qwen2.5:7b"
    return models[0], models[1]


def metric_values(records: list[dict[str, Any]]) -> dict[str, list[float]]:
    return {
        "fact_coverage": _meta_values(records, "fact_coverage"),
        "citation_precision": _meta_values(records, "citation_precision"),
        "citation_recall": _meta_values(records, "citation_recall"),
        "schema_ok": _meta_values(records, "schema_ok"),
        "abstain_correct": _meta_values(records, "abstain_correct"),
        "latency_p50_s": [record["latency_s"] for record in records],
        "latency_p95_s": [record["latency_s"] for record in records],
        "ttft_p50_s": _streaming_record_values(records, "ttft_s"),
        "tpot_mean_s": _streaming_record_values(records, "tpot_s"),
        "decode_tokens_per_s_mean": _streaming_meta_values(records, "decode_tokens_per_s"),
        "input_tokens_mean": [record["input_tokens"] for record in records],
        "output_tokens_mean": [record["output_tokens"] for record in records],
    }


def summarize_values(
    values: list[float],
    *,
    resamples: int,
    seed: int,
    stat: Callable[[list[float]], float] = statistics.mean,
) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "value": None, "ci_low": None, "ci_high": None}

    value = stat(values)
    if len(values) == 1:
        return {"n": 1, "value": round(value, 6), "ci_low": round(value, 6), "ci_high": round(value, 6)}

    rng = random.Random(seed)
    boot_values: list[float] = []
    for _ in range(resamples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        boot_values.append(stat(sample))
    boot_values.sort()
    low = boot_values[int(0.025 * (len(boot_values) - 1))]
    high = boot_values[int(0.975 * (len(boot_values) - 1))]
    return {"n": len(values), "value": round(value, 6), "ci_low": round(low, 6), "ci_high": round(high, 6)}


def metric_stat(metric_name: str) -> Callable[[list[float]], float]:
    if metric_name == "latency_p50_s":
        return lambda values: _percentile(values, 0.5)
    if metric_name == "latency_p95_s":
        return lambda values: _percentile(values, 0.95)
    return statistics.mean


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "model",
        "strategy",
        "n",
        "errors",
        "fact coverage",
        "citation precision",
        "citation recall",
        "schema ok",
        "abstain correct",
        "p50 latency s",
        "p50 TTFT s",
        "mean TPOT s",
        "mean decode tok/s",
        "mean input tokens",
        "source",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    row["strategy"],
                    str(row["n"]),
                    str(row["errors"]),
                    _fmt_ci(metrics["fact_coverage"]),
                    _fmt_ci(metrics["citation_precision"]),
                    _fmt_ci(metrics["citation_recall"]),
                    _fmt_ci(metrics["schema_ok"]),
                    _fmt_ci(metrics["abstain_correct"]),
                    _fmt_ci(metrics["latency_p50_s"]),
                    _fmt_ci(metrics["ttft_p50_s"]),
                    _fmt_ci(metrics["tpot_mean_s"]),
                    _fmt_ci(metrics["decode_tokens_per_s_mean"]),
                    _fmt_ci(metrics["input_tokens_mean"]),
                    row["run_dir"],
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def delta_markdown_table(deltas: list[dict[str, Any]]) -> str:
    if not deltas:
        return "_No matched model pairs found._\n"
    comparison = f'{deltas[0]["comparison_model"]} - {deltas[0]["baseline_model"]}'
    headers = [
        "strategy",
        "comparison",
        "delta fact coverage",
        "delta citation precision",
        "delta citation recall",
        "delta schema ok",
        "delta abstain correct",
        "delta p50 latency s",
        "delta mean input tokens",
        "comparison run",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in deltas:
        metrics = row["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["strategy"],
                    comparison,
                    _fmt_delta(metrics["fact_coverage"]["delta"]),
                    _fmt_delta(metrics["citation_precision"]["delta"]),
                    _fmt_delta(metrics["citation_recall"]["delta"]),
                    _fmt_delta(metrics["schema_ok"]["delta"]),
                    _fmt_delta(metrics["abstain_correct"]["delta"]),
                    _fmt_delta(metrics["latency_p50_s"]["delta"], suffix="s"),
                    _fmt_delta(metrics["input_tokens_mean"]["delta"]),
                    row["comparison_run_dir"],
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def frontier_svg(rows: list[dict[str, Any]]) -> str:
    width = 920
    height = 560
    left = 96
    right = 760
    top = 72
    bottom = 460
    latencies = [row["metrics"]["latency_p50_s"]["value"] for row in rows if row["metrics"]["latency_p50_s"]["value"] is not None]
    qualities = [row["metrics"]["fact_coverage"]["value"] for row in rows if row["metrics"]["fact_coverage"]["value"] is not None]
    tokens = [row["metrics"]["input_tokens_mean"]["value"] for row in rows if row["metrics"]["input_tokens_mean"]["value"] is not None]
    min_latency = min(latencies, default=0.0)
    max_latency = max(latencies, default=1.0)
    max_tokens = max(tokens, default=1.0)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        '<text x="32" y="36" font-family="Arial, sans-serif" font-size="21" font-weight="700">Context Strategy Frontier: Quality vs Latency</text>',
        '<text x="32" y="58" font-family="Arial, sans-serif" font-size="13" fill="#555">Quality = fact coverage; latency = p50 end-to-end seconds; point size = mean prompt tokens.</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333" stroke-width="1"/>',
        f'<text x="{(left + right) // 2 - 80}" y="{height - 40}" font-family="Arial, sans-serif" font-size="13">p50 latency (s)</text>',
        f'<text x="18" y="{(top + bottom) // 2}" font-family="Arial, sans-serif" font-size="13" transform="rotate(-90 18 {(top + bottom) // 2})">fact coverage</text>',
    ]

    palette = {
        "full_context": "#2563eb",
        "rag_topk": "#0f766e",
        "summary_memory": "#b45309",
        "structured_memory": "#7c3aed",
        "prefix_cache_friendly": "#be123c",
    }
    for row in rows:
        metrics = row["metrics"]
        latency = metrics["latency_p50_s"]["value"]
        quality = metrics["fact_coverage"]["value"]
        input_tokens = metrics["input_tokens_mean"]["value"] or 0.0
        if latency is None or quality is None:
            continue
        x = _scale(latency, min_latency, max_latency, left, right)
        y = _scale(quality, 0.0, 1.0, bottom, top)
        r = 7 + 17 * (input_tokens / max_tokens if max_tokens else 0.0)
        color = palette.get(row["strategy"], "#444")
        lat_ci = metrics["latency_p50_s"]
        fact_ci = metrics["fact_coverage"]
        x_low = _scale(lat_ci["ci_low"], min_latency, max_latency, left, right) if lat_ci["ci_low"] is not None else x
        x_high = _scale(lat_ci["ci_high"], min_latency, max_latency, left, right) if lat_ci["ci_high"] is not None else x
        y_low = _scale(fact_ci["ci_low"], 0.0, 1.0, bottom, top) if fact_ci["ci_low"] is not None else y
        y_high = _scale(fact_ci["ci_high"], 0.0, 1.0, bottom, top) if fact_ci["ci_high"] is not None else y
        lines.extend(
            [
                f'<line x1="{x_low:.1f}" y1="{y:.1f}" x2="{x_high:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="1.5" opacity="0.7"/>',
                f'<line x1="{x:.1f}" y1="{y_low:.1f}" x2="{x:.1f}" y2="{y_high:.1f}" stroke="{color}" stroke-width="1.5" opacity="0.7"/>',
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}" stroke="{_model_stroke(row["model"])}" stroke-width="2" opacity="0.74"/>',
                f'<text x="{x + r + 6:.1f}" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#222">{_escape(_point_label(row))}</text>',
            ]
        )

    for idx, row in enumerate(rows):
        y = 96 + idx * 26
        color = palette.get(row["strategy"], "#444")
        lines.extend(
            [
                f'<circle cx="790" cy="{y}" r="7" fill="{color}" stroke="{_model_stroke(row["model"])}" stroke-width="2" opacity="0.74"/>',
                f'<text x="806" y="{y + 4}" font-family="Arial, sans-serif" font-size="12">{_escape(_point_label(row))}</text>',
            ]
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _meta_values(records: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get("meta", {}).get(key)
        if isinstance(value, bool):
            values.append(1.0 if value else 0.0)
        elif isinstance(value, int | float):
            values.append(float(value))
    return values


def _streaming_record_values(records: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        if record.get("meta", {}).get("timing_mode") != "streaming":
            continue
        value = record.get(key)
        if isinstance(value, int | float):
            values.append(float(value))
    return values


def _streaming_meta_values(records: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        if record.get("meta", {}).get("timing_mode") != "streaming":
            continue
        value = record.get("meta", {}).get(key)
        if isinstance(value, int | float):
            values.append(float(value))
    return values


def _percentile(values: list[float], fraction: float) -> float:
    sorted_values = sorted(values)
    if not sorted_values:
        return 0.0
    midpoint = (len(sorted_values) - 1) * fraction
    low = int(midpoint)
    high = min(low + 1, len(sorted_values) - 1)
    weight = midpoint - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def _metric_seed(seed: int, model: str, strategy: str, metric: str) -> int:
    return seed + sum(ord(ch) for ch in model + strategy + metric)


def _fmt_ci(metric: dict[str, float | int | None]) -> str:
    value = metric["value"]
    if value is None:
        return "n/a"
    return f'{value:.3f} [{metric["ci_low"]:.3f}, {metric["ci_high"]:.3f}]'


def _fmt_delta(value: float | int | None, *, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.3f}{suffix}"


def _point_label(row: dict[str, Any]) -> str:
    return f'{_model_short_label(row["model"])} {row["strategy"]}'


def _model_short_label(model: str) -> str:
    if ":" in model:
        return model.rsplit(":", maxsplit=1)[1]
    return model


def _model_stroke(model: str) -> str:
    return "#111827" if _model_short_label(model) == "7b" else "#ffffff"


def _scale(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    if in_max == in_min:
        return (out_min + out_max) / 2
    return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    raise SystemExit(main())
