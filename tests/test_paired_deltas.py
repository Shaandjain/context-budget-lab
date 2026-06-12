from __future__ import annotations

import json
from pathlib import Path

from analysis.paired_deltas import build_report, main


def test_paired_reanalysis_finds_paired_only_separation(tmp_path: Path) -> None:
    _write_run(
        tmp_path / "qwen2-5-3b" / "full_context" / "run-1",
        model="qwen2.5:3b",
        strategy="full_context",
        values=[0.0, 0.2, 0.4, 0.6],
    )
    _write_run(
        tmp_path / "qwen2-5-3b" / "rag_topk" / "run-1",
        model="qwen2.5:3b",
        strategy="rag_topk",
        values=[0.2, 0.4, 0.6, 0.8],
    )
    _write_run(
        tmp_path / "qwen2-5-3b" / "summary_memory" / "run-1",
        model="qwen2.5:3b",
        strategy="summary_memory",
        values=[0.0, 0.2, 0.4, 0.6],
    )

    report = build_report(tmp_path, resamples=200, seed=7)

    row = _find_row(report, "qwen2.5:3b", "full_context", "qwen2.5:3b", "rag_topk")
    metric = row["metrics"]["fact_coverage"]
    assert metric["per_arm_ci_overlap"] is True
    assert metric["paired_separation"] is True
    assert metric["paired_only_separation"] is True
    assert metric["paired_delta"]["value"] == 0.2

    null_row = _find_row(report, "qwen2.5:3b", "full_context", "qwen2.5:3b", "summary_memory")
    null_metric = null_row["metrics"]["fact_coverage"]
    assert null_metric["paired_delta"]["value"] == 0.0
    assert null_metric["paired_separation"] is False


def test_paired_deltas_main_writes_summary_files(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_run(
        results_root / "qwen2-5-3b" / "full_context" / "run-1",
        model="qwen2.5:3b",
        strategy="full_context",
        values=[0.5, 0.5],
    )
    _write_run(
        results_root / "qwen2-5-3b" / "rag_topk" / "run-1",
        model="qwen2.5:3b",
        strategy="rag_topk",
        values=[0.5, 0.75],
    )
    out_dir = tmp_path / "paired"

    assert main([str(results_root), "--out-dir", str(out_dir), "--resamples", "20", "--seed", "7"]) == 0

    assert (out_dir / "summary.json").exists()
    assert (out_dir / "summary.md").exists()


def _write_run(path: Path, *, model: str, strategy: str, values: list[float]) -> None:
    path.mkdir(parents=True)
    (path / "run.json").write_text(
        json.dumps(
            {
                "source": "live",
                "config": {
                    "model": model,
                    "strategies": [strategy],
                    "seed": 7,
                    "repeats": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    records = [_record(strategy, index, value) for index, value in enumerate(values)]
    (path / "traces.jsonl").write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def _record(strategy: str, index: int, fact_coverage: float) -> dict[str, object]:
    return {
        "strategy": strategy,
        "request_id": f"task-{index}:r0:{strategy}",
        "latency_s": 1.0,
        "input_tokens": 100,
        "output_tokens": 10,
        "error": None,
        "meta": {
            "task_id": f"task-{index}",
            "fact_coverage": fact_coverage,
            "citation_precision": 1.0,
            "citation_recall": 1.0,
            "schema_ok": True,
            "abstain_correct": True,
        },
    }


def _find_row(
    report: dict[str, object],
    baseline_model: str,
    baseline_strategy: str,
    comparison_model: str,
    comparison_strategy: str,
) -> dict[str, object]:
    for row in report["rows"]:
        if (
            row["baseline_model"] == baseline_model
            and row["baseline_strategy"] == baseline_strategy
            and row["comparison_model"] == comparison_model
            and row["comparison_strategy"] == comparison_strategy
        ):
            return row
    raise AssertionError("row not found")
