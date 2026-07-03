from __future__ import annotations

import json
from pathlib import Path

from analysis.v2_sweep import build_report, haystack_size, main


def test_haystack_size_parses_registered_suffix() -> None:
    assert haystack_size("public_ai_policy_v2_h32") == 32
    assert haystack_size("synthetic_agent_memory_abstain_v2_h8") == 8
    assert haystack_size("public_ai_policy_v0") is None


def test_v2_sweep_reports_completeness_and_hypothesis_rows(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    datasets = ["public_ai_policy_v2_h2", "public_ai_policy_v2_h8", "public_ai_policy_v2_h32"]
    for strategy, h2, h8, h32 in [
        ("full_context", 0.7, 0.65, 0.55),
        ("rag_topk", 0.7, 0.75, 0.9),
        ("summary_memory", 0.62, 0.55, 0.35),
        ("structured_memory", 0.6, 0.52, 0.3),
    ]:
        _write_run(
            results_root / "qwen2-5-3b" / strategy / "run-1",
            model="qwen2.5-3b-instruct",
            strategy=strategy,
            datasets=datasets,
            records=[
                _record(strategy, "public_ai_policy_v2_h2", "task-a", h2, 1.0, 200),
                _record(strategy, "public_ai_policy_v2_h8", "task-a", h8, 2.0, 400),
                _record(strategy, "public_ai_policy_v2_h32", "task-a", h32, 4.0, 900),
            ],
        )

    report = build_report(results_root, resamples=20, seed=7)

    assert report["clean"] is False
    assert report["completeness"][0]["expected_records"] == 60
    h1_h32 = [
        row
        for row in report["h1"]["rows"]
        if row["model"] == "qwen2.5-3b-instruct" and row["haystack_size"] == 32
    ][0]
    assert h1_h32["paired_delta"]["value"] == 0.35
    assert any(row["strategy"] == "full_context" for row in report["frontier"])


def test_v2_sweep_main_writes_outputs(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_run(
        results_root / "qwen2-5-3b" / "full_context" / "run-1",
        model="qwen2.5-3b-instruct",
        strategy="full_context",
        datasets=["public_ai_policy_v2_h2"],
        records=[_record("full_context", "public_ai_policy_v2_h2", "task-a", 0.5, 1.0, 100)],
    )
    out_dir = tmp_path / "analysis"

    assert main([str(results_root), "--out-dir", str(out_dir), "--resamples", "10", "--seed", "7"]) == 0

    assert (out_dir / "summary.json").exists()
    assert (out_dir / "summary.md").exists()


def _write_run(
    run_dir: Path,
    *,
    model: str,
    strategy: str,
    datasets: list[str],
    records: list[dict[str, object]],
) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "source": "live",
                "config": {
                    "model": model,
                    "strategies": [strategy],
                    "datasets": datasets,
                    "seed": 7,
                    "repeats": 1,
                    "limit": None,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "traces.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def _record(
    strategy: str,
    dataset_id: str,
    task_id: str,
    fact_coverage: float,
    latency_s: float,
    input_tokens: int,
) -> dict[str, object]:
    return {
        "strategy": strategy,
        "request_id": f"{dataset_id}:{task_id}:r0:{strategy}",
        "latency_s": latency_s,
        "input_tokens": input_tokens,
        "output_tokens": 10,
        "error": None,
        "meta": {
            "dataset_id": dataset_id,
            "task_id": f"{dataset_id}/{task_id}",
            "fact_coverage": fact_coverage,
        },
    }
