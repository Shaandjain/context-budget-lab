from __future__ import annotations

import json
from pathlib import Path

from analysis.abstention_variant import build_report, markdown_report


def test_build_report_compares_memory_lane_baseline_and_variant(tmp_path: Path) -> None:
    baseline_root = tmp_path / "baseline"
    variant_root = tmp_path / "variant"
    _write_run(
        baseline_root / "qwen2-5-3b" / "prefix_cache_friendly" / "run-1",
        model="qwen2.5:3b",
        strategy="prefix_cache_friendly",
        records=[
            _record("prefix_cache_friendly", "synthetic_agent_memory_v0", abstain_correct=False, fact_coverage=0.5),
            _record("prefix_cache_friendly", "public_ai_policy_v0", abstain_correct=True, fact_coverage=1.0),
        ],
    )
    _write_run(
        variant_root / "qwen2-5-3b" / "prefix_cache_abstain" / "run-1",
        model="qwen2.5:3b",
        strategy="prefix_cache_abstain",
        records=[
            _record("prefix_cache_abstain", "synthetic_agent_memory_v0", abstain_correct=True, fact_coverage=0.25),
        ],
    )

    report = build_report(
        baseline_root=baseline_root,
        variant_root=variant_root,
        models=["qwen2.5:3b"],
        dataset_id="synthetic_agent_memory_v0",
    )

    assert report["rows"][0]["n"] == 1
    assert report["rows"][0]["avg_abstain_correct"] == 0.0
    assert report["rows"][1]["avg_abstain_correct"] == 1.0
    assert report["deltas"][0]["delta_avg_abstain_correct"] == 1.0
    assert report["deltas"][0]["delta_avg_fact_coverage"] == -0.25
    table = markdown_report(report)
    assert "abstain_variant" in table
    assert "+1.000" in table


def _write_run(
    run_dir: Path,
    *,
    model: str,
    strategy: str,
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
                    "seed": 7,
                    "repeats": len(records),
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
    *,
    abstain_correct: bool,
    fact_coverage: float,
) -> dict[str, object]:
    return {
        "strategy": strategy,
        "latency_s": 1.0,
        "input_tokens": 100,
        "output_tokens": 10,
        "error": None,
        "cost_usd": None,
        "meta": {
            "dataset_id": dataset_id,
            "fact_coverage": fact_coverage,
            "citation_precision": 1.0,
            "citation_recall": 1.0,
            "evidence_recall": 1.0,
            "schema_ok": True,
            "abstain_correct": abstain_correct,
        },
    }
