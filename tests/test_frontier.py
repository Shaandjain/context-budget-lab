from __future__ import annotations

import json
from pathlib import Path

from analysis.frontier import main, markdown_table, summarize_matrix


def test_summarize_matrix_computes_bootstrap_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "qwen" / "full_context" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "source": "live",
                "config": {
                    "model": "qwen2.5:3b",
                    "strategies": ["full_context"],
                    "seed": 7,
                    "repeats": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    records = [
        _record("full_context", 0.5, 2.0, 100),
        _record("full_context", 1.0, 4.0, 120),
    ]
    (run_dir / "traces.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    rows = summarize_matrix(tmp_path, resamples=25, seed=7)

    assert len(rows) == 1
    row = rows[0]
    assert row["strategy"] == "full_context"
    assert row["n"] == 2
    assert row["metrics"]["fact_coverage"]["value"] == 0.75
    assert row["metrics"]["latency_p50_s"]["value"] == 3.0
    assert "fact coverage" in markdown_table(rows)


def test_frontier_main_writes_outputs(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    run_dir = results_root / "qwen" / "rag_topk" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "source": "live",
                "config": {
                    "model": "qwen2.5:3b",
                    "strategies": ["rag_topk"],
                    "seed": 7,
                    "repeats": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "traces.jsonl").write_text(json.dumps(_record("rag_topk", 1.0, 1.0, 80)) + "\n", encoding="utf-8")
    out_dir = tmp_path / "frontier"

    assert main([str(results_root), "--out-dir", str(out_dir), "--resamples", "10", "--seed", "7"]) == 0

    assert (out_dir / "summary.json").exists()
    assert (out_dir / "summary.md").exists()
    assert (out_dir / "frontier.svg").exists()


def _record(strategy: str, fact_coverage: float, latency_s: float, input_tokens: int) -> dict[str, object]:
    return {
        "strategy": strategy,
        "latency_s": latency_s,
        "input_tokens": input_tokens,
        "output_tokens": 10,
        "error": None,
        "meta": {
            "fact_coverage": fact_coverage,
            "citation_precision": 1.0,
            "citation_recall": 1.0,
            "schema_ok": True,
            "abstain_correct": True,
        },
    }
