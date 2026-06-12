import json
from pathlib import Path

from benchmarks.run_context_benchmark import main
from benchmarks.run_matrix import main as run_matrix
from context_budget_lab.metrics import summarize_records


def test_cli_mock_smoke_writes_shared_trace(tmp_path: Path) -> None:
    assert main(["--mock", "--limit", "1", "--out", str(tmp_path)]) == 0

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    lines = (run_dir / "traces.jsonl").read_text(encoding="utf-8").strip().splitlines()

    assert run["source"] == "sim"
    assert run["repo"] == "context-budget-lab"
    assert run["n_requests"] == 10
    assert len(lines) == 10
    first = json.loads(lines[0])
    assert first["schema_version"] == "0.1"
    assert first["ttft_s"] < first["latency_s"]
    assert first["tpot_s"] is not None
    assert "fact_coverage" in first["meta"]
    assert "citation_precision" in first["meta"]
    assert "citation_recall" in first["meta"]
    assert "schema_ok" in first["meta"]
    assert "abstain_correct" in first["meta"]
    assert first["meta"]["timing_mode"] == "mock"
    assert first["meta"]["decode_tokens_per_s"] is not None

    summary = summarize_records([json.loads(line) for line in lines])
    assert "avg_fact_coverage" in summary[0]
    assert "avg_citation_precision" in summary[0]
    assert "avg_citation_recall" in summary[0]
    assert "avg_schema_ok" in summary[0]


def test_cli_repeats_record_seed_in_run_and_trace(tmp_path: Path) -> None:
    assert (
        main(
            [
                "--mock",
                "--limit",
                "1",
                "--strategies",
                "full_context",
                "--repeats",
                "2",
                "--seed",
                "42",
                "--out",
                str(tmp_path),
            ]
        )
        == 0
    )

    run_dir = next(tmp_path.iterdir())
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    lines = (run_dir / "traces.jsonl").read_text(encoding="utf-8").strip().splitlines()

    assert run["config"]["repeats"] == 2
    assert run["config"]["seed"] == 42
    assert run["n_requests"] == 4
    assert len(lines) == 4
    first = json.loads(lines[0])
    assert first["meta"]["repeat_index"] == 0
    assert first["meta"]["seed"] == 42
    assert ":r0:" in first["request_id"]


def test_cli_records_prefix_abstain_variant_metadata(tmp_path: Path) -> None:
    assert (
        main(
            [
                "--mock",
                "--limit",
                "1",
                "--datasets",
                "synthetic_agent_memory_v0",
                "--strategies",
                "prefix_cache_abstain",
                "--out",
                str(tmp_path),
            ]
        )
        == 0
    )

    run_dir = next(tmp_path.iterdir())
    first = json.loads((run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["strategy"] == "prefix_cache_abstain"
    assert first["meta"]["prefix_cacheable_tokens"] > 0
    assert first["meta"]["strategy_metadata"]["variant"] == "explicit_insufficient_context_instruction"


def test_run_matrix_writes_condition_summary_and_plot(tmp_path: Path) -> None:
    assert (
        run_matrix(
            [
                "--mock",
                "--limit",
                "1",
                "--strategies",
                "full_context",
                "--repeats",
                "1",
                "--out",
                str(tmp_path),
            ]
        )
        == 0
    )

    run_dirs = list((tmp_path / "simulated" / "full_context").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "latency_quality.svg").exists()
