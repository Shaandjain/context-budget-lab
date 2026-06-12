import json
from pathlib import Path

from benchmarks.run_context_benchmark import main


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
    assert "answer_score" in first["meta"]
