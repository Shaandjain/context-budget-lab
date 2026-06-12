"""Shared trace-schema v0.1 helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any


SCHEMA_VERSION = "0.1"


@dataclass(frozen=True)
class TraceRecord:
    schema_version: str
    run_id: str
    source: str
    request_id: str
    ts_arrival_s: float
    strategy: str
    model: str
    endpoint: str
    input_tokens: int
    output_tokens: int
    queue_wait_s: float | None
    ttft_s: float
    tpot_s: float | None
    latency_s: float
    error: str | None
    cost_usd: float | None
    meta: dict[str, Any]


def now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_id(prefix: str) -> str:
    safe_prefix = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in prefix).strip("-")
    return f"{safe_prefix}-{now_slug()}"


def write_jsonl_record(path: Path, record: TraceRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def write_run_sidecar(
    run_dir: Path,
    *,
    run_id: str,
    source: str,
    repo: str,
    config: dict[str, Any],
    environment: dict[str, Any],
    n_requests: int,
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "source": source,
        "created_at": now_iso(),
        "repo": repo,
        "config": config,
        "environment": environment,
        "n_requests": n_requests,
        "git_commit": current_git_commit(run_dir),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def current_git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None
    return result.stdout.strip()


def validate_record(record: TraceRecord) -> None:
    if record.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unexpected schema_version {record.schema_version!r}")
    if record.source not in {"sim", "live"}:
        raise ValueError("source must be 'sim' or 'live'")
    if record.ttft_s > record.latency_s + 1e-9:
        raise ValueError("ttft_s must be <= latency_s")
    if record.input_tokens < 0 or record.output_tokens < 0:
        raise ValueError("token counts must be non-negative")
