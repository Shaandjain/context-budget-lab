from pathlib import Path

import pytest

from context_budget_lab.traces import SCHEMA_VERSION, TraceRecord, validate_record, write_jsonl_record


def test_validate_trace_record_accepts_shared_schema() -> None:
    record = TraceRecord(
        schema_version=SCHEMA_VERSION,
        run_id="run-1",
        source="live",
        request_id="req-1",
        ts_arrival_s=0.0,
        strategy="full_context",
        model="qwen2.5:3b",
        endpoint="http://localhost:11434/v1",
        input_tokens=10,
        output_tokens=5,
        queue_wait_s=None,
        ttft_s=0.2,
        tpot_s=None,
        latency_s=0.2,
        error=None,
        cost_usd=None,
        meta={"dataset_id": "toy", "answer_score": 1.0},
    )

    validate_record(record)


def test_validate_trace_record_rejects_bad_timing() -> None:
    record = TraceRecord(
        schema_version=SCHEMA_VERSION,
        run_id="run-1",
        source="live",
        request_id="req-1",
        ts_arrival_s=0.0,
        strategy="full_context",
        model="qwen2.5:3b",
        endpoint="http://localhost:11434/v1",
        input_tokens=10,
        output_tokens=5,
        queue_wait_s=None,
        ttft_s=0.3,
        tpot_s=None,
        latency_s=0.2,
        error=None,
        cost_usd=None,
        meta={},
    )

    with pytest.raises(ValueError, match="ttft_s"):
        validate_record(record)


def test_write_jsonl_record(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    record = TraceRecord(
        schema_version=SCHEMA_VERSION,
        run_id="run-1",
        source="sim",
        request_id="req-1",
        ts_arrival_s=0.0,
        strategy="rag_topk",
        model="simulated",
        endpoint="sim",
        input_tokens=1,
        output_tokens=1,
        queue_wait_s=None,
        ttft_s=0.0,
        tpot_s=None,
        latency_s=0.0,
        error=None,
        cost_usd=None,
        meta={},
    )

    write_jsonl_record(path, record)

    assert path.read_text(encoding="utf-8").count("\n") == 1


def test_validate_trace_record_accepts_streaming_tpot() -> None:
    record = TraceRecord(
        schema_version=SCHEMA_VERSION,
        run_id="run-1",
        source="live",
        request_id="req-1",
        ts_arrival_s=0.0,
        strategy="full_context",
        model="qwen2.5:7b",
        endpoint="http://localhost:11434/v1",
        input_tokens=40,
        output_tokens=8,
        queue_wait_s=None,
        ttft_s=0.4,
        tpot_s=0.05,
        latency_s=0.75,
        error=None,
        cost_usd=None,
        meta={"timing_mode": "streaming", "decode_tokens_per_s": 20.0},
    )

    validate_record(record)
