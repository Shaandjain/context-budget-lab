"""Aggregate trace records into context-strategy comparison rows."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


def read_trace_records(run_dir: Path) -> list[dict[str, Any]]:
    trace_path = run_dir / "traces.jsonl"
    records: list[dict[str, Any]] = []
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_strategy: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_strategy.setdefault(str(record["strategy"]), []).append(record)

    rows: list[dict[str, Any]] = []
    for strategy, strategy_records in sorted(by_strategy.items()):
        ok_records = [record for record in strategy_records if record.get("error") is None]
        quality = [_meta_float(record, "answer_score") for record in ok_records]
        citations = [_meta_float(record, "citation_score") for record in ok_records]
        evidence = [_meta_float(record, "evidence_recall") for record in ok_records]
        useful = [
            record
            for record in ok_records
            if _meta_float(record, "answer_score") >= 0.7 and _meta_float(record, "citation_score") >= 0.7
        ]
        total_cost = sum(_record_cost(record) for record in ok_records)
        rows.append(
            {
                "strategy": strategy,
                "n": len(strategy_records),
                "errors": len(strategy_records) - len(ok_records),
                "avg_input_tokens": round(mean(record["input_tokens"] for record in ok_records), 2) if ok_records else None,
                "avg_output_tokens": round(mean(record["output_tokens"] for record in ok_records), 2) if ok_records else None,
                "avg_latency_s": round(mean(record["latency_s"] for record in ok_records), 4) if ok_records else None,
                "avg_answer_score": round(mean(quality), 3) if quality else None,
                "avg_citation_score": round(mean(citations), 3) if citations else None,
                "avg_evidence_recall": round(mean(evidence), 3) if evidence else None,
                "useful_answers": len(useful),
                "cost_per_useful_answer_usd": round(total_cost / len(useful), 6) if useful and total_cost else None,
            }
        )
    return rows


def write_summary(run_dir: Path) -> list[dict[str, Any]]:
    rows = summarize_records(read_trace_records(run_dir))
    (run_dir / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rows


def format_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        "strategy",
        "n",
        "errors",
        "avg_input_tokens",
        "avg_latency_s",
        "avg_answer_score",
        "avg_citation_score",
        "avg_evidence_recall",
        "useful_answers",
    ]
    widths = {column: max(len(column), *(len(str(row.get(column))) for row in rows)) for column in columns}
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    divider = "  ".join("-" * widths[column] for column in columns)
    lines = [header, divider]
    for row in rows:
        lines.append("  ".join(str(row.get(column)).ljust(widths[column]) for column in columns))
    return "\n".join(lines)


def _meta_float(record: dict[str, Any], key: str) -> float:
    value = record.get("meta", {}).get(key, 0.0)
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _record_cost(record: dict[str, Any]) -> float:
    value = record.get("cost_usd")
    if isinstance(value, int | float):
        return float(value)
    return 0.0
