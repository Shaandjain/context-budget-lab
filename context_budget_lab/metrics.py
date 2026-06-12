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
        fact_coverage = _meta_numbers(ok_records, "fact_coverage")
        citation_precision = _meta_numbers(ok_records, "citation_precision")
        citation_recall = _meta_numbers(ok_records, "citation_recall")
        evidence = _meta_numbers(ok_records, "evidence_recall")
        schema_ok = _meta_numbers(ok_records, "schema_ok")
        abstain_correct = _meta_numbers(ok_records, "abstain_correct")
        useful = [
            record
            for record in ok_records
            if _is_useful(record)
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
                "avg_fact_coverage": round(mean(fact_coverage), 3) if fact_coverage else None,
                "avg_citation_precision": round(mean(citation_precision), 3) if citation_precision else None,
                "avg_citation_recall": round(mean(citation_recall), 3) if citation_recall else None,
                "avg_evidence_recall": round(mean(evidence), 3) if evidence else None,
                "avg_schema_ok": round(mean(schema_ok), 3) if schema_ok else None,
                "avg_abstain_correct": round(mean(abstain_correct), 3) if abstain_correct else None,
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
        "avg_fact_coverage",
        "avg_citation_precision",
        "avg_citation_recall",
        "avg_schema_ok",
        "avg_abstain_correct",
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


def _meta_numbers(records: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get("meta", {}).get(key)
        if isinstance(value, bool):
            values.append(1.0 if value else 0.0)
        elif isinstance(value, int | float):
            values.append(float(value))
    return values


def _meta_float(record: dict[str, Any], key: str) -> float | None:
    value = record.get("meta", {}).get(key, 0.0)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    return None


def _is_useful(record: dict[str, Any]) -> bool:
    fact_coverage = _meta_float(record, "fact_coverage")
    citation_recall = _meta_float(record, "citation_recall")
    schema_ok = record.get("meta", {}).get("schema_ok")
    abstain_correct = record.get("meta", {}).get("abstain_correct")
    if fact_coverage is None or fact_coverage < 0.7:
        return False
    if citation_recall is not None and citation_recall < 0.7:
        return False
    if schema_ok is False:
        return False
    if abstain_correct is False:
        return False
    return True


def _record_cost(record: dict[str, Any]) -> float:
    value = record.get("cost_usd")
    if isinstance(value, int | float):
        return float(value)
    return 0.0
