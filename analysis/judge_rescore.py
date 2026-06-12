"""Judge-calibrate fact coverage on committed context-budget traces.

The deterministic scorer remains the primary instrument. This script writes a
separate judge_scores.jsonl keyed by request_id so paraphrase calibration can be
audited without mutating trace fields.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import sys
from typing import Any, Protocol
from urllib import error, request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_budget_lab.datasets import available_datasets, load_dataset
from context_budget_lab.metrics import read_trace_records


DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"
PROMPT_VERSION = "fact-coverage-judge-v1"
ANSWER_TEXT_SOURCE = "meta.answer_excerpt"
DEFAULT_INPUT_PRICE_PER_MTOK = 1.0
DEFAULT_OUTPUT_PRICE_PER_MTOK = 5.0
DEFAULT_MAX_TOKENS = 600
MATRIX_ROOT_NAME = "matrix"
ABSTENTION_ROOT_NAME = "abstention_variant"


@dataclass(frozen=True)
class RequestCase:
    request_id: str
    run_id: str
    result_set: str
    run_dir: str
    model: str
    strategy: str
    dataset_id: str
    task_id: str
    repeat_index: int | None
    question: str
    expected_answer: str
    expected_facts: tuple[str, ...]
    literal_matched_facts: tuple[str, ...]
    literal_fact_coverage: float | None
    answer_text: str
    answer_text_source: str = ANSWER_TEXT_SOURCE


@dataclass(frozen=True)
class JudgeCallResult:
    verdicts: tuple[dict[str, Any], ...]
    raw_text: str
    input_tokens: int
    output_tokens: int
    stop_reason: str | None = None


class JudgeClient(Protocol):
    def judge(self, case: RequestCase, *, model: str, max_tokens: int) -> JudgeCallResult:
        ...


class AnthropicJudgeClient:
    def __init__(self, *, api_key: str, timeout_s: float = 60.0) -> None:
        self.api_key = api_key
        self.timeout_s = timeout_s

    def judge(self, case: RequestCase, *, model: str, max_tokens: int) -> JudgeCallResult:
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0,
            "system": system_prompt(),
            "messages": [{"role": "user", "content": user_prompt(case)}],
        }
        req = request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Anthropic API error {exc.code}: {message}") from exc

        text = "\n".join(
            block.get("text", "")
            for block in payload.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
        usage = payload.get("usage", {})
        return JudgeCallResult(
            verdicts=tuple(parse_judge_text(text, case.expected_facts)),
            raw_text=text,
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            stop_reason=payload.get("stop_reason"),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, default=Path("results/local-matrix"))
    parser.add_argument("--abstention-root", type=Path, default=Path("results/abstention-variant"))
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/judge_rescore"))
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--api-key-env", default="ANTHROPIC_API_KEY")
    parser.add_argument("--max-cost-usd", type=float, default=10.0)
    parser.add_argument(
        "--unrecovered-prior-spend-usd",
        type=float,
        default=0.0,
        help="Already-incurred spend not represented by cached score records.",
    )
    parser.add_argument("--input-price-per-mtok", type=float, default=DEFAULT_INPUT_PRICE_PER_MTOK)
    parser.add_argument("--output-price-per-mtok", type=float, default=DEFAULT_OUTPUT_PRICE_PER_MTOK)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true", help="Count judge cases and estimate cost without network calls.")
    parser.add_argument("--limit-cases", type=int, default=None, help="Only judge this many unique cases; useful for smoke tests.")
    args = parser.parse_args(argv)

    cases = load_request_cases(matrix_root=args.matrix_root, abstention_root=args.abstention_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        dry_run = dry_run_report(
            cases,
            model=args.model,
            max_tokens=args.max_tokens,
            input_price_per_mtok=args.input_price_per_mtok,
            output_price_per_mtok=args.output_price_per_mtok,
            limit_cases=args.limit_cases,
        )
        (args.out_dir / "dry_run.json").write_text(json.dumps(dry_run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (args.out_dir / "dry_run.md").write_text(markdown_dry_run(dry_run), encoding="utf-8")
        print(args.out_dir)
        return 0

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is required for live judge calls")

    client = AnthropicJudgeClient(api_key=api_key, timeout_s=args.timeout_s)
    outputs = run_judge_rescore(
        cases,
        client=client,
        out_dir=args.out_dir,
        model=args.model,
        max_cost_usd=args.max_cost_usd,
        unrecovered_prior_spend_usd=args.unrecovered_prior_spend_usd,
        input_price_per_mtok=args.input_price_per_mtok,
        output_price_per_mtok=args.output_price_per_mtok,
        max_tokens=args.max_tokens,
        resamples=args.resamples,
        seed=args.seed,
        limit_cases=args.limit_cases,
    )
    print(f"{args.out_dir} (${outputs['total_cost_usd']:.6f})")
    return 0


def run_judge_rescore(
    cases: list[RequestCase],
    *,
    client: JudgeClient,
    out_dir: Path,
    model: str,
    max_cost_usd: float,
    unrecovered_prior_spend_usd: float = 0.0,
    input_price_per_mtok: float,
    output_price_per_mtok: float,
    max_tokens: int,
    resamples: int,
    seed: int,
    limit_cases: int | None = None,
) -> dict[str, Any]:
    score_path = out_dir / "judge_scores.jsonl"
    existing = read_score_records(score_path)
    existing_by_request = {score_id_from_record(record): record for record in existing}
    existing_by_key: dict[str, dict[str, Any]] = {}
    for record in existing:
        existing_by_key.setdefault(str(record["dedupe_key"]), record)

    groups = group_cases_by_dedupe_key(cases)
    unique_keys = sorted(groups)
    if limit_cases is not None:
        unique_keys = unique_keys[:limit_cases]

    cached_cost_usd = sum(float(record.get("cost_usd", 0.0) or 0.0) for record in existing_by_key.values())
    total_cost_usd = unrecovered_prior_spend_usd + cached_cost_usd
    newly_judged = 0
    for key in unique_keys:
        group = groups[key]
        cached = existing_by_key.get(key)
        if cached is None:
            estimate = estimated_case_cost(
                group[0],
                input_price_per_mtok=input_price_per_mtok,
                output_price_per_mtok=output_price_per_mtok,
                max_tokens=max_tokens,
            )
            if total_cost_usd + estimate > max_cost_usd:
                raise RuntimeError(
                    f"stopping before {key}: estimated spend ${total_cost_usd + estimate:.4f} "
                    f"would exceed cap ${max_cost_usd:.4f}"
                )
            call = client.judge(group[0], model=model, max_tokens=max_tokens)
            cost_usd = token_cost(
                input_tokens=call.input_tokens,
                output_tokens=call.output_tokens,
                input_price_per_mtok=input_price_per_mtok,
                output_price_per_mtok=output_price_per_mtok,
            )
            total_cost_usd += cost_usd
            cached = call_to_cache_record(
                key,
                group[0],
                call,
                model=model,
                cost_usd=cost_usd,
                input_price_per_mtok=input_price_per_mtok,
                output_price_per_mtok=output_price_per_mtok,
            )
            existing_by_key[key] = cached
            newly_judged += 1

        for case in group:
            case_score_id = score_id(case)
            if case_score_id not in existing_by_request:
                existing_by_request[case_score_id] = score_record_from_cache(cached, case, deduped_request_count=len(group))

    score_records = sorted(existing_by_request.values(), key=score_id_from_record)
    write_score_records(score_path, score_records)

    report = build_report(score_records, resamples=resamples, seed=seed)
    score_file_cost_usd = unique_score_file_cost(score_records)
    actual_total_cost_usd = unrecovered_prior_spend_usd + score_file_cost_usd
    report.update(
        {
            "judge_model": model,
            "prompt_version": PROMPT_VERSION,
            "answer_text_source": ANSWER_TEXT_SOURCE,
            "score_file_cost_usd": round(score_file_cost_usd, 8),
            "unrecovered_prior_spend_usd": round(unrecovered_prior_spend_usd, 8),
            "total_cost_usd": round(actual_total_cost_usd, 8),
            "newly_judged_unique_cases": newly_judged,
            "pricing": {
                "input_usd_per_mtok": input_price_per_mtok,
                "output_usd_per_mtok": output_price_per_mtok,
            },
        }
    )
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(markdown_report(report), encoding="utf-8")
    (out_dir / "judge_disagreements.md").write_text(markdown_disagreements(sample_disagreements(score_records, limit=20)), encoding="utf-8")
    (out_dir / "proposed_fact_aliases.md").write_text(markdown_aliases(proposed_aliases(score_records)), encoding="utf-8")
    return report


def unique_score_file_cost(score_records: list[dict[str, Any]]) -> float:
    by_key: dict[str, dict[str, Any]] = {}
    for record in score_records:
        by_key.setdefault(str(record["dedupe_key"]), record)
    return sum(float(record.get("cost_usd", 0.0) or 0.0) for record in by_key.values())


def load_request_cases(*, matrix_root: Path, abstention_root: Path) -> list[RequestCase]:
    task_lookup = load_task_lookup()
    cases: list[RequestCase] = []
    for result_set, root in [(MATRIX_ROOT_NAME, matrix_root), (ABSTENTION_ROOT_NAME, abstention_root)]:
        if not root.exists():
            continue
        for run_dir in find_run_dirs(root):
            run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            config = run.get("config", {})
            for record in read_trace_records(run_dir):
                if record.get("error") is not None:
                    continue
                meta = record.get("meta", {})
                dataset_id = str(meta.get("dataset_id", ""))
                task_id = str(meta.get("task_id", ""))
                task = task_lookup.get((dataset_id, task_id))
                if task is None:
                    raise RuntimeError(f"no dataset task found for {dataset_id}:{task_id} in {run_dir}")
                answer_text = str(meta.get("answer_excerpt") or "")
                cases.append(
                    RequestCase(
                        request_id=str(record["request_id"]),
                        run_id=str(record.get("run_id", run.get("run_id", ""))),
                        result_set=result_set,
                        run_dir=str(run_dir),
                        model=str(record.get("model") or config.get("model", "")),
                        strategy=str(record.get("strategy") or (config.get("strategies") or [""])[0]),
                        dataset_id=dataset_id,
                        task_id=task_id,
                        repeat_index=_optional_int(meta.get("repeat_index")),
                        question=str(meta.get("question") or task.query),
                        expected_answer=task.expected_answer,
                        expected_facts=tuple(task.expected_facts),
                        literal_matched_facts=tuple(_string_list(meta.get("matched_expected_facts"))),
                        literal_fact_coverage=_optional_float(meta.get("fact_coverage")),
                        answer_text=answer_text,
                    )
                )
    return sorted(cases, key=lambda case: case.request_id)


def load_task_lookup() -> dict[tuple[str, str], Any]:
    lookup: dict[tuple[str, str], Any] = {}
    for dataset_id in available_datasets():
        for task in load_dataset(dataset_id):
            lookup[(task.dataset_id, task.task_id)] = task
    return lookup


def find_run_dirs(results_root: Path) -> list[Path]:
    return sorted(path.parent for path in results_root.rglob("run.json") if (path.parent / "traces.jsonl").exists())


def group_cases_by_dedupe_key(cases: list[RequestCase]) -> dict[str, list[RequestCase]]:
    groups: dict[str, list[RequestCase]] = defaultdict(list)
    for case in cases:
        groups[dedupe_key(case)].append(case)
    return dict(groups)


def dedupe_key(case: RequestCase) -> str:
    payload = {
        "dataset_id": case.dataset_id,
        "task_id": case.task_id,
        "answer_text": normalize_answer_for_dedupe(case.answer_text),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return digest[:24]


def normalize_answer_for_dedupe(answer_text: str) -> str:
    return re.sub(r"\s+", " ", answer_text).strip()


def system_prompt() -> str:
    return (
        "You judge fact coverage for a benchmark. Credit a fact when the model answer states "
        "the same meaning, including clear paraphrases. Do not require exact wording. Do not judge "
        "citations, formatting, style, or extra facts. Return only valid JSON."
    )


def user_prompt(case: RequestCase) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "question": case.question,
        "gold_answer": case.expected_answer,
        "expected_facts": list(case.expected_facts),
        "model_answer": case.answer_text,
        "instructions": {
            "verdict_schema": {
                "facts": [
                    {
                        "fact": "copy the expected fact exactly",
                        "credited": "boolean",
                        "reason": "brief reason grounded in the model answer",
                    }
                ]
            },
            "rules": [
                "Evaluate each expected fact independently.",
                "Set credited=true for exact matches or unambiguous paraphrases.",
                "Set credited=false when the answer omits, contradicts, or only vaguely hints at the fact.",
                "Use the expected_facts order and include exactly one verdict per expected fact.",
            ],
        },
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def parse_judge_text(text: str, expected_facts: tuple[str, ...] | list[str]) -> list[dict[str, Any]]:
    payload = json.loads(extract_json_object(text))
    facts = payload.get("facts")
    if not isinstance(facts, list):
        raise ValueError("judge response must contain a facts list")
    if len(facts) != len(expected_facts):
        raise ValueError(f"judge returned {len(facts)} facts, expected {len(expected_facts)}")

    verdicts: list[dict[str, Any]] = []
    for expected, item in zip(expected_facts, facts, strict=True):
        if not isinstance(item, dict):
            raise ValueError("each judge fact verdict must be an object")
        credited = item.get("credited")
        if not isinstance(credited, bool):
            raise ValueError(f"credited must be boolean for fact {expected!r}")
        verdicts.append(
            {
                "fact": str(expected),
                "credited": credited,
                "reason": str(item.get("reason", "")).strip(),
            }
        )
    return verdicts


def extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("judge response did not contain a JSON object")
    return stripped[start : end + 1]


def call_to_cache_record(
    key: str,
    case: RequestCase,
    call: JudgeCallResult,
    *,
    model: str,
    cost_usd: float,
    input_price_per_mtok: float,
    output_price_per_mtok: float,
) -> dict[str, Any]:
    return {
        "schema_version": "judge-rescore-v1",
        "dedupe_key": key,
        "judge_model": model,
        "prompt_version": PROMPT_VERSION,
        "answer_text_hash": answer_text_hash(case.answer_text),
        "answer_text_source": case.answer_text_source,
        "answer_text": case.answer_text,
        "expected_facts": list(case.expected_facts),
        "verdicts": list(call.verdicts),
        "judge_matched_facts": [item["fact"] for item in call.verdicts if item["credited"]],
        "judge_fact_coverage": _ratio(sum(1 for item in call.verdicts if item["credited"]), len(call.verdicts)),
        "raw_judge_text": call.raw_text,
        "usage": {
            "input_tokens": call.input_tokens,
            "output_tokens": call.output_tokens,
        },
        "cost_usd": round(cost_usd, 8),
        "pricing": {
            "input_usd_per_mtok": input_price_per_mtok,
            "output_usd_per_mtok": output_price_per_mtok,
        },
        "stop_reason": call.stop_reason,
    }


def score_record_from_cache(cache: dict[str, Any], case: RequestCase, *, deduped_request_count: int) -> dict[str, Any]:
    literal_flags = literal_credit_flags(case.expected_facts, case.literal_matched_facts)
    judge_flags = [bool(item["credited"]) for item in cache["verdicts"]]
    return {
        "schema_version": cache["schema_version"],
        "score_id": score_id(case),
        "request_id": case.request_id,
        "run_id": case.run_id,
        "result_set": case.result_set,
        "source_run_dir": case.run_dir,
        "model": case.model,
        "strategy": case.strategy,
        "dataset_id": case.dataset_id,
        "task_id": case.task_id,
        "repeat_index": case.repeat_index,
        "question": case.question,
        "expected_answer": case.expected_answer,
        "expected_facts": list(case.expected_facts),
        "answer_text_source": cache["answer_text_source"],
        "answer_text_hash": cache["answer_text_hash"],
        "answer_text": cache["answer_text"],
        "dedupe_key": cache["dedupe_key"],
        "deduped_request_count": deduped_request_count,
        "judge_model": cache["judge_model"],
        "prompt_version": cache["prompt_version"],
        "literal_matched_facts": list(case.literal_matched_facts),
        "literal_fact_coverage": case.literal_fact_coverage,
        "literal_credit_flags": literal_flags,
        "judge_matched_facts": cache["judge_matched_facts"],
        "judge_fact_coverage": cache["judge_fact_coverage"],
        "judge_credit_flags": judge_flags,
        "verdicts": cache["verdicts"],
        "usage": cache["usage"],
        "cost_usd": cache["cost_usd"],
        "pricing": cache["pricing"],
        "stop_reason": cache.get("stop_reason"),
    }


def score_id(case: RequestCase) -> str:
    payload = {
        "result_set": case.result_set,
        "run_id": case.run_id,
        "model": case.model,
        "strategy": case.strategy,
        "request_id": case.request_id,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:24]


def score_id_from_record(record: dict[str, Any]) -> str:
    value = record.get("score_id")
    if isinstance(value, str) and value:
        return value
    payload = {
        "result_set": record.get("result_set"),
        "run_id": record.get("run_id"),
        "model": record.get("model"),
        "strategy": record.get("strategy"),
        "request_id": record.get("request_id"),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:24]


def read_score_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_score_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8")


def build_report(score_records: list[dict[str, Any]], *, resamples: int, seed: int) -> dict[str, Any]:
    return {
        "n_requests": len(score_records),
        "n_unique_judge_cases": len({record["dedupe_key"] for record in score_records}),
        "agreement_rows": agreement_rows(score_records),
        "h2": h2_report(score_records, resamples=resamples, seed=seed),
        "disagreement_count": sum(disagreement_count(record) for record in score_records),
        "judge_scores_path": "analysis/judge_rescore/judge_scores.jsonl",
    }


def agreement_rows(score_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in score_records:
        grouped[(str(record["model"]), str(record["strategy"]))].append(record)

    rows: list[dict[str, Any]] = []
    for (model, strategy), records in sorted(grouped.items()):
        literal_values = [_float(record.get("literal_fact_coverage")) for record in records]
        judge_values = [_float(record.get("judge_fact_coverage")) for record in records]
        literal_values = [value for value in literal_values if value is not None]
        judge_values = [value for value in judge_values if value is not None]
        fact_total = 0
        fact_agree = 0
        judge_credited_literal_missed = 0
        literal_credited_judge_missed = 0
        for record in records:
            for literal, judge in zip(record["literal_credit_flags"], record["judge_credit_flags"], strict=True):
                fact_total += 1
                fact_agree += int(literal == judge)
                judge_credited_literal_missed += int(judge and not literal)
                literal_credited_judge_missed += int(literal and not judge)
        literal_mean = statistics.mean(literal_values) if literal_values else None
        judge_mean = statistics.mean(judge_values) if judge_values else None
        rows.append(
            {
                "model": model,
                "strategy": strategy,
                "n_requests": len(records),
                "n_tasks": len({record["task_id"] for record in records}),
                "n_unique_judge_cases": len({record["dedupe_key"] for record in records}),
                "literal_fact_coverage_mean": _round_or_none(literal_mean),
                "judge_fact_coverage_mean": _round_or_none(judge_mean),
                "delta_judge_minus_literal": _round_or_none(None if literal_mean is None or judge_mean is None else judge_mean - literal_mean),
                "per_fact_agreement": _round_or_none(_ratio(fact_agree, fact_total)),
                "judge_credited_literal_missed": judge_credited_literal_missed,
                "literal_credited_judge_missed": literal_credited_judge_missed,
            }
        )
    return rows


def h2_report(score_records: list[dict[str, Any]], *, resamples: int, seed: int) -> dict[str, Any]:
    matrix_records = [record for record in score_records if record.get("result_set") == MATRIX_ROOT_NAME]
    task_values = aggregate_metric_by_task(matrix_records, metric="judge_fact_coverage")
    rows: list[dict[str, Any]] = []
    for strategy in ["full_context", "rag_topk"]:
        baseline = task_values.get(("qwen2.5:3b", strategy), {})
        comparison = task_values.get(("qwen2.5:7b", strategy), {})
        row = paired_model_delta(
            baseline,
            comparison,
            baseline_model="qwen2.5:3b",
            comparison_model="qwen2.5:7b",
            strategy=strategy,
            resamples=resamples,
            seed=seed + sum(ord(ch) for ch in strategy),
        )
        rows.append(row)

    statuses = [row["status"] for row in rows]
    if statuses and all(status == "supports_h2" for status in statuses):
        verdict = "supported"
    elif any(status == "refutes_h2" for status in statuses):
        verdict = "refuted"
    else:
        verdict = "inconclusive"
    return {
        "hypothesis": "H2: judge-scored fact coverage shows qwen2.5:7b >= qwen2.5:3b on full_context and rag_topk.",
        "verdict": verdict,
        "delta": "qwen2.5:7b_minus_qwen2.5:3b",
        "rows": rows,
    }


def aggregate_metric_by_task(
    records: list[dict[str, Any]],
    *,
    metric: str,
) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for record in records:
        value = _float(record.get(metric))
        if value is None:
            continue
        grouped[(str(record["model"]), str(record["strategy"]), str(record["task_id"]))].append(value)
    by_arm: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for (model, strategy, task_id), values in grouped.items():
        by_arm[(model, strategy)][task_id] = statistics.mean(values)
    return dict(by_arm)


def paired_model_delta(
    baseline: dict[str, float],
    comparison: dict[str, float],
    *,
    baseline_model: str,
    comparison_model: str,
    strategy: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    common_task_ids = sorted(set(baseline).intersection(comparison))
    deltas = [comparison[task_id] - baseline[task_id] for task_id in common_task_ids]
    summary = summarize_values(deltas, resamples=resamples, seed=seed)
    if summary["ci_low"] is not None and float(summary["ci_low"]) >= 0:
        status = "supports_h2"
    elif summary["ci_high"] is not None and float(summary["ci_high"]) < 0:
        status = "refutes_h2"
    else:
        status = "inconclusive"
    return {
        "strategy": strategy,
        "baseline_model": baseline_model,
        "comparison_model": comparison_model,
        "n_tasks": len(common_task_ids),
        "paired_delta": summary,
        "status": status,
    }


def summarize_values(values: list[float], *, resamples: int, seed: int) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "value": None, "ci_low": None, "ci_high": None}
    value = statistics.mean(values)
    if len(values) == 1:
        rounded = round(value, 6)
        return {"n": 1, "value": rounded, "ci_low": rounded, "ci_high": rounded}
    rng = random.Random(seed)
    boot_values: list[float] = []
    for _ in range(resamples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        boot_values.append(statistics.mean(sample))
    boot_values.sort()
    low = boot_values[int(0.025 * (len(boot_values) - 1))]
    high = boot_values[int(0.975 * (len(boot_values) - 1))]
    return {"n": len(values), "value": round(value, 6), "ci_low": round(low, 6), "ci_high": round(high, 6)}


def sample_disagreements(score_records: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for record in score_records:
        for index, (literal, judge) in enumerate(zip(record["literal_credit_flags"], record["judge_credit_flags"], strict=True)):
            if literal == judge:
                continue
            key = (
                record["task_id"],
                record["model"],
                record["strategy"],
                record["expected_facts"][index],
                literal,
                judge,
                record["answer_text_hash"],
            )
            if key in seen:
                continue
            seen.add(key)
            verdict = record["verdicts"][index]
            rows.append(
                {
                    "request_id": record["request_id"],
                    "task_id": record["task_id"],
                    "model": record["model"],
                    "strategy": record["strategy"],
                    "fact": record["expected_facts"][index],
                    "literal_credited": literal,
                    "judge_credited": judge,
                    "reason": verdict.get("reason", ""),
                    "answer_excerpt": record["answer_text"][:240],
                }
            )
    rows.sort(key=lambda row: (not row["judge_credited"], row["task_id"], row["model"], row["strategy"], row["fact"]))
    return rows[:limit]


def proposed_aliases(score_records: list[dict[str, Any]], *, limit: int = 50) -> list[dict[str, Any]]:
    aliases: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in score_records:
        for index, (literal, judge) in enumerate(zip(record["literal_credit_flags"], record["judge_credit_flags"], strict=True)):
            if literal or not judge:
                continue
            fact = str(record["expected_facts"][index])
            key = (str(record["task_id"]), fact, candidate_alias(record["answer_text"], fact))
            aliases.setdefault(
                key,
                {
                    "task_id": record["task_id"],
                    "expected_fact": fact,
                    "candidate_alias": key[2],
                    "judge_reason": record["verdicts"][index].get("reason", ""),
                    "example_request_id": record["request_id"],
                },
            )
    return sorted(aliases.values(), key=lambda row: (row["task_id"], row["expected_fact"]))[:limit]


def candidate_alias(answer_text: str, fact: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", answer_text.strip())
    fact_terms = [term for term in re.split(r"[^a-z0-9_]+", fact.lower()) if len(term) >= 4]
    for sentence in sentences:
        normalized = sentence.lower()
        if any(term in normalized for term in fact_terms):
            return sentence[:180]
    return answer_text[:180]


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Judge Fact-Coverage Calibration",
        "",
        f"Judge model: `{report['judge_model']}`. Prompt version: `{report['prompt_version']}`.",
        f"Answer text source: `{report['answer_text_source']}`. These traces store saved answer excerpts, not full unrecoverable answers.",
        f"Requests scored: {report['n_requests']}; unique judge calls: {report['n_unique_judge_cases']}; total API spend: ${report['total_cost_usd']:.6f}.",
        "",
        "## Agreement",
        "",
        "| model | strategy | requests | unique cases | literal fact coverage | judge fact coverage | judge - literal | per-fact agreement | judge credited/literal missed | literal credited/judge missed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    if report.get("unrecovered_prior_spend_usd", 0.0):
        lines.insert(
            5,
            f"Spend includes ${report['unrecovered_prior_spend_usd']:.6f} from an unrecovered first pass; score-file calls account for ${report['score_file_cost_usd']:.6f}.",
        )
    for row in report["agreement_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    row["strategy"],
                    str(row["n_requests"]),
                    str(row["n_unique_judge_cases"]),
                    _fmt(row["literal_fact_coverage_mean"]),
                    _fmt(row["judge_fact_coverage_mean"]),
                    _fmt_delta(row["delta_judge_minus_literal"]),
                    _fmt(row["per_fact_agreement"]),
                    str(row["judge_credited_literal_missed"]),
                    str(row["literal_credited_judge_missed"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## H2 Verdict",
            "",
            f"Verdict: **{report['h2']['verdict']}**.",
            "",
            "| strategy | n tasks | 7b - 3b judge fact coverage | status |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in report["h2"]["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["strategy"],
                    str(row["n_tasks"]),
                    _fmt_ci(row["paired_delta"]),
                    row["status"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Disagreements are sampled in `judge_disagreements.md`; proposed scorer aliases are review-only in `proposed_fact_aliases.md`.",
        ]
    )
    return "\n".join(lines) + "\n"


def markdown_disagreements(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Judge vs Literal Disagreements",
        "",
        "Sampled deterministically, with judge-credited/literal-missed cases first.",
        "",
        "| task | model | strategy | fact | literal | judge | reason | answer excerpt |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(row["task_id"]),
                    row["model"],
                    row["strategy"],
                    _md(row["fact"]),
                    str(row["literal_credited"]),
                    str(row["judge_credited"]),
                    _md(row["reason"]),
                    _md(row["answer_excerpt"]),
                ]
            )
            + " |"
        )
    if not rows:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    return "\n".join(lines) + "\n"


def markdown_aliases(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Proposed Fact Aliases",
        "",
        "Review-only list. These are cases where the judge credited a fact that the literal scorer missed; datasets are not edited by this script.",
        "",
        "| task | expected fact | candidate alias/evidence | judge reason | example request |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(row["task_id"]),
                    _md(row["expected_fact"]),
                    _md(row["candidate_alias"]),
                    _md(row["judge_reason"]),
                    _md(row["example_request_id"]),
                ]
            )
            + " |"
        )
    if not rows:
        lines.append("| n/a | n/a | n/a | n/a | n/a |")
    return "\n".join(lines) + "\n"


def dry_run_report(
    cases: list[RequestCase],
    *,
    model: str,
    max_tokens: int,
    input_price_per_mtok: float,
    output_price_per_mtok: float,
    limit_cases: int | None = None,
) -> dict[str, Any]:
    groups = group_cases_by_dedupe_key(cases)
    keys = sorted(groups)
    if limit_cases is not None:
        keys = keys[:limit_cases]
    estimates = [
        estimated_case_cost(
            groups[key][0],
            input_price_per_mtok=input_price_per_mtok,
            output_price_per_mtok=output_price_per_mtok,
            max_tokens=max_tokens,
        )
        for key in keys
    ]
    return {
        "judge_model": model,
        "prompt_version": PROMPT_VERSION,
        "answer_text_source": ANSWER_TEXT_SOURCE,
        "n_requests": len(cases),
        "n_unique_judge_cases_total": len(groups),
        "n_unique_judge_cases_planned": len(keys),
        "max_tokens": max_tokens,
        "estimated_worst_case_cost_usd": round(sum(estimates), 6),
        "pricing": {
            "input_usd_per_mtok": input_price_per_mtok,
            "output_usd_per_mtok": output_price_per_mtok,
        },
    }


def markdown_dry_run(report: dict[str, Any]) -> str:
    return (
        "# Judge Rescore Dry Run\n\n"
        f"Judge model: `{report['judge_model']}`\n\n"
        f"Requests: {report['n_requests']}\n\n"
        f"Unique judge cases: {report['n_unique_judge_cases_total']}\n\n"
        f"Planned unique judge cases: {report['n_unique_judge_cases_planned']}\n\n"
        f"Estimated worst-case cost: ${report['estimated_worst_case_cost_usd']:.6f}\n"
    )


def estimated_case_cost(
    case: RequestCase,
    *,
    input_price_per_mtok: float,
    output_price_per_mtok: float,
    max_tokens: int,
) -> float:
    prompt_tokens = max(1, round((len(system_prompt()) + len(user_prompt(case))) / 4))
    return token_cost(
        input_tokens=prompt_tokens,
        output_tokens=max_tokens,
        input_price_per_mtok=input_price_per_mtok,
        output_price_per_mtok=output_price_per_mtok,
    )


def token_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    input_price_per_mtok: float,
    output_price_per_mtok: float,
) -> float:
    return (input_tokens / 1_000_000 * input_price_per_mtok) + (output_tokens / 1_000_000 * output_price_per_mtok)


def literal_credit_flags(expected_facts: tuple[str, ...], matched_facts: tuple[str, ...]) -> list[bool]:
    matched = {_normalize_fact(fact) for fact in matched_facts}
    return [_normalize_fact(fact) in matched for fact in expected_facts]


def disagreement_count(record: dict[str, Any]) -> int:
    return sum(
        1
        for literal, judge in zip(record["literal_credit_flags"], record["judge_credit_flags"], strict=True)
        if literal != judge
    )


def answer_text_hash(answer_text: str) -> str:
    return hashlib.sha256(answer_text.encode("utf-8")).hexdigest()[:16]


def _normalize_fact(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list | tuple):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except ValueError:
        return None


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None


def _float(value: Any) -> float | None:
    return _optional_float(value)


def _round_or_none(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _fmt_delta(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value > 0 else ""
    return f"{sign}{float(value):.3f}"


def _fmt_ci(summary: dict[str, float | int | None]) -> str:
    value = summary["value"]
    if value is None:
        return "n/a"
    return f'{float(value):+.3f} [{float(summary["ci_low"]):+.3f}, {float(summary["ci_high"]):+.3f}]'


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


if __name__ == "__main__":
    raise SystemExit(main())
