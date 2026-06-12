"""Run a local-first context-strategy benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_budget_lab.client import ChatClient, MockClient, OpenAICompatClient, estimate_tokens
from context_budget_lab.datasets import load_dataset, task_to_strategy_input
from context_budget_lab.scoring import score_answer
from context_budget_lab.strategies import build_strategy_prompt, list_strategy_names
from context_budget_lab.traces import SCHEMA_VERSION, TraceRecord, make_run_id, validate_record, write_jsonl_record, write_run_sidecar


DEFAULT_DATASETS = ["public_ai_policy_v0", "synthetic_agent_memory_v0"]
DEFAULT_STRATEGIES = ["full_context", "rag_topk", "summary_memory", "structured_memory", "prefix_cache_friendly"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES))
    parser.add_argument("--limit", type=int, default=None, help="Maximum tasks per dataset.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--mock", action="store_true", help="Use deterministic in-process completions.")
    args = parser.parse_args(argv)

    dataset_ids = _split_csv(args.datasets)
    strategy_names = _split_csv(args.strategies)
    unknown_strategies = sorted(set(strategy_names) - set(list_strategy_names()))
    if unknown_strategies:
        raise SystemExit(f"unknown strategies: {', '.join(unknown_strategies)}")

    source = "sim" if args.mock else "live"
    endpoint = "sim" if args.mock else args.base_url
    client: ChatClient = MockClient(args.model) if args.mock else OpenAICompatClient(args.base_url, args.model)
    run_id = make_run_id("context-budget")
    run_dir = args.out / run_id
    traces_path = run_dir / "traces.jsonl"
    start = time.perf_counter()
    n_requests = 0

    for dataset_id in dataset_ids:
        tasks = [task_to_strategy_input(task) for task in load_dataset(dataset_id)]
        if args.limit is not None:
            tasks = tasks[: args.limit]
        for task in tasks:
            for strategy_name in strategy_names:
                request_id = f"{dataset_id}:{task['task_id']}:{strategy_name}"
                ts_arrival_s = time.perf_counter() - start
                strategy_result = build_strategy_prompt(task, strategy_name)
                messages = _messages_from_prompt(strategy_result["prompt"])
                selected_source_ids = strategy_result.get("source_ids", [])
                strategy_meta = strategy_result.get("metadata", {})
                error: str | None = None
                answer = ""
                try:
                    completion = client.complete(
                        messages,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                    answer = completion.text
                except Exception as exc:  # traces record failed requests too
                    completion = _failed_completion(messages, time.perf_counter() - start - ts_arrival_s)
                    error = str(exc)

                score = score_answer(task, strategy_result, answer)
                scores = score["scores"]
                meta: dict[str, Any] = {
                    "dataset_id": dataset_id,
                    "task_id": task["task_id"],
                    "question": task["question"],
                    "retrieved_source_ids": selected_source_ids,
                    "gold_source_ids": task.get("gold_source_ids", []),
                    "fact_coverage": scores["fact_coverage"],
                    "citation_precision": scores["citation_precision"],
                    "citation_recall": scores["citation_recall"],
                    "evidence_recall": scores["evidence_recall"],
                    "schema_ok": scores["schema_ok"],
                    "abstain_correct": scores["abstain_correct"],
                    "prefix_cacheable_tokens": _prefix_cacheable_tokens(strategy_name, strategy_result["prompt"]),
                    "strategy_build_cost_usd": 0.0,
                    "strategy_metadata": strategy_meta,
                    "matched_expected_facts": score["matched_expected_facts"],
                    "cited_source_ids": score["cited_source_ids"],
                    "unknown_citation_ids": score["unknown_citation_ids"],
                    "schema_errors": score["schema_errors"],
                    "abstained": score["abstained"],
                    "abstain_expected": score["abstain_expected"],
                    "answer_excerpt": answer[:400],
                }
                record = TraceRecord(
                    schema_version=SCHEMA_VERSION,
                    run_id=run_id,
                    source=source,
                    request_id=request_id,
                    ts_arrival_s=round(ts_arrival_s, 6),
                    strategy=strategy_name,
                    model=args.model if not args.mock else "simulated",
                    endpoint=endpoint,
                    input_tokens=completion.input_tokens if completion.input_tokens is not None else estimate_tokens(_messages_text(messages)),
                    output_tokens=completion.output_tokens if completion.output_tokens is not None else estimate_tokens(answer),
                    queue_wait_s=None,
                    ttft_s=round(completion.ttft_s, 6),
                    tpot_s=round(completion.tpot_s, 6) if completion.tpot_s is not None else None,
                    latency_s=round(completion.latency_s, 6),
                    error=error,
                    cost_usd=None,
                    meta=meta,
                )
                validate_record(record)
                write_jsonl_record(traces_path, record)
                n_requests += 1

    write_run_sidecar(
        run_dir,
        run_id=run_id,
        source=source,
        repo="context-budget-lab",
        config={
            "datasets": dataset_ids,
            "strategies": strategy_names,
            "limit": args.limit,
            "model": args.model,
            "base_url": args.base_url,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "mock": args.mock,
        },
        environment={"machine": "m3-pro-local", "notes": "local-first Project 2 smoke"},
        n_requests=n_requests,
    )
    print(run_dir)
    return 0


def _failed_completion(messages: list[dict[str, str]], latency_s: float) -> Any:
    observed_latency_s = max(0.0, latency_s)
    return SimpleNamespace(
        text="",
        input_tokens=estimate_tokens(_messages_text(messages)),
        output_tokens=0,
        ttft_s=observed_latency_s,
        tpot_s=None,
        latency_s=observed_latency_s,
    )


def _messages_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(message["content"] for message in messages)


def _messages_from_prompt(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def _prefix_cacheable_tokens(strategy_name: str, prompt: str) -> int:
    if strategy_name != "prefix_cache_friendly":
        return 0
    prefix = prompt.split("Payload:", 1)[0]
    return estimate_tokens(prefix)


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
