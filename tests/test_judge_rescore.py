from __future__ import annotations

import json
from pathlib import Path

from analysis.judge_rescore import (
    JudgeCallResult,
    RequestCase,
    build_report,
    parse_judge_text,
    run_judge_rescore,
)


def test_parse_judge_text_accepts_fenced_json() -> None:
    text = """```json
{"facts":[{"fact":"rapid","credited":true,"reason":"says fast"},{"fact":"safe","credited":false,"reason":"missing"}]}
```"""

    verdicts = parse_judge_text(text, ("rapid", "safe"))

    assert verdicts == [
        {"fact": "rapid", "credited": True, "reason": "says fast"},
        {"fact": "safe", "credited": False, "reason": "missing"},
    ]


def test_run_judge_rescore_dedupes_identical_task_answer_pairs(tmp_path: Path) -> None:
    cases = [
        _case(request_id="dataset:task-1:r0:full_context", repeat_index=0),
        _case(request_id="dataset:task-1:r1:full_context", repeat_index=1),
    ]
    client = FakeJudgeClient(
        [
            JudgeCallResult(
                verdicts=(
                    {"fact": "rapid", "credited": True, "reason": "fast is a paraphrase"},
                    {"fact": "safe", "credited": True, "reason": "safe appears"},
                ),
                raw_text='{"facts":[]}',
                input_tokens=100,
                output_tokens=40,
            )
        ]
    )

    report = run_judge_rescore(
        cases,
        client=client,
        out_dir=tmp_path,
        model="claude-haiku-4-5-20251001",
        max_cost_usd=1.0,
        input_price_per_mtok=1.0,
        output_price_per_mtok=5.0,
        max_tokens=100,
        resamples=20,
        seed=7,
    )

    assert client.calls == 1
    assert report["n_requests"] == 2
    assert report["n_unique_judge_cases"] == 1
    assert report["agreement_rows"][0]["literal_fact_coverage_mean"] == 0.5
    assert report["agreement_rows"][0]["judge_fact_coverage_mean"] == 1.0

    lines = (tmp_path / "judge_scores.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert {record["request_id"] for record in records} == {case.request_id for case in cases}
    assert records[0]["dedupe_key"] == records[1]["dedupe_key"]
    assert (tmp_path / "judge_disagreements.md").exists()
    assert (tmp_path / "proposed_fact_aliases.md").exists()


def test_build_report_computes_h2_from_paired_judge_scores() -> None:
    records = [
        _score_record("qwen2.5:3b", "full_context", "task-1", judge=0.5, literal=0.5),
        _score_record("qwen2.5:7b", "full_context", "task-1", judge=1.0, literal=0.5),
        _score_record("qwen2.5:3b", "rag_topk", "task-1", judge=0.0, literal=0.0),
        _score_record("qwen2.5:7b", "rag_topk", "task-1", judge=0.5, literal=0.0),
    ]

    report = build_report(records, resamples=20, seed=7)

    assert report["h2"]["verdict"] == "supported"
    assert [row["status"] for row in report["h2"]["rows"]] == ["supports_h2", "supports_h2"]
    assert report["agreement_rows"][0]["per_fact_agreement"] == 1.0


class FakeJudgeClient:
    def __init__(self, results: list[JudgeCallResult]) -> None:
        self.results = results
        self.calls = 0

    def judge(self, case: RequestCase, *, model: str, max_tokens: int) -> JudgeCallResult:
        del case, model, max_tokens
        result = self.results[self.calls]
        self.calls += 1
        return result


def _case(*, request_id: str, repeat_index: int) -> RequestCase:
    return RequestCase(
        request_id=request_id,
        run_id="run-1",
        result_set="matrix",
        run_dir="results/run-1",
        model="qwen2.5:3b",
        strategy="full_context",
        dataset_id="dataset",
        task_id="task-1",
        repeat_index=repeat_index,
        question="Which traits matter?",
        expected_answer="The answer should mention rapid and safe.",
        expected_facts=("rapid", "safe"),
        literal_matched_facts=("safe",),
        literal_fact_coverage=0.5,
        answer_text="The system is fast and safe.",
    )


def _score_record(model: str, strategy: str, task_id: str, *, judge: float, literal: float) -> dict[str, object]:
    judge_flags = [judge >= 0.5]
    literal_flags = [literal >= 0.5]
    return {
        "request_id": f"{model}:{strategy}:{task_id}",
        "result_set": "matrix",
        "model": model,
        "strategy": strategy,
        "task_id": task_id,
        "dedupe_key": f"{model}:{strategy}:{task_id}",
        "expected_facts": ["fact"],
        "literal_fact_coverage": literal,
        "judge_fact_coverage": judge,
        "literal_credit_flags": literal_flags,
        "judge_credit_flags": judge_flags,
        "verdicts": [{"fact": "fact", "credited": judge_flags[0], "reason": "fixture"}],
        "answer_text": "fixture answer",
    }
