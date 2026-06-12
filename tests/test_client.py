from __future__ import annotations

from collections.abc import Iterator

from context_budget_lab.client import MockClient, OpenAICompatClient
import context_budget_lab.client as client_module


class FakeResponse:
    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def __iter__(self) -> Iterator[bytes]:
        yield b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n'
        yield b'data: {"choices":[{"delta":{"content":" world"}}]}\n'
        yield b'data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":12,"completion_tokens":3}}\n'
        yield b"data: [DONE]\n"


def test_openai_compat_client_streams_ttft_and_tpot(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    times = iter([10.0, 10.05, 10.2])

    monkeypatch.setattr(client_module.request, "urlopen", lambda *args, **kwargs: FakeResponse())
    monkeypatch.setattr(client_module.time, "perf_counter", lambda: next(times))

    result = OpenAICompatClient("http://localhost:11434/v1", "qwen2.5:3b").complete(
        [{"role": "user", "content": "Say hello."}],
        max_tokens=8,
        temperature=0.0,
    )

    assert result.text == "Hello world"
    assert result.input_tokens == 12
    assert result.output_tokens == 3
    assert round(result.ttft_s, 3) == 0.05
    assert round(result.latency_s, 3) == 0.2
    assert round(result.tpot_s or 0.0, 3) == 0.075


def test_mock_client_simulates_distinct_ttft_and_latency() -> None:
    result = MockClient().complete(
        [{"role": "user", "content": "Question:\nWhat happened?\nSource:\n[S1] Example."}],
        max_tokens=20,
        temperature=0.0,
    )

    assert result.output_tokens is not None
    assert result.output_tokens > 1
    assert result.ttft_s < result.latency_s
    assert result.tpot_s is not None
