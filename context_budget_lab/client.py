"""Small OpenAI-compatible chat client used by the benchmark harness."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Protocol
from urllib import error, request


ChatMessage = dict[str, str]


@dataclass(frozen=True)
class CompletionResult:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    ttft_s: float
    tpot_s: float | None
    latency_s: float


class ChatClient(Protocol):
    def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
    ) -> CompletionResult:
        """Return one chat completion."""


class OpenAICompatClient:
    """Minimal `/v1/chat/completions` client for Ollama, llama.cpp, or hosted APIs."""

    def __init__(self, base_url: str, model: str, timeout_s: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
    ) -> CompletionResult:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        start = time.perf_counter()
        try:
            with request.urlopen(req, timeout=self.timeout_s) as response:
                raw = response.read()
        except error.URLError as exc:
            latency_s = time.perf_counter() - start
            raise RuntimeError(f"OpenAI-compatible request failed after {latency_s:.2f}s: {exc}") from exc

        latency_s = time.perf_counter() - start
        parsed = json.loads(raw.decode("utf-8"))
        choice = parsed.get("choices", [{}])[0]
        message = choice.get("message", {})
        text = str(message.get("content", "")).strip()
        usage = parsed.get("usage", {})
        output_tokens = _int_or_none(usage.get("completion_tokens"))
        input_tokens = _int_or_none(usage.get("prompt_tokens"))

        # Non-streaming endpoints do not expose first-token timing. Record the
        # observable latency as TTFT and leave TPOT unknown.
        return CompletionResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            ttft_s=latency_s,
            tpot_s=None,
            latency_s=latency_s,
        )


class MockClient:
    """Deterministic client for tests and no-network smoke runs."""

    def __init__(self, model: str = "mock-context-model") -> None:
        self.model = model

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
    ) -> CompletionResult:
        prompt = "\n".join(message["content"] for message in messages)
        source_ids = _extract_bracketed_source_ids(prompt)
        keywords = _extract_line_values(prompt, "Target answer keywords")
        if keywords:
            answer = f"Answer: {'; '.join(keywords[:4])}."
        else:
            answer = "Answer: insufficient evidence in the supplied context."
        if source_ids:
            answer += " Citations: " + " ".join(f"[{source_id}]" for source_id in source_ids[:2])
        answer = answer[: max(40, max_tokens * 6)]
        output_tokens = estimate_tokens(answer)
        latency_s = 0.001
        return CompletionResult(
            text=answer,
            input_tokens=estimate_tokens(prompt),
            output_tokens=output_tokens,
            ttft_s=latency_s,
            tpot_s=0.0 if output_tokens > 1 else None,
            latency_s=latency_s,
        )


def estimate_tokens(text: str) -> int:
    """Cheap tokenizer-independent approximation for local comparisons."""

    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _int_or_none(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _extract_bracketed_source_ids(text: str) -> list[str]:
    source_ids: list[str] = []
    for token in text.replace("\n", " ").split():
        if token.startswith("[") and token.endswith("]") and len(token) > 2:
            source_id = token.strip("[]").strip(".,:;")
            if source_id and source_id not in source_ids:
                source_ids.append(source_id)
    return source_ids


def _extract_line_values(text: str, prefix: str) -> list[str]:
    for line in text.splitlines():
        if line.startswith(prefix + ":"):
            values = line.split(":", 1)[1]
            return [value.strip() for value in values.split(",") if value.strip()]
    return []
