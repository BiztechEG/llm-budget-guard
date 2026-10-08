"""Real SDK clients talking to a fake HTTP transport, so no request leaves the machine."""

import json
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List

import httpx2
import pytest

import openai
import anthropic

from budget_guard import BudgetGuard, InMemoryStorage


def sse(events: List[Any], named: bool = False, done: bool = False) -> bytes:
    out = []
    for event in events:
        if named:
            out.append(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n")
        else:
            out.append(f"data: {json.dumps(event)}\n\n")
    if done:
        out.append("data: [DONE]\n\n")
    return "".join(out).encode()


def stream_response(body: bytes) -> httpx2.Response:
    return httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})


class FakeServer:
    """Records each request and answers with whatever the test queued."""

    def __init__(self) -> None:
        self.requests: List[Dict[str, Any]] = []
        self.responder: Callable[[Dict[str, Any]], httpx2.Response] = lambda body: httpx2.Response(500)

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content) if request.content else {}
        self.requests.append(body)
        return self.responder(body)

    def sync_http(self) -> httpx2.Client:
        return httpx2.Client(transport=httpx2.MockTransport(self.handle))

    def async_http(self) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(self.handle))


@pytest.fixture
def server() -> FakeServer:
    return FakeServer()


@pytest.fixture
def openai_client(server):
    return openai.OpenAI(api_key="test", http_client=server.sync_http(), max_retries=0)


@pytest.fixture
def async_openai_client(server):
    return openai.AsyncOpenAI(api_key="test", http_client=server.async_http(), max_retries=0)


@pytest.fixture
def anthropic_client(server):
    return anthropic.Anthropic(api_key="test", http_client=server.sync_http(), max_retries=0)


@pytest.fixture
def async_anthropic_client(server):
    return anthropic.AsyncAnthropic(api_key="test", http_client=server.async_http(), max_retries=0)


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def budget(clock) -> BudgetGuard:
    return BudgetGuard(storage=InMemoryStorage(), clock=clock)


# -- canned provider payloads ----------------------------------------------

def chat_completion(model: str, prompt: int, completion: int, cached: int = 0) -> Dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
    }


def chat_chunks(model: str, prompt: int, completion: int, with_usage: bool) -> List[Dict[str, Any]]:
    base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": model}
    chunks = [
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}, "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}]},
    ]
    if with_usage:
        chunks.append(
            {**base, "choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}}
        )
    return chunks


def responses_object(model: str, inp: int, out: int, status: str = "completed") -> Dict[str, Any]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "model": model,
        "status": status,
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Hello", "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": inp,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": out,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": inp + out,
        },
    }


def responses_events(model: str, inp: int, out: int) -> List[Dict[str, Any]]:
    created = {**responses_object(model, inp, out, status="in_progress"), "output": [], "usage": None}
    return [
        {"type": "response.created", "sequence_number": 0, "response": created},
        {"type": "response.completed", "sequence_number": 1, "response": responses_object(model, inp, out)},
    ]


def anthropic_message(model: str, inp: int, out: int, cache_read: int = 0) -> Dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": inp, "output_tokens": out, "cache_read_input_tokens": cache_read},
    }


def anthropic_events(model: str, inp: int, out: int, cache_read: int = 0) -> List[Dict[str, Any]]:
    start = {**anthropic_message(model, inp, 1, cache_read), "content": [], "stop_reason": None}
    return [
        {"type": "message_start", "message": start},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": out}},
        {"type": "message_stop"},
    ]
