import asyncio

import pytest

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils import llm_client  # noqa: E402


class _Completions:
    def __init__(self):
        self.calls = []

    def create(self, *args, **kwargs):
        self.calls.append(kwargs)
        return {"usage": {"prompt_tokens": 1, "completion_tokens": 1}, "model": "m"}


class _ACompletions(_Completions):
    async def create(self, *args, **kwargs):
        self.calls.append(kwargs)
        return {"usage": {"prompt_tokens": 1, "completion_tokens": 1}, "model": "m"}


class _Tracker:
    def record(self, *a, **k):
        pass


def _cfg(module):
    return llm_client.LLMRetryConfig(module, 0, 0.0, 0.0, 1.0)


def test_unset_leaves_request_unchanged(monkeypatch):
    monkeypatch.delenv("EVAL_EXTRA_BODY", raising=False)
    inner = _Completions()
    llm_client._TrackedSyncCompletions(inner, _Tracker(), _cfg("EVAL")).create(model="m", messages=[])
    assert inner.calls == [{"model": "m", "messages": []}]


def test_extra_body_merged_sync_and_async(monkeypatch):
    monkeypatch.setenv("EVAL_EXTRA_BODY", '{"thinking": {"type": "disabled"}}')
    monkeypatch.delenv("ANSWER_EXTRA_BODY", raising=False)
    inner = _Completions()
    llm_client._TrackedSyncCompletions(inner, _Tracker(), _cfg("EVAL")).create(
        model="m", messages=[], extra_body={"keep": 1}
    )
    assert inner.calls[0]["extra_body"] == {"keep": 1, "thinking": {"type": "disabled"}}
    ainner = _ACompletions()
    asyncio.run(llm_client._TrackedAsyncCompletions(ainner, _Tracker(), _cfg("EVAL")).create(model="m"))
    assert ainner.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    other = _Completions()
    llm_client._TrackedSyncCompletions(other, _Tracker(), _cfg("ANSWER")).create(model="m")
    assert "extra_body" not in other.calls[0]


def test_extra_body_must_be_object(monkeypatch):
    monkeypatch.setenv("EVAL_EXTRA_BODY", "[1]")
    with pytest.raises(ValueError):
        llm_client._TrackedSyncCompletions(_Completions(), _Tracker(), _cfg("EVAL")).create(model="m")
