import json
import os
import sys
import types
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from client_factory import lcmx_client  # noqa: E402

# Synthetic two-session fixture; hermes_lcm is faked (no store, no network).
SESSIONS = {"s1": {"order": 1, "time": "2023-05-01T09:00:00+00:00"},
            "s2": {"order": 2, "time": "2023-05-08T18:30:00+00:00"}}
ROWS = {1: {"role": "user", "content": "I adopted a  beagle\nnamed Biscuit."},
        2: {"role": "assistant", "content": "Biscuit is a great name."},
        4: {"role": "user", "content": "Biscuit turned two today."}}
HITS = [
    {"kind": "message_excerpt", "store_id": 1, "session_id": "s1", "snippet": "adopted a >>>beagle<<<",
     "event_time": "2023-05-01T09:00:00Z", "score": 0.9},
    {"kind": "summary", "node_id": 11, "session_id": "s2",
     "snippet": "user: Biscuit turned two.\nassistant: Happy birthday.", "score": 0.8},
    {"kind": "message_excerpt", "store_id": 2, "session_id": "s1", "snippet": "great name", "score": 0.7},
    {"kind": "message_excerpt", "store_id": 3, "session_id": "s2", "snippet": "a >>>cake<<< for the dog",
     "score": 0.6},
    {"kind": "message_excerpt", "store_id": 4, "session_id": "s2", "role": "user", "snippet": "turned two",
     "event_time": "2023-05-08T18:31:00Z", "score": 0.5},
]
EXTRAS = [
    {"kind": "message_excerpt", "store_id": 5, "session_id": "s2", "event_time": "2023-05-08T18:32:00Z",
     "role": "assistant", "window": "Happy  birthday,\nBiscuit!", "extra": True},
    {"kind": "message_excerpt", "store_id": 6, "session_id": "s1", "event_time": None,
     "window": "no role window", "extra": True},
    {"kind": "summary", "session_id": "s1", "window": "summary window", "extra": True},
]
# Captured from the unchanged adapter (aa7d96c) on this fixture before the v2 change.
GOLDEN_BASE = (
    "[2023-05-01T09:00:00Z] I adopted a beagle named Biscuit.\n"
    "[2023-05-08T18:30:00+00:00] (session summary) user: Biscuit turned two. assistant: Happy birthday.\n"
    "[2023-05-01T09:00:00+00:00] Biscuit is a great name.\n"
    "[2023-05-08T18:30:00+00:00] a cake for the dog\n"
    "[2023-05-08T18:31:00Z] Biscuit turned two today."
)
GOLDEN_EXTRAS = GOLDEN_BASE + (
    "\n[2023-05-08T18:32:00Z] Happy birthday, Biscuit!"
    "\n[2023-05-01T09:00:00+00:00] no role window"
    "\n[2023-05-01T09:00:00+00:00] (session summary) summary window"
)


def _env(monkeypatch, name, value):
    monkeypatch.delenv(name, raising=False) if value is None else monkeypatch.setenv(name, value)


def _search(monkeypatch, tmp_path, hits, delivery="full"):
    store = SimpleNamespace(get_batch=lambda ids: {i: ROWS[i] for i in ids if i in ROWS}, close=lambda: None)
    closer = SimpleNamespace(close=lambda: None)
    fake_tools = types.ModuleType("hermes_lcm.tools")
    fake_tools.lcm_recall = lambda args, engine: json.dumps({"hits": hits})
    pkg = types.ModuleType("hermes_lcm")
    pkg.tools = fake_tools
    monkeypatch.setitem(sys.modules, "hermes_lcm", pkg)
    monkeypatch.setitem(sys.modules, "hermes_lcm.tools", fake_tools)
    c = lcmx_client.LcmxClient.__new__(lcmx_client.LcmxClient)
    (tmp_path / "lcm.db").write_text("")
    c.__dict__.update(_db_root=tmp_path, _embeddings=False, _scope_bias=None, _top_k_cap=25, _delivery=delivery)
    c._require_healthy_store = lambda user_id: None
    c._db_path = lambda user_id: tmp_path / "lcm.db"
    c._open = lambda user_id: (SimpleNamespace(), store, closer, closer)
    c._record_search_payload = lambda user_id, payload: None
    c._log_recall_call = lambda *a: None
    c._load_sessions = lambda user_id: SESSIONS
    return c.search("q", "u", 5)


@pytest.mark.parametrize("version", [None, "1", " 1 "])
def test_v1_golden_is_byte_identical(monkeypatch, tmp_path, version):
    _env(monkeypatch, "LCMX_ADAPTER_VERSION", version)
    _env(monkeypatch, "LCMX_EXTRA_ENTRIES", None)
    assert _search(monkeypatch, tmp_path, HITS) == GOLDEN_BASE
    monkeypatch.setenv("LCMX_EXTRA_ENTRIES", "1")
    assert _search(monkeypatch, tmp_path, HITS + EXTRAS) == GOLDEN_EXTRAS


def test_v2_role_and_time_on_message_lines_and_missing_role_omitted(monkeypatch, tmp_path):
    monkeypatch.setenv("LCMX_ADAPTER_VERSION", "2")
    monkeypatch.setenv("LCMX_EXTRA_ENTRIES", "1")
    assert _search(monkeypatch, tmp_path, HITS + EXTRAS).splitlines() == [
        "[2023-05-01T09:00:00Z] user: I adopted a beagle named Biscuit.",  # role from the full row
        "[2023-05-08T18:30:00+00:00] (session summary) user: Biscuit turned two. assistant: Happy birthday.",
        "[2023-05-01T09:00:00+00:00] assistant: Biscuit is a great name.",
        "[2023-05-08T18:30:00+00:00] a cake for the dog",  # no hit role, no stored row: omitted
        "[2023-05-08T18:31:00Z] user: Biscuit turned two today.",  # role from the hit
        "[2023-05-08T18:32:00Z] assistant: Happy birthday, Biscuit!",
        "[2023-05-01T09:00:00+00:00] no role window",
        "[2023-05-01T09:00:00+00:00] (session summary) summary window",
    ]
    # Snippet delivery reads no stored row, so only hit roles render.
    monkeypatch.delenv("LCMX_EXTRA_ENTRIES")
    assert _search(monkeypatch, tmp_path, HITS, "snippet").splitlines()[0] == "[2023-05-01T09:00:00Z] adopted a beagle"


def test_invalid_version_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("LCMX_ADAPTER_VERSION", "3")
    with pytest.raises(ValueError, match="LCMX_ADAPTER_VERSION"):
        _search(monkeypatch, tmp_path, HITS)
    with pytest.raises(ValueError, match="LCMX_ADAPTER_VERSION"):  # before any store write
        lcmx_client.LcmxClient.__new__(lcmx_client.LcmxClient).add([{"role": "user", "content": "x"}], "u")


@pytest.mark.parametrize("version,chat_time,created_at", [
    (None, "2023-05-08T18:30:00Z", 1.0),
    ("2", "2023-05-08T18:30:00Z", 1683570600.0),
    ("2", "2023-05-08T18:30:00+07:00", 1683545400.0),
    ("2", " 1683570600 ", 1683570600.0),  # numeric string: the store accepts it too
    ("2", "nan", 1.0),
    ("2", "1:56 pm on 8 May, 2023", 1.0),  # not a time: keep the order
    ("2", "2023-05-08T18:30:00", 1.0),  # naive: the store rejects it too
    ("2", None, 1.0),
])
def test_summary_created_at(monkeypatch, tmp_path, version, chat_time, created_at):
    _env(monkeypatch, "LCMX_ADAPTER_VERSION", version)
    c = lcmx_client.LcmxClient.__new__(lcmx_client.LcmxClient)
    c.__dict__.update(_db_root=tmp_path, _embeddings=False, _summary=lambda rows: "summary")
    for name in ("hermes_lcm", "hermes_lcm.chunking", "hermes_lcm.dag", "hermes_lcm.vector_store"):
        module = ModuleType(name)
        module.iter_message_chunks, module.SummaryNode, module.EmbeddingIdentity = Mock(), SimpleNamespace, Mock()
        monkeypatch.setitem(sys.modules, name, module)
    store, dag = Mock(), Mock()
    store.append_batch.return_value = [1]
    c._open = Mock(return_value=(object(), store, dag, Mock()))
    c._catch_up = Mock()
    message = {"role": "user", "content": "text"}
    c.add([dict(message, chat_time=chat_time) if chat_time else message], "u", session_key="s1")
    assert dag.add_node.call_args.args[0].created_at == created_at
    assert json.loads(c._sessions_path("u").read_text()) == {"s1": {"order": 1, "time": chat_time}}


@pytest.mark.parametrize("version,expected", [(None, 1), ("1", 1), ("2", 2)])
def test_adapter_version_recorded_in_stage_report(monkeypatch, tmp_path, version, expected):
    _env(monkeypatch, "LCMX_ADAPTER_VERSION", version)
    c = lcmx_client.LcmxClient.__new__(lcmx_client.LcmxClient)
    c.__dict__.update(_db_root=tmp_path, _embeddings=False)
    c._user_dir("u").mkdir()
    c.set_search_results_dir(tmp_path / "results")
    c._record_search_payload("u", {"hits": []})
    assert json.loads((tmp_path / "results/lcmx_recall_status.json").read_text()) == {
        "stage": "search", "payloads": 1, "degraded_payloads": 0, "adapter_version": expected}
