import json
import os
import sys
import types
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from client_factory import lcmx_client  # noqa: E402

BASE_HITS = [
    {"kind": "message_excerpt", "store_id": 1, "session_id": "s1", "snippet": "one >>>cut<<<",
     "event_time": "2023-05-01T10:00:00Z", "score": 0.5},
    {"kind": "summary", "node_id": 7, "session_id": "s2", "snippet": "a  summary\nline", "score": 0.4},
    {"kind": "message_excerpt", "store_id": 3, "session_id": "s3", "snippet": "three", "score": 0.3},
]
EXTRAS = [
    {"kind": "message_excerpt", "store_id": 4, "session_id": "s4", "event_time": "2023-05-02T09:00:00Z",
     "window": "four  window\ntext", "extra": True},
    {"kind": "message_excerpt", "store_id": 5, "session_id": "s3", "event_time": None,
     "window": "five window", "extra": True},
    {"kind": "summary", "store_id": None, "session_id": "s2", "event_time": None,
     "window": "summary window", "extra": True},
    {"kind": "message_excerpt", "store_id": 6, "session_id": "s6", "event_time": None,
     "window": "  ", "extra": True},
]
ROWS = {1: {"content": "one full\n stored row"}, 3: {"content": "three full row"},
        4: {"content": "four FULL ROW must not appear"}, 5: {"content": "five FULL ROW"}}
SESSIONS = {"s2": {"time": "2023-04-30"}, "s3": {"time": "2023-04-29"}}
# Rendered by the unchanged adapter (2aaf336) for BASE_HITS at top_k=3.
BASE_TEXT = ("[2023-05-01T10:00:00Z] one full stored row\n"
             "[2023-04-30] (session summary) a summary line\n"
             "[2023-04-29] three full row")


def _search(monkeypatch, tmp_path, hits, top_k=3):
    batches = []
    store = SimpleNamespace(get_batch=lambda ids: batches.append(list(ids)) or {i: ROWS[i] for i in ids if i in ROWS},
                            close=lambda: None)
    closer = SimpleNamespace(close=lambda: None)
    fake_tools = types.ModuleType("hermes_lcm.tools")
    fake_tools.lcm_recall = lambda args, engine: json.dumps({"hits": hits})
    pkg = types.ModuleType("hermes_lcm")
    pkg.tools = fake_tools
    monkeypatch.setitem(sys.modules, "hermes_lcm", pkg)
    monkeypatch.setitem(sys.modules, "hermes_lcm.tools", fake_tools)
    c = lcmx_client.LcmxClient.__new__(lcmx_client.LcmxClient)
    (tmp_path / "lcm.db").write_text("")
    c.__dict__.update(_db_root=tmp_path, _embeddings=False, _scope_bias=None, _top_k_cap=25, _delivery="full")
    c._require_healthy_store = lambda user_id: None
    c._db_path = lambda user_id: tmp_path / "lcm.db"
    c._open = lambda user_id: (SimpleNamespace(), store, closer, closer)
    c._record_search_payload = lambda user_id, payload: None
    c._log_recall_call = lambda *a: None
    c._load_sessions = lambda user_id: SESSIONS
    return c.search("q", "u", top_k), batches


def test_off_by_default_base_bytes_and_extras_cut(monkeypatch, tmp_path):
    monkeypatch.delenv("LCMX_EXTRA_ENTRIES", raising=False)
    assert _search(monkeypatch, tmp_path, BASE_HITS)[0] == BASE_TEXT
    # Flag off, a payload carrying extras renders exactly as today: cut at top_k.
    assert _search(monkeypatch, tmp_path, BASE_HITS + EXTRAS)[0] == BASE_TEXT


@pytest.mark.parametrize("value", ["", "0", "false"])
def test_other_values_stay_off(monkeypatch, tmp_path, value):
    monkeypatch.setenv("LCMX_EXTRA_ENTRIES", value)
    assert _search(monkeypatch, tmp_path, BASE_HITS + EXTRAS)[0] == BASE_TEXT


def test_on_without_extras_is_byte_identical(monkeypatch, tmp_path):
    monkeypatch.setenv("LCMX_EXTRA_ENTRIES", "1")
    text, batches = _search(monkeypatch, tmp_path, BASE_HITS)
    assert text == BASE_TEXT and batches == [[1, 3]]


def test_on_renders_extras_as_window_after_base(monkeypatch, tmp_path):
    monkeypatch.setenv("LCMX_EXTRA_ENTRIES", "1")
    text, batches = _search(monkeypatch, tmp_path, BASE_HITS + EXTRAS)
    assert text == BASE_TEXT + "\n" + "\n".join([
        "[2023-05-02T09:00:00Z] four window text",
        "[2023-04-29] five window",
        "[2023-04-30] (session summary) summary window",
    ])
    # Extras never read their full stored row; base hits read the same ids as before.
    assert batches == [[1, 3]]


def test_on_extras_not_cut_at_top_k(monkeypatch, tmp_path):
    monkeypatch.setenv("LCMX_EXTRA_ENTRIES", "1")
    text, _ = _search(monkeypatch, tmp_path, BASE_HITS[:1] + EXTRAS[:2], top_k=1)
    assert text.splitlines() == ["[2023-05-01T10:00:00Z] one full stored row",
                                 "[2023-05-02T09:00:00Z] four window text", "[2023-04-29] five window"]


def test_on_top_k_above_delivered_limit_keeps_extras_as_windows(monkeypatch, tmp_path):
    # LoCoMo runs ask for top_k 40 while lcm-x delivers at most 25 full hits: extras must
    # still render from their window and never as full stored rows.
    monkeypatch.setenv("LCMX_EXTRA_ENTRIES", "1")
    text, batches = _search(monkeypatch, tmp_path, BASE_HITS + EXTRAS, top_k=40)
    assert "FULL ROW" not in text and batches == [[1, 3]]
    assert text.splitlines()[3:] == ["[2023-05-02T09:00:00Z] four window text", "[2023-04-29] five window",
                                     "[2023-04-30] (session summary) summary window"]
