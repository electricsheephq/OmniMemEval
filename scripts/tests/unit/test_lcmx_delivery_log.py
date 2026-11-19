import hashlib
import json
import os
import sys
import threading
import types
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from client_factory import lcmx_client  # noqa: E402

# Synthetic fixture; hermes_lcm is faked (no store, no network).
SESSIONS = {"s1": {"order": 1, "time": "2023-05-01T09:00:00+00:00"},
            "s2": {"order": 2, "time": "2023-05-08T18:30:00+00:00"}}
ROWS = {1: {"role": "user", "content": "I adopted a  beagle\nnamed Biscuit."},
        4: {"role": "user", "content": "Biscuit turned two today."}}
HITS = [
    {"kind": "message_excerpt", "store_id": 1, "session_id": "s1", "snippet": "adopted a >>>beagle<<<",
     "event_time": "2023-05-01T09:00:00Z"},
    {"kind": "message_excerpt", "store_id": 2, "session_id": "s1", "snippet": " >>><<< "},  # empty: skipped
    {"kind": "summary", "node_id": 11, "session_id": "s2", "snippet": "user: Biscuit turned two."},
    {"kind": "message_excerpt", "store_id": 4, "session_id": "s2", "role": "user", "snippet": "turned two"},
    {"kind": "message_excerpt", "store_id": 9, "session_id": "s2", "snippet": "past top_k"},  # beyond top_k
]
EXTRAS = [
    {"kind": "message_excerpt", "store_id": 5, "session_id": "s2", "window": "Happy birthday!", "extra": True},
    {"kind": "message_excerpt", "store_id": 6, "session_id": "s1", "window": "  ", "extra": True},  # empty
    {"kind": "summary", "node_id": 12, "session_id": "s1", "window": "summary window", "extra": True},
]
SOURCES = {11: [3, 4, 5], 12: [1, 2]}
# Captured from the unchanged adapter (b468f655) on this fixture with top_k=4, before the hook.
_BASE = ("[2023-05-01T09:00:00Z] {r}I adopted a beagle named Biscuit.\n"
         "[2023-05-08T18:30:00+00:00] (session summary) user: Biscuit turned two.\n"
         "[2023-05-08T18:30:00+00:00] {r}Biscuit turned two today.")
_EXTRAS = "\n[2023-05-08T18:30:00+00:00] Happy birthday!\n[2023-05-01T09:00:00+00:00] (session summary) summary window"
GOLDEN = {(v, x): _BASE.format(r="user: " if v == "2" else "") + (_EXTRAS if x else "")
          for v in ("1", "2") for x in (False, True)}


class FakeDag:
    def __init__(self):
        self.calls = []

    def source_message_ids(self, node_id, *, limit):
        self.calls.append((node_id, limit))
        return list(SOURCES[node_id])

    def close(self):
        pass


def _client(monkeypatch, tmp_path, hits, dag=None):
    store = SimpleNamespace(get_batch=lambda ids: {i: ROWS[i] for i in ids if i in ROWS}, close=lambda: None)
    fake_tools = types.ModuleType("hermes_lcm.tools")
    fake_tools.lcm_recall = lambda args, engine: json.dumps({"hits": hits})
    pkg = types.ModuleType("hermes_lcm")
    pkg.tools = fake_tools
    monkeypatch.setitem(sys.modules, "hermes_lcm", pkg)
    monkeypatch.setitem(sys.modules, "hermes_lcm.tools", fake_tools)
    monkeypatch.setattr(lcmx_client, "_DELIVERY_CALLS", {})
    c = lcmx_client.LcmxClient.__new__(lcmx_client.LcmxClient)
    (tmp_path / "lcm.db").write_text("")
    c.__dict__.update(_db_root=tmp_path, _embeddings=False, _scope_bias=None, _top_k_cap=25, _delivery="full")
    c._require_healthy_store = lambda user_id: None
    c._db_path = lambda user_id: tmp_path / "lcm.db"
    c._open = lambda user_id: (SimpleNamespace(), store, dag or FakeDag(), SimpleNamespace(close=lambda: None))
    c._record_search_payload = lambda user_id, payload: None
    c._log_recall_call = lambda *a: None
    c._load_sessions = lambda user_id: SESSIONS
    return c


@pytest.mark.parametrize("version", ["1", "2"])
@pytest.mark.parametrize("extras", [False, True])
def test_on_and_off_return_identical_text_and_off_writes_nothing(monkeypatch, tmp_path, version, extras):
    monkeypatch.setenv("LCMX_ADAPTER_VERSION", version)
    monkeypatch.delenv("LCMX_EXTRA_ENTRIES", raising=False)
    if extras:
        monkeypatch.setenv("LCMX_EXTRA_ENTRIES", "1")
    hits = HITS + (EXTRAS if extras else [])
    log = tmp_path / "out" / "delivery.jsonl"
    log.parent.mkdir()
    monkeypatch.delenv("LCMX_DELIVERY_LOG", raising=False)
    dag = FakeDag()
    off = _client(monkeypatch, tmp_path, hits, dag).search("q", "u", 4)
    assert off == GOLDEN[(version, extras)]  # byte-identical to b468f655
    assert dag.calls == [] and list(log.parent.iterdir()) == []  # off: no lineage read, no file
    monkeypatch.setenv("LCMX_DELIVERY_LOG", str(log))
    assert _client(monkeypatch, tmp_path, hits).search("q", "u", 4) == off
    assert len(log.read_text().splitlines()) == 1


def test_record_follows_rendered_order(monkeypatch, tmp_path):
    monkeypatch.setenv("LCMX_ADAPTER_VERSION", "2")
    monkeypatch.setenv("LCMX_EXTRA_ENTRIES", "1")
    monkeypatch.setenv("LCMX_ARM", "arm-a")
    log = tmp_path / "delivery.jsonl"
    monkeypatch.setenv("LCMX_DELIVERY_LOG", str(log))
    dag = FakeDag()
    text = _client(monkeypatch, tmp_path, HITS + EXTRAS, dag).search("secret question", "u", 4)
    rendered = text.split("\n")
    record = json.loads(log.read_text())
    assert dag.calls == [(11, sys.maxsize), (12, sys.maxsize)]  # never truncated
    assert isinstance(record.pop("ts"), float) and record.pop("pid") == os.getpid()
    assert record.pop("arm") == "arm-a"
    assert record == {
        "user_id": "u",
        "query_sha256": hashlib.sha256(b"secret question").hexdigest(),
        "call_index": 0,
        "adapter_version": 2,
        "lines": [
            {"kind": "message", "store_id": 1, "session_id": "s1", "chars": len(rendered[0]), "position": 0},
            {"kind": "summary", "node_id": 11, "session_id": "s2", "source_store_ids": [3, 4, 5],
             "chars": len(rendered[1]), "position": 1},
            {"kind": "message", "store_id": 4, "session_id": "s2", "chars": len(rendered[2]), "position": 2},
            {"kind": "extra", "store_id": 5, "session_id": "s2", "chars": len(rendered[3]), "position": 3},
            {"kind": "extra", "node_id": 12, "session_id": "s1", "source_store_ids": [1, 2],
             "chars": len(rendered[4]), "position": 4},
        ],
        "skipped": [
            {"kind": "message", "store_id": 2, "session_id": "s1", "reason": "empty_text"},
            {"kind": "extra", "store_id": 6, "session_id": "s1", "reason": "empty_text"},
            {"kind": "message", "store_id": 9, "session_id": "s2", "reason": "beyond_top_k"},
        ],
    }
    assert rendered[1].startswith("[2023-05-08T18:30:00+00:00] (session summary) ")
    assert "secret question" not in log.read_text() and "Biscuit" not in log.read_text()


def test_records_are_whole_lines_under_concurrency(monkeypatch, tmp_path):
    monkeypatch.delenv("LCMX_EXTRA_ENTRIES", raising=False)
    log = tmp_path / "delivery.jsonl"
    monkeypatch.setenv("LCMX_DELIVERY_LOG", str(log))
    monkeypatch.delenv("LCMX_ARM", raising=False)
    c = _client(monkeypatch, tmp_path, HITS)
    users, threads_per_user, calls = ["a", "b", "c"], 4, 25
    errors = []

    def run(user):
        try:
            for _ in range(calls):
                c.search("q", user, 4)
        except Exception as exc:  # noqa: BLE001 - surfaced by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(u,)) for u in users for _ in range(threads_per_user)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == len(users) * threads_per_user * calls
    assert {r["arm"] for r in records} == {None} and {r["pid"] for r in records} == {os.getpid()}
    for user in users:  # per-user call_index is dense and in file order
        assert [r["call_index"] for r in records if r["user_id"] == user] == list(range(threads_per_user * calls))


def test_missing_store_still_writes_an_empty_record(monkeypatch, tmp_path):
    log = tmp_path / "delivery.jsonl"
    monkeypatch.setenv("LCMX_DELIVERY_LOG", str(log))
    c = _client(monkeypatch, tmp_path, HITS)
    c._db_path = lambda user_id: tmp_path / "absent.db"
    assert c.search("q", "u", 4) == ""
    assert json.loads(log.read_text())["lines"] == []


@pytest.mark.parametrize("bad", ["relative.jsonl", "{tmp}/delivery.json", "{tmp}/missing/delivery.jsonl",
                                 "{tmp}/dir.jsonl"])
def test_bad_path_fails_at_startup(monkeypatch, tmp_path, bad):
    (tmp_path / "dir.jsonl").mkdir()
    for name, value in (("LCMX_REPO", str(tmp_path)), ("LCMX_DB_ROOT", str(tmp_path / "db")),
                        ("LCMX_DELIVERY_LOG", bad.format(tmp=tmp_path))):
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("LCMX_ADAPTER_VERSION", raising=False)
    with pytest.raises(ValueError, match="LCMX_DELIVERY_LOG"):
        lcmx_client.LcmxClient()
    assert not (tmp_path / "db").exists()
