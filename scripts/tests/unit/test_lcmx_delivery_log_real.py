"""Read-only proof for LCMX_DELIVERY_LOG on a real synthetic LCM-X store (needs LCMX_REPO).

Imports the real hermes_lcm; the fake-module tests patch sys.modules per test, so a combined run also passes.
"""
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from client_factory import lcmx_client  # noqa: E402

REPO = os.environ.get("LCMX_REPO", "")
pytestmark = pytest.mark.skipif(not REPO or not Path(REPO).is_dir(), reason="LCMX_REPO (an LCM-X tree) not set")


def _hashes(user_dir):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(user_dir.glob("lcm.db*"))}


def test_hook_reads_lineage_without_writing(monkeypatch, tmp_path):
    for name, value in (("LCMX_DB_ROOT", str(tmp_path / "db")), ("LCMX_CONFIG", "off")):
        monkeypatch.setenv(name, value)
    for name in ("LCMX_DELIVERY_LOG", "LCMX_EXTRA_ENTRIES", "LCMX_CONFIG_SET", "LCMX_ADAPTER_VERSION"):
        monkeypatch.delenv(name, raising=False)
    c = lcmx_client.LcmxClient()
    c.add([{"role": "user", "content": "I adopted a beagle named Biscuit.", "chat_time": "2023-05-01T09:00:00Z"},
           {"role": "assistant", "content": "Biscuit is a great name for a beagle."}], "u", session_key="s1")
    db, user_dir = c._db_path("u"), c._user_dir("u")
    conn = sqlite3.connect(db)
    # The adapter's add() leaves source_ids empty; set the lineage test-side so the read returns real ids.
    assert conn.execute("SELECT node_id, source_ids FROM summary_nodes").fetchall() == [(1, "[]")]
    with conn:
        conn.execute("UPDATE summary_nodes SET source_ids = '[1, 2]' WHERE node_id = 1")
    conn.close()

    import hermes_lcm.tools as tools
    from hermes_lcm.dag import SummaryDAG

    # FTS-only recall returns no cross-session summary hits; append the stored node as a summary
    # hit (the shape the semantic arm returns) so the hook reads the real DAG.
    recall = tools.lcm_recall

    def recall_with_summary(args, engine):
        payload = json.loads(recall(args, engine=engine))
        node = engine._dag.get_node(1)
        payload["hits"].append({"kind": "summary", "node_id": 1, "session_id": node.session_id,
                                "snippet": node.summary})
        return json.dumps(payload)

    monkeypatch.setattr(tools, "lcm_recall", recall_with_summary)
    changes = []
    read = SummaryDAG.source_message_ids

    def counted(self, node_id, *, limit):
        before = self.connection.total_changes
        result = read(self, node_id, limit=limit)
        changes.append(self.connection.total_changes - before)
        return result

    monkeypatch.setattr(SummaryDAG, "source_message_ids", counted)

    off = c.search("beagle", "u", 5)  # warm open: any first-open writes land before the baseline
    baseline = _hashes(user_dir)
    assert c.search("beagle", "u", 5) == off and _hashes(user_dir) == baseline and changes == []
    log = tmp_path / "delivery.jsonl"
    monkeypatch.setenv("LCMX_DELIVERY_LOG", str(log))
    assert c.search("beagle", "u", 5) == off
    assert _hashes(user_dir) == baseline and changes == [0]
    record = json.loads(log.read_text())
    assert [line["kind"] for line in record["lines"]] == ["message", "message", "summary"]
    assert record["lines"][2]["source_store_ids"] == [1, 2] and record["skipped"] == []
