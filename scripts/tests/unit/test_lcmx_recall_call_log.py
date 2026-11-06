import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from client_factory import lcmx_client  # noqa: E402


def _client(tmp_path):
    c = lcmx_client.LcmxClient.__new__(lcmx_client.LcmxClient)
    c._db_root = tmp_path
    c.set_search_results_dir(tmp_path / "res")
    return c


def test_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("LCMX_RECALL_CALL_LOG", raising=False)
    _client(tmp_path)._log_recall_call("u", "q", SimpleNamespace(), {"hits": []})
    assert not (tmp_path / "res" / "lcmx_recall_calls.jsonl").exists()


def test_logs_arm_config_and_withheld(tmp_path, monkeypatch):
    monkeypatch.setenv("LCMX_RECALL_CALL_LOG", "1")
    monkeypatch.setenv("LCMX_ARM", "candidate")
    monkeypatch.setenv("LCMX_CONFIG_SET", "recall_honour_forget=true")
    monkeypatch.setenv("LCMX_LOG_CONFIG_FIELDS", "recall_delivery_order")
    payload = {"hits": [{"store_id": 3}, {"store_id": 9}],
               "provenance": {"forget_withheld": [{"store_id": 7, "forget_store_id": 11}]}}
    c = _client(tmp_path)
    c._log_recall_call("u", "secret question", SimpleNamespace(recall_honour_forget=True), payload)
    line = json.loads((tmp_path / "res" / "lcmx_recall_calls.jsonl").read_text())
    assert line["arm"] == "candidate"
    assert line["config"] == {"recall_honour_forget": True, "recall_delivery_order": "<absent>"}
    assert line["forget_withheld"] == [7] and line["hit_store_ids"] == [3, 9]
    assert "secret question" not in json.dumps(line)
