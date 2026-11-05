"""Offline R1 contracts: fake product modules, tiny SQLite metadata, no models."""

import json
import sqlite3
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import client_factory.lcmx_client as adapter_module
import pytest
from client_factory.lcmx_client import LcmxClient
from longmemeval.lme_common import build_search_result, write_question_id_map


@pytest.fixture
def client(tmp_path, monkeypatch):
    client = LcmxClient.__new__(LcmxClient)
    client._db_root = tmp_path
    client._embeddings = True
    client._provider_name = "fastembed"
    client._model = "test-model"
    client._delivery = "full"
    client._scope_bias = None
    client._top_k_cap = 25
    client._embedder = SimpleNamespace(model_id="test-model")
    client._user_dir("u").mkdir()
    package = ModuleType("hermes_lcm")
    package.__path__ = []
    tools = ModuleType("hermes_lcm.tools")
    tools._lcm_recall_catchup_reason = Mock()
    maintenance = ModuleType("hermes_lcm.embedding_maintenance")
    maintenance.run_incremental_embedding_pass = Mock()
    maintenance._scan_frontier = Mock(return_value=None)
    command = ModuleType("hermes_lcm.command")
    package.tools = tools
    package.embedding_maintenance = maintenance
    monkeypatch.setitem(sys.modules, "hermes_lcm", package)
    monkeypatch.setitem(sys.modules, "hermes_lcm.tools", tools)
    monkeypatch.setitem(sys.modules, "hermes_lcm.embedding_maintenance", maintenance)
    monkeypatch.setitem(sys.modules, "hermes_lcm.command", command)
    monkeypatch.setenv("LCMX_CATCHUP_MAX_PASSES", "3")
    seed_receipts(client)
    return client, tools, maintenance.run_incremental_embedding_pass


def seed_receipts(client, summary=0, chunk=False, *, missing=None):
    conn = sqlite3.connect(client._db_path("u"))
    with conn:
        conn.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT)")
        # cf558e999 db_bootstrap.py:1495: full product profile schema.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS lcm_embedding_profile("
            "identity_hash TEXT PRIMARY KEY, provider TEXT NOT NULL, model_name TEXT NOT NULL, "
            "revision TEXT NOT NULL DEFAULT '', dim INTEGER CHECK(dim BETWEEN 1 AND 4096), "
            "dtype TEXT NOT NULL DEFAULT 'float32', byteorder TEXT NOT NULL DEFAULT 'little', "
            "task TEXT NOT NULL DEFAULT 'summary', registered_at TEXT, active INTEGER DEFAULT 1, "
            "archived_at TEXT NULL, data_version INTEGER NOT NULL DEFAULT 0)"
        )
        for task, remaining in (("summary", summary), ("chunk", chunk)):
            conn.execute(
                "INSERT OR REPLACE INTO lcm_embedding_profile "
                "(identity_hash,task,provider,model_name,registered_at) VALUES(?,?,?,?,?)",
                (task + "-id", task, "fastembed", "test-model", "2026-11-05T00:00:00"),
            )
            key = f"embedding_catchup:{task}:{task}-id"
            conn.execute("DELETE FROM metadata WHERE key=?", (key,))
            if task != missing:
                # command.py:3615,3619: real summary and incremental chunk receipts.
                receipt = {"remaining": remaining, "operator_retry": 0, "updated_at": 1.0}
                if task == "chunk":
                    receipt.update(inflight_cursor="", inflight_actionable_seen=False,
                                   inflight_uncertain_count=0, inflight_cycle_started_at=0)
                conn.execute("INSERT INTO metadata VALUES(?,?)", (key, json.dumps(receipt)))
    conn.close()


def test_product_pass_repeats_until_both_complete_and_receipts_zero(client):
    adapter, _, run = client
    config = object()
    outcomes = iter([
        ("summaries=partial:2 chunks=complete:1", 1, True),
        ("summaries=complete:0 chunks=complete:0", 0, True),
        ("summaries=complete:0 chunks=complete:0", 0, False),
    ])

    def advance(db, supplied_config):
        assert db == str(adapter._db_path("u"))
        assert supplied_config is config
        outcome, summary, chunk = next(outcomes)
        seed_receipts(adapter, summary, chunk)
        return outcome

    run.side_effect = advance
    adapter._catch_up("u", config)
    assert run.call_count == 3
    report = json.loads((adapter._user_dir("u") / "embedding_catchup.json").read_text())
    assert report["status"] == "complete"
    assert report["receipts"]["summary"]["remaining"] == 0
    assert report["receipts"]["chunk"]["remaining"] is False
    assert all(r["operator_retry"] == 0 for r in report["receipts"].values())


@pytest.mark.parametrize("missing", ["summary", "chunk"])
def test_missing_receipt_aborts_even_with_complete_outcomes(client, missing):
    adapter, _, run = client
    seed_receipts(adapter, missing=missing)
    run.return_value = "summaries=complete:0 chunks=complete:0"
    with pytest.raises(SystemExit, match="incomplete after 3"):
        adapter._catch_up("u", object())
    assert run.call_count == 3
    assert json.loads((adapter._user_dir("u") / "embedding_catchup.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("outcome", ["error", "model_unavailable", "summaries=error", "summaries=complete:0"])
def test_pass_failure_is_not_silent(client, outcome):
    adapter, _, run = client
    seed_receipts(adapter)
    run.return_value = outcome
    with pytest.raises(SystemExit, match="catch-up failed"):
        adapter._catch_up("u", object())
    assert run.call_count == 1


def test_v028_skips_maintenance_and_records_not_applicable(client):
    adapter, tools, run = client
    del tools._lcm_recall_catchup_reason
    adapter._catch_up("u", object())
    run.assert_not_called()
    assert json.loads((adapter._user_dir("u") / "embedding_catchup.json").read_text()) == {
        "status": "not_applicable", "passes": 0, "reason": "no_receipt_gate",
    }


@pytest.mark.parametrize("payload", [
    {"degraded": True, "degraded_reason": "chunk vectors are unavailable", "hits": []},
    {"degraded_reason": "summary semantic index catching up", "hits": []},
    {"degraded": True, "hits": []},
    # cf558e999 tools.py:4384,4425,4478,4514,4558: five timeout paths.
    {"degraded": False, "timeout": True, "hits": []},
    {"degraded": False, "timeout": True, "provenance": {"coverage": {"fts": "full"}}, "hits": []},
    {"degraded": False, "timeout": True, "provenance": {"coverage": {"chunk": "none"}}, "hits": []},
    {"degraded": False, "timeout": True, "provenance": {"coverage": {"summary": "none"}}, "hits": []},
    {"degraded": False, "timeout": True, "provenance": {"coverage": {"summary": "full", "chunk": "none"}}, "hits": []},
    *[{"provenance": {"coverage": {arm: state}}, "hits": []}
      for arm in ("summary", "chunk") for state in ("none", "disabled")],
    *[{"degraded_reason": f"{arm} semantic index catching up (coverage not certified; catch-up receipt missing)", "hits": []}
      for arm in ("summary", "chunk")],
])
def test_search_counts_degraded_payload_and_aborts_before_delivery(client, payload):
    adapter, tools, _ = client
    adapter._db_path("u").touch()
    store, dag, vectors = Mock(), Mock(), Mock()
    adapter._open = Mock(return_value=(object(), store, dag, vectors))
    tools.lcm_recall = Mock(return_value=json.dumps(payload))
    with pytest.raises(SystemExit, match="1 degraded payload"):
        adapter.search("question", "u", 20)
    with pytest.raises(SystemExit, match="stage already aborted"):
        adapter.search("next question", "u", 20)
    store.get_batch.assert_not_called()
    assert store.close.call_count == dag.close.call_count == vectors.close.call_count == 1
    assert json.loads((adapter._user_dir("u") / "recall_status.json").read_text()) == {
        "payloads": 1, "degraded_payloads": 1,
    }
    assert tools.lcm_recall.call_count == 1
    adapter.delete_all("u")
    assert json.loads((adapter._db_root / "search_status.json").read_text()) == {
        "stage": "search", "payloads": 1, "degraded_payloads": 1,
    }


def test_clean_recall_keeps_single_call_and_full_delivery(client):
    adapter, tools, _ = client
    adapter._db_path("u").touch()
    store, dag, vectors = Mock(), Mock(), Mock()
    store.get_batch.return_value = {1: {"content": "full stored text"}}
    adapter._open = Mock(return_value=(object(), store, dag, vectors))
    tools.lcm_recall = Mock(return_value=json.dumps({"hits": [{"store_id": 1, "snippet": "cut"}]}))
    assert adapter.search("question", "u", 20) == "full stored text"
    tools.lcm_recall.assert_called_once()
    assert tools.lcm_recall.call_args.args[0] == {"query": "question", "limit": 20}
    assert json.loads((adapter._user_dir("u") / "recall_status.json").read_text())["degraded_payloads"] == 0


def test_add_runs_catchup_after_closing_ingest_handles(client, monkeypatch):
    adapter, _, _ = client
    adapter._embeddings = False
    adapter._summary = lambda rows: "summary"
    for name in ("chunking", "dag", "vector_store"):
        module = ModuleType(f"hermes_lcm.{name}")
        module.iter_message_chunks = Mock()
        module.SummaryNode = SimpleNamespace
        module.EmbeddingIdentity = Mock()
        monkeypatch.setitem(sys.modules, f"hermes_lcm.{name}", module)
    store, dag, vectors = Mock(), Mock(), Mock()
    store.append_batch.return_value = [1]
    config = object()
    adapter._open = Mock(return_value=(config, store, dag, vectors))

    def catch_up(user_id, supplied):
        assert user_id == "u" and supplied is config
        for handle in (store, dag, vectors):
            handle.close.assert_called_once()
        assert not adapter._sessions_path("u").exists()

    adapter._catch_up = Mock(side_effect=catch_up)
    adapter.add([{"role": "user", "content": "text"}], "u")
    adapter._catch_up.assert_called_once_with("u", config)
    assert adapter._sessions_path("u").exists()


def test_original_ids_survive_search_and_join_answer_judge(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ids = ["original-99_abs", "original-07"]
    rows = [{"question_id": qid} for qid in ids]
    write_question_id_map(rows, "lcmx", "repeat1")
    path = tmp_path / "results/lme/lcmx-repeat1/lcmx_lme_question_ids.json"
    mapping = json.loads(path.read_text())
    assert mapping == {"lme_exper_user_repeat1_0": ids[0], "lme_exper_user_repeat1_1": ids[1]}
    # These are the stable top-level keys in search, answer and judge outputs.
    for key, qid in mapping.items():
        row = {"question": "same question", "question_type": "temporal", "question_date": "date",
               "answer": "gold", "haystack_sessions": [], "answer_session_ids": [], "haystack_session_ids": [],
               "question_id": qid}
        result = build_search_result(row, user_id=key, status="success", context="text")
        assert result[key][0]["question_id"] == qid
    # A different repeat keeps its own mapping and the exact abstention suffix.
    write_question_id_map(rows, "lcmx", "repeat2")
    repeat = json.loads((tmp_path / "results/lme/lcmx-repeat2/lcmx_lme_question_ids.json").read_text())
    assert repeat == {"lme_exper_user_repeat2_0": ids[0], "lme_exper_user_repeat2_1": ids[1]}
    assert not (repeat.keys() & mapping.keys())


@pytest.mark.parametrize("ids", [["a", "a"], [""], [None]])
def test_ambiguous_id_mapping_rejected(tmp_path, monkeypatch, ids):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="question_id"):
        write_question_id_map([{"question_id": qid} for qid in ids], "lcmx", "r1")


def test_failed_catchup_refuses_readd_and_search_without_duplicate_writes(client, monkeypatch):
    adapter, _, run = client
    adapter._summary = lambda rows: "summary"
    adapter._chunk_policy = "conversational"
    adapter._dim = 1
    adapter._embedder.embed_documents = Mock(return_value=[[1.0]])
    for name in ("chunking", "dag", "vector_store"):
        module = ModuleType(f"hermes_lcm.{name}")
        module.iter_message_chunks = Mock(return_value=[])
        module.SummaryNode = SimpleNamespace
        module.EmbeddingIdentity = Mock()
        monkeypatch.setitem(sys.modules, f"hermes_lcm.{name}", module)
    store, dag, vectors = Mock(), Mock(), Mock()
    store.append_batch.return_value = [1]
    adapter._open = Mock(return_value=(object(), store, dag, vectors))
    run.return_value = "error"
    for attempt in range(2):
        with pytest.raises(SystemExit, match="catch-up failed" if attempt == 0 else "--clear / --restart-unit"):
            adapter.add([{"role": "user", "content": "text"}], "u", session_key="s1")
    with pytest.raises(SystemExit, match="--clear / --restart-unit"):
        adapter.search("question", "u", 20)
    store.append_batch.assert_called_once()
    dag.add_node.assert_called_once()
    vectors.record_embedding.assert_called_once()
    assert not adapter._sessions_path("u").exists()
    adapter.delete_all("u")
    adapter._user_dir("u").mkdir()
    seed_receipts(adapter)
    run.return_value = "summaries=complete:0 chunks=complete:0"
    adapter.add([{"role": "user", "content": "text"}], "u", session_key="s1")
    assert store.append_batch.call_count == 2  # Explicit rebuild is allowed.


def test_embeddings_disabled_skips_catchup_and_does_not_abort_search(client):
    adapter, tools, run = client
    adapter._embeddings = False
    adapter._catch_up("u", object())
    run.assert_not_called()
    assert json.loads((adapter._user_dir("u") / "embedding_catchup.json").read_text())["reason"] == "embeddings_disabled"
    adapter._open = Mock(return_value=(object(), Mock(), Mock(), Mock()))
    tools.lcm_recall = Mock(return_value=json.dumps({
        "hits": [], "degraded": True, "degraded_reason": "semantic retrieval is disabled",
        "provenance": {"coverage": {"summary": "disabled", "chunk": "disabled"}},
    }))
    assert adapter.search("question", "u", 20) == ""
    assert json.loads((adapter._db_root / "search_status.json").read_text())["degraded_payloads"] == 0


@pytest.mark.parametrize("outcome", ["lease_held", "model_preparing", "summaries=complete:0 chunks=lease_held"])
def test_transient_retries_sleep_then_complete_even_with_small_pass_cap(client, monkeypatch, outcome):
    adapter, _, run = client
    elapsed = [0]
    sleep = Mock(side_effect=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds))
    monkeypatch.setattr(adapter_module, "time", SimpleNamespace(monotonic=lambda: elapsed[0], sleep=sleep))
    run.side_effect = [outcome] * 4 + ["summaries=complete:0 chunks=complete:0"]
    adapter._catch_up("u", object())
    assert elapsed[0] == 20 and sleep.call_count == 4
    assert run.call_count == 5


@pytest.mark.parametrize("outcome", ["lease_held", "model_preparing"])
def test_transient_retries_abort_at_ten_minute_ttl(client, monkeypatch, outcome):
    adapter, _, run = client
    elapsed = [0]
    sleep = Mock(side_effect=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds))
    monkeypatch.setattr(adapter_module, "time", SimpleNamespace(monotonic=lambda: elapsed[0], sleep=sleep))
    run.return_value = outcome
    with pytest.raises(SystemExit, match="exceeded 600s"):
        adapter._catch_up("u", object())
    assert elapsed[0] == 600 and sleep.call_count == 120
    assert run.call_count == 121


def test_twenty_unchanged_zero_write_passes_abort_ignoring_receipt_timestamp(client, monkeypatch):
    adapter, _, run = client
    monkeypatch.delenv("LCMX_CATCHUP_MAX_PASSES")
    seed_receipts(adapter, chunk=True)

    def stalled(db, config):
        conn = sqlite3.connect(db)
        with conn:
            conn.execute("UPDATE metadata SET value=json_set(value, '$.updated_at', ?)", (run.call_count,))
        conn.close()
        return "summaries=complete:0 chunks=partial:0"

    run.side_effect = stalled
    with pytest.raises(SystemExit, match="stalled for 20"):
        adapter._catch_up("u", object())
    assert run.call_count == 20


def test_scan_frontier_progress_prevents_false_stall_and_default_cap_is_500(client, monkeypatch):
    adapter, _, run = client
    monkeypatch.delenv("LCMX_CATCHUP_MAX_PASSES")
    seed_receipts(adapter, chunk=True)
    run.return_value = "summaries=complete:0 chunks=partial:0"
    # Product _scan_frontier: (chunk cursor, three ordered metadata key/value rows).
    frontier = sys.modules["hermes_lcm.embedding_maintenance"]._scan_frontier
    frontier.side_effect = lambda db, command: ((1, run.call_count), (
        ("chunk_gapfill_scan:chunk-id", str(run.call_count)),
        ("chunk_policy_revision:chunk-id", "2"),
        ("chunk_policy_revision_through:chunk-id", str(run.call_count)),
    ))
    with pytest.raises(SystemExit, match="incomplete after 500"):
        adapter._catch_up("u", object())
    assert run.call_count == 500


@pytest.mark.parametrize("field,value", [("revision", "+prefix:stale"), ("operator_retry", 1)])
def test_stale_revision_and_operator_retry_cannot_certify_completion(client, field, value):
    adapter, _, run = client
    conn = sqlite3.connect(adapter._db_path("u"))
    with conn:
        if field == "revision":
            conn.execute("UPDATE lcm_embedding_profile SET revision=? WHERE task='chunk'", (value,))
        else:
            conn.execute("UPDATE metadata SET value=json_set(value, '$.operator_retry', ?)", (value,))
    conn.close()
    run.return_value = "summaries=complete:0 chunks=complete:0"
    with pytest.raises(SystemExit, match="incomplete after 3"):
        adapter._catch_up("u", object())


def test_vector_writes_reset_the_consecutive_stall_counter(client, monkeypatch):
    adapter, _, run = client
    monkeypatch.delenv("LCMX_CATCHUP_MAX_PASSES")
    seed_receipts(adapter, chunk=True)
    run.side_effect = lambda db, config: (
        "summaries=complete:1 chunks=partial:0" if run.call_count in {19, 38}
        else "summaries=complete:0 chunks=partial:0"
    )
    with pytest.raises(SystemExit, match="stalled for 20"):
        adapter._catch_up("u", object())
    assert run.call_count == 58


def test_full_text_mode_does_not_inherit_a_semantic_abort(client):
    adapter, tools, _ = client
    adapter._open = Mock(return_value=(object(), Mock(), Mock(), Mock()))
    tools.lcm_recall = Mock(return_value=json.dumps({"hits": [], "timeout": True}))
    with pytest.raises(SystemExit, match="degraded payload"):
        adapter.search("question", "u", 20)
    adapter._embeddings = False
    assert adapter.search("question", "u", 20) == ""
    assert tools.lcm_recall.call_count == 2
