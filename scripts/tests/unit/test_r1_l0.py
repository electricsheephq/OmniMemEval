"""Synthetic-only tests for the R1 measurement switches."""

import hashlib
import json
import random
from pathlib import Path
from unittest.mock import Mock, mock_open

import pandas as pd
import pytest
import tiktoken

import client_factory
from locomo import locomo_search
from longmemeval import lme_common
from personamem_v2 import pm_common, pm_search
from utils import evidence_delivery as delivery
from utils import r1_run, splits


@pytest.fixture(autouse=True)
def unset_switches(monkeypatch):
    for suffix in ("SPLIT", "EVIDENCE", "STORE_VERSION", "BUILD_PIN", "LOCAL_RNG"):
        monkeypatch.delenv(f"R1_{suffix}", raising=False)


@pytest.fixture
def encoding():
    # A tiny local tokenizer using tiktoken's actual prefix encode/decode API.
    return tiktoken.Encoding(
        name="synthetic",
        pat_str=r"(?s).",
        mergeable_ranks={bytes([i]): i for i in range(256)},
        special_tokens={},
    )


def scored_lme(encoding, context, evidence=("a fact",), **extra):
    row = dict(question_id="q-7", category="multi-session", search_context=context,
               answer_evidences=[f"user : {item}" for item in evidence], **extra)
    return delivery.score({"arbitrary-store-key": [row]}, encoding)


def test_frozen_splits_preserve_positions_and_sign_the_id_lists(monkeypatch, capsys):
    rows = pd.DataFrame({"question_id": [f"q-{i}" for i in range(1000)]})
    emit = Mock()
    monkeypatch.setattr(splits, "bind_run", Mock())
    monkeypatch.setattr(splits, "atomic_json_dump", emit)
    directory = Path("results/lme/synthetic")
    assert list(splits.select_indices(rows, "question_id", directory, "synthetic")) == list(range(1000))
    emit.assert_not_called()
    groups = {}
    for name in ("dev", "confirm"):
        monkeypatch.setenv("R1_SPLIT", name)
        groups[name] = splits.select_indices(rows, "question_id", directory, "synthetic")
        assert groups[name] == splits.select_indices(rows, "question_id", directory, "synthetic")
    assert not set(groups["dev"]) & set(groups["confirm"])
    assert sorted(groups["dev"] + groups["confirm"]) == list(rows.index)
    assert 350 < len(groups["dev"]) < 450
    stdout = capsys.readouterr().out
    for name, call in zip(groups, emit.call_args_list[:2], strict=True):
        manifest, path = call.args
        assert path.name == f"r1_split_{name}.json"
        ids = sorted(rows.loc[i, "question_id"] for i in groups[name])
        assert manifest == {
            "ids": ids, "rows": len(ids),
            "sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        }
        assert f"lme {name}: {len(ids)} rows" in stdout


def test_lme_id_map_preserves_original_store_positions(monkeypatch):
    emit = Mock()
    monkeypatch.setattr(lme_common, "atomic_json_dump", emit)
    monkeypatch.setattr(Path, "mkdir", Mock())
    monkeypatch.setenv("R1_STORE_VERSION", "new-store")
    lme_common.write_question_id_map([{"question_id": "q0"}, {"question_id": "q1"}],
                                     "lcmx", "result-label", [1])
    mapping, out = emit.call_args.args
    assert mapping == {"lme_exper_user_new-store_1": "q1"}
    assert out.parent == Path("results/lme/lcmx-result-label")


def test_lme_search_entry_default_golden():
    row = dict(question_id="q1", question="question", question_type="temporal",
               question_date="2024-01-01", answer="gold", answer_session_ids=["s1"],
               haystack_session_ids=["s1"],
               haystack_sessions=[[dict(role="user", content="a fact", has_answer=True)]])
    entry = lme_common.build_search_entry(row, context="[time] a fact", duration_ms=1.0, status="success")
    expected = ('{"question": "question", "category": "temporal", "date": "2024-01-01", "golden_answer": "gold", '
                '"answer_evidences": ["user : a fact"], "search_context": "[time] a fact", '
                '"search_duration_ms": 1.0, "status": "success", "question_id": "q1"}')
    assert json.dumps(entry) == expected


@pytest.mark.parametrize("context, label", [("[time] a  fact", "verbatim"), ("", "none")])
def test_controls_separate_positive_and_empty_context(encoding, context, label):
    scores = scored_lme(encoding, context)["q-7"][1]
    assert set(scores) == {(n, mode) for n in delivery.BUDGETS for mode in ("strict", "lenient")}
    assert all(items == [label] for items in scores.values())


def test_count_prefixes_and_multiple_evidence_turns(encoding):
    context = "[timestamp] a fact\n" + "x" * 4070 + "\n[timestamp] second fact"
    scores = scored_lme(encoding, context, ("a fact", "second fact"))["q-7"][1]
    assert scores[4096, "strict"] == ["verbatim", "none"]
    assert scores[8192, "strict"] == ["verbatim", "verbatim"]


def test_lenient_accepts_any_contiguous_span(encoding):
    middle = "x" * 300
    scores = scored_lme(encoding, middle, ("leading " + middle + " trailing",))["q-7"][1]
    assert scores[4096, "strict"] == ["none"]
    assert scores[4096, "lenient"] == ["verbatim"]


def test_summary_coverage_requires_the_corresponding_complete_delivered_line(encoding):
    hits = [
        {"position": 0, "kind": "raw", "store_id": "raw-7", "node_id": None, "source_store_ids": []},
        {"position": 1, "kind": "summary", "store_id": "sum-8", "node_id": "node-8",
         "source_store_ids": ["raw-8"]},
    ]
    context = "x" * 4090 + "\n[time] summarized evidence"
    scores = scored_lme(encoding, context, ("a fact", "another fact"),
                        delivered_hits=hits, evidence_source_store_ids=["raw-8", "missing"])["q-7"][1]
    assert scores[4096, "strict"] == ["none", "none"]
    assert scores[8192, "strict"] == ["summary", "none"]


def test_excludes_abstention_and_empty_locomo_gold(encoding):
    data = {
        "lme": [{"question_id": "q_abs", "answer_evidences": []}],
        "locomo": [{"question_id": "sample:0", "evidence": []}],
    }
    assert delivery.score(data, encoding) == {}


def test_pairing_uses_question_id_and_identifies_changed_delivery(encoding):
    left = scored_lme(encoding, "[time] a fact")
    replica = scored_lme(encoding, "[time] a fact")
    broken = scored_lme(encoding, "[time] unrelated")
    assert dict(delivery.pair(left, replica)) == {}
    assert dict(delivery.pair(left, broken)) == {"q-7": ("[time] a fact", "[time] unrelated")}


def test_same_store_version_refused_before_cache_lookup(monkeypatch):
    monkeypatch.setenv("R1_STORE_VERSION", "baseline")
    with pytest.raises(ValueError, match="differ from --version"):
        r1_run.bind_run(Path("synthetic"), "baseline")
    monkeypatch.setattr(Path, "exists", lambda path: path.name == "tmp")
    with pytest.raises(ValueError, match="Existing R1 checkpoints"):
        r1_run.bind_run(Path("synthetic"), "replica")


def test_build_pin_is_opt_in_and_writes_full_head_and_dirty_flag(monkeypatch):
    head = "a" * 40
    git = Mock(side_effect=[head + "\n", " M synthetic.py\n"])
    writer = mock_open()
    monkeypatch.setattr(r1_run.subprocess, "check_output", git)
    monkeypatch.setattr(Path, "open", writer)
    r1_run.pin_build("run-dir")
    git.assert_not_called()
    writer.assert_not_called()
    monkeypatch.setenv("R1_BUILD_PIN", "1")
    monkeypatch.setenv("LCMX_REPO", "synthetic-repo")
    r1_run.pin_build("run-dir")
    assert git.call_args_list[0].args[0][-2:] == ["rev-parse", "HEAD"]
    assert git.call_args_list[1].args[0][-2:] == ["status", "--porcelain"]
    writer().write.assert_called_once_with(f"lcmx_commit={head}\nlcmx_dirty=1\n")


def test_pm_options_golden_and_local_rng_isolation(monkeypatch):
    state = random.getstate()
    try:
        fixture = ("correct", '["one", "two", "three"]', 7)
        expected = ("(a) three\n(b) one\n(c) correct\n(d) two", "(c)")
        assert pm_search.build_options(*fixture) == expected
        monkeypatch.setenv("R1_LOCAL_RNG", "1")
        state_after_default = random.getstate()
        assert pm_search.build_options(*fixture) == expected
        pm_search.build_options(*fixture[:-1], 5)
        assert random.getstate() == state_after_default
    finally:
        random.setstate(state)


def test_pm_search_entry_golden_and_opt_in_gold_capture(monkeypatch, encoding):
    data = dict(result_key="pm_exper_user_7_r1", persona_id=2, row_idx=7,
                question="q", category="pref", all_options="(a) choice", topic="topic",
                golden_answer="(a)", context="[time] a fact", duration_ms=1.0,
                status="success", related_conversation_snippet='[{"role":"user","content":"a fact"}]')
    expected = ('{"key": "pm_exper_user_7_r1", "user_id": "pm_exper_user_7_r1", "question": "q", '
                '"category": "pref", "persona_id": 2, "row_idx": 7, "all_options": "(a) choice", '
                '"topic": "topic", "golden_answer": "(a)", "search_context": "[time] a fact", '
                '"search_duration_ms": 1.0, "status": "success"}')
    assert json.dumps(pm_common.build_search_entry(**data)) == expected
    monkeypatch.setenv("R1_EVIDENCE", "1")
    row = pm_common.build_search_entry(**data)
    assert row["related_conversation_snippet"] == data["related_conversation_snippet"]
    assert delivery.score({"group": [row]}, encoding)["7"][1][4096, "strict"] == ["verbatim"]


def test_locomo_golden_and_opt_in_gold_capture(monkeypatch, encoding):
    rows = pd.DataFrame([{
        "sample_id": "sample",
        "qa": [{"question": "q", "category": 1, "evidence": ["D1:1"]}],
        "conversation": {"speaker_a": "A", "speaker_b": "B",
                         "session_1": [{"dia_id": "D1:1", "speaker": "A", "text": "a fact"}]},
    }])
    monkeypatch.setattr(client_factory, "create_client", Mock())
    monkeypatch.setattr(locomo_search, "load_existing_results", lambda *args, **kwargs: ({}, False))
    monkeypatch.setattr(locomo_search, "search_query", lambda *args, **kwargs: ("[time] a fact", 1.0, None))
    monkeypatch.setattr(locomo_search, "atomic_json_dump", Mock())
    monkeypatch.setattr(locomo_search.os, "makedirs", Mock())
    monkeypatch.setattr(locomo_search, "track", lambda rows, **kwargs: rows)
    results, blocking = locomo_search.process_user(0, rows, "lcmx", "r1", num_workers=1)
    assert not blocking
    expected = '{"locomo_exp_user_0": [{"query": "q", "context": "[time] a fact", "duration_ms": 1.0, "status": "success"}]}'
    assert json.dumps(results) == expected
    monkeypatch.setenv("R1_EVIDENCE", "1")
    results, blocking = locomo_search.process_user(0, rows, "lcmx", "r1", num_workers=1)
    assert not blocking
    row = next(iter(results.values()))[0]
    assert row["evidence"] == ["D1:1"]
    assert row["evidence_turns"] == {"D1:1": "A: a fact"}
    assert delivery.score(results, encoding)["sample:0"][1][4096, "strict"] == ["verbatim"]
