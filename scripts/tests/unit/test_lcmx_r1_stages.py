"""Real LME stage wiring with fake recall/model calls; no weights or downloads."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import client_factory
import pandas as pd
import pytest
from client_factory.lcmx_client import LcmxClient
from longmemeval import lme_responses, lme_search, lme_streaming
from longmemeval.lme_common import user_id_for


@pytest.fixture
def dataframe():
    return pd.DataFrame([
        {"question_id": qid, "question": "identical question", "question_type": "temporal",
         "question_date": "2024-01-02", "answer": "gold", "haystack_sessions": [],
         "haystack_session_ids": [], "answer_session_ids": []}
        for qid in ("original-99_abs", "original-07")
    ])


def load_offline_judge(monkeypatch):
    # Keep actual judge stage/record construction; replace only resource/model dependencies.
    transformers = ModuleType("transformers")
    transformers.logging = SimpleNamespace(set_verbosity_error=lambda: None)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    metrics = ModuleType("utils.nlp_metrics")
    metrics.init_nlp = lambda: None
    metrics.calculate_nlp_metrics = lambda **kwargs: {}
    metrics.extract_label_json = lambda text: text
    metrics.LLMGrade = SimpleNamespace
    monkeypatch.setitem(sys.modules, "utils.nlp_metrics", metrics)
    path = Path(lme_responses.__file__).with_name("lme_eval.py")
    spec = importlib.util.spec_from_file_location("offline_lme_eval", path)
    judge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judge)
    return judge


def test_search_answer_judge_files_join_original_and_abs_ids_across_repeats(tmp_path, monkeypatch, dataframe):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(lme_search, "load_env", lambda: None)
    monkeypatch.setattr(lme_responses, "load_env", lambda: None)
    monkeypatch.setattr(lme_search, "load_lme_dataframe", lambda: dataframe)
    fake_recall = Mock(search=Mock(return_value="stored evidence"))
    monkeypatch.setattr(client_factory, "create_client", lambda frame: fake_recall)

    async def complete(**kwargs):
        content = "gold" if kwargs["model"] == "answer" else '{"label":"CORRECT"}'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    completions = AsyncMock(side_effect=complete)
    llm = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completions)))
    clients = ModuleType("utils.llm_client")
    clients.create_async_openai_client = lambda stage: (llm, "answer" if stage == "ANSWER" else "judge")
    monkeypatch.setitem(sys.modules, "utils.llm_client", clients)
    judge = load_offline_judge(monkeypatch)
    monkeypatch.setattr(judge, "load_env", lambda: None)

    mappings = []
    for version in ("repeat1", "repeat2"):
        lme_search.main("lcmx", version, num_workers=1)
        asyncio.run(lme_responses.main("lcmx", version, llm_workers=1))
        asyncio.run(judge.main("lcmx", version, llm_workers=1))
        folder = tmp_path / "results/lme" / f"lcmx-{version}"
        mapping = json.loads((folder / "lcmx_lme_question_ids.json").read_text())
        expected = {user_id_for(version, i): qid for i, qid in enumerate(dataframe["question_id"])}
        assert mapping == expected
        for suffix in ("search_results", "responses", "judged"):
            records = json.loads((folder / f"lcmx_lme_{suffix}.json").read_text())
            assert records.keys() == mapping.keys()
            assert {mapping[key] for key in records} == {"original-99_abs", "original-07"}
            if suffix != "search_results":
                assert all(record["user_id"] == key for key, record in records.items())
                assert all(record["answer"] == "gold" for record in records.values())
        mappings.append(mapping)
    assert not (mappings[0].keys() & mappings[1].keys())
    assert list(mappings[0].values()) == list(mappings[1].values())
    assert completions.await_count == 8
    assert fake_recall.search.call_count == 4


@pytest.mark.parametrize("workers", [1, 2])
def test_search_stage_skip_flag_cannot_accept_timeout_or_run_queued_recalls(tmp_path, monkeypatch, dataframe, workers):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(lme_search, "load_env", lambda: None)
    monkeypatch.setattr(lme_search, "load_lme_dataframe", lambda: dataframe)
    adapter = LcmxClient.__new__(LcmxClient)
    adapter._db_root = tmp_path / "stores"
    adapter._embeddings = True
    adapter._provider_name = "fastembed"
    adapter._embedder = SimpleNamespace(model_id="test-model")
    adapter._top_k_cap = 25
    adapter._scope_bias = None
    adapter._delivery = "full"
    adapter._open = Mock(return_value=(object(), Mock(), Mock(), Mock()))
    for idx in range(len(dataframe)):
        user = user_id_for("abort", idx)
        adapter._user_dir(user).mkdir(parents=True)
        adapter._db_path(user).touch()
    package = ModuleType("hermes_lcm")
    package.__path__ = []
    tools = ModuleType("hermes_lcm.tools")
    tools.lcm_recall = Mock(return_value=json.dumps({"hits": [], "degraded": False, "timeout": True}))
    package.tools = tools
    monkeypatch.setitem(sys.modules, "hermes_lcm", package)
    monkeypatch.setitem(sys.modules, "hermes_lcm.tools", tools)
    monkeypatch.setattr(client_factory, "create_client", lambda frame: adapter)
    with pytest.raises(SystemExit):
        lme_search.main("lcmx", "abort", num_workers=workers, skip_failed_search=True)
    tools.lcm_recall.assert_called_once()
    assert json.loads((adapter._db_root / "search_status.json").read_text()) == {
        "stage": "search", "payloads": 1, "degraded_payloads": 1,
    }


def test_streaming_skip_flag_cannot_accept_fatal_catchup(tmp_path, monkeypatch, dataframe):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["lme_streaming", "--lib", "lcmx", "--skip-failed-streaming"])
    monkeypatch.setattr(lme_streaming, "load_env", lambda: None)
    monkeypatch.setattr(lme_streaming, "load_lme_dataframe", lambda: dataframe)
    monkeypatch.setattr(lme_streaming, "configure_single_user_streaming", lambda frame: None)
    monkeypatch.setattr(lme_streaming, "resolve_max_batch_chars", lambda frame: 0)
    adapter = LcmxClient.__new__(LcmxClient)
    adapter._db_root = tmp_path / "stores"
    adapter._user_dir("u").mkdir(parents=True)
    (adapter._user_dir("u") / "embedding_catchup.json").write_text('{"status":"failed"}')
    # Run the real poisoned-store add guard from inside the real streaming exception handler.
    def ingest(*args, **kwargs):
        adapter._require_healthy_store("u")
    monkeypatch.setattr(lme_streaming, "process_fresh_conversation", ingest)
    skipped = Mock()
    monkeypatch.setattr(lme_streaming, "mark_streaming_failure_skipped", skipped)
    with pytest.raises(SystemExit, match="--clear / --restart-unit"):
        lme_streaming.main()
    skipped.assert_not_called()


def test_example_env_preserves_job_cache_values(tmp_path, monkeypatch):
    from utils.env import load_env

    example = Path(__file__).resolve().parents[3] / "env_examples/.env.lcmx"
    cache_names = ("FASTEMBED_CACHE_PATH", "LCM_LONGMEMEVAL_FASTEMBED_CACHE", "TIKTOKEN_CACHE_DIR")
    # Load only the relevant example lines, avoiding unrelated environment settings.
    env_file = tmp_path / "cache-example.env"
    env_file.write_text("\n".join(line for line in example.read_text().splitlines()
                                  if any(name in line for name in cache_names)))
    monkeypatch.setenv("OMNIMEMEVAL_ENV_FILE", str(env_file))
    for name in cache_names:
        monkeypatch.setenv(name, "/job/copied/cache")
    load_env()
    import os

    assert all(os.environ[name] == "/job/copied/cache" for name in cache_names)
