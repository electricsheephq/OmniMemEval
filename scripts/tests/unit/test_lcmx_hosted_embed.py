"""Hosted (openai-compatible) embedder: wiring, retry, product-owned identity with the privacy filter off, chunk
parity, catch-up gate, startup refusals, typed config, plus a golden for the default fastembed path.

Real-tree tests use LCM-X (needs LCMX_REPO) with a fake HTTP transport: no network, a fake key. Trees with the
catch-up receipt gate (ae042245) run ingest and recall end to end; trees without it (v0.28.0 GA) must refuse
after registering the product's profiles. Refusal, gate and golden tests fake LCM-X and run without a tree.
"""
import json
import os
import sqlite3
import sys
import threading
import urllib.error
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from client_factory import lcmx_client  # noqa: E402

REPO = os.environ.get("LCMX_REPO", "")
needs_repo = pytest.mark.skipif(not REPO or not Path(REPO).is_dir(), reason="LCMX_REPO (an LCM-X tree) not set")
FAKE_KEY = "sk-fake-hosted-test-0000"
URL = "https://embed.example.invalid/compatible-mode/v1"
MODEL = "text-embedding-v4"


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch, tmp_path):
    monkeypatch.setattr(lcmx_client, "_ABORT", threading.Event())
    monkeypatch.setattr(lcmx_client, "_ABORT_MESSAGE", None)
    monkeypatch.setattr(lcmx_client, "_EMBEDDER", None, raising=False)  # the pre-round-4 single cache
    monkeypatch.setattr(lcmx_client, "_EMBEDDERS", {}, raising=False)
    monkeypatch.setattr(lcmx_client, "_CONFIG_SET_LOGGED", set())
    for name in ("LCMX_CONFIG", "LCMX_CONFIG_SET", "LCMX_EMBED_CACHE", "LCMX_ADAPTER_VERSION", "LCMX_DELIVERY_LOG",
                 "LCMX_EMBED_PROVIDER", "LCMX_EMBED_MODEL", "LCMX_EMBED_BASE_URL", "LCMX_EMBED_BATCH_SIZE",
                 "LCMX_EMBED_CHUNK_PARITY", "LCMX_CHUNK_POLICY", "LCM_EMBEDDING_API_KEY", "SILICONFLOW_API_KEY",
                 "LCM_EMBEDDING_BACKFILL_BATCH_SIZE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LCMX_DB_ROOT", str(tmp_path / "db"))
    monkeypatch.setenv("LCMX_CATCHUP_MAX_PASSES", "20")


def _hosted_env(monkeypatch, provider="openai-compatible", **extra):
    values = {"LCMX_EMBED_PROVIDER": provider, "LCMX_EMBED_MODEL": MODEL, "LCMX_EMBED_BASE_URL": URL,
              "LCM_EMBEDDING_API_KEY": FAKE_KEY, **extra}
    for name, value in values.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


# -- startup refusals (no LCM-X tree needed: they happen before the import) ---------------------------------

@pytest.mark.parametrize("missing", ["LCMX_EMBED_BASE_URL", "LCM_EMBEDDING_API_KEY"])
def test_hosted_startup_refuses_missing_url_or_key_without_key_text(monkeypatch, tmp_path, capsys, missing):
    monkeypatch.setenv("LCMX_REPO", str(tmp_path))
    _hosted_env(monkeypatch, **{missing: None})
    with pytest.raises(ValueError, match=missing) as exc:
        lcmx_client.LcmxClient()
    out = capsys.readouterr()
    assert FAKE_KEY not in str(exc.value) + out.out + out.err


@pytest.mark.parametrize("extra,match", [
    ({"LCMX_EMBED_CACHE": "/x/cache.sqlite"}, "LCMX_EMBED_CACHE is not supported"),
    ({"LCMX_CHUNK_POLICY": "full"}, "LCMX_CHUNK_POLICY=full is not supported"),  # A9
    ({"LCMX_CONFIG_SET": "embedding_storage_dtype=int8"}, "embedding_storage_dtype"),  # A2
    ({"LCMX_CONFIG_SET": "recall_extra_entries=2,embedding_query_prefix=q: "}, "embedding_query_prefix"),
    ({"LCMX_CONFIG_SET": "embedding_document_prefix=d: "}, "embedding_document_prefix"),
    ({"LCMX_CONFIG_SET": "embedding_privacy_enabled=true"}, "embedding_privacy_enabled"),
])
def test_hosted_startup_refusals(monkeypatch, tmp_path, capsys, extra, match):
    monkeypatch.setenv("LCMX_REPO", str(tmp_path))
    _hosted_env(monkeypatch, **extra)
    with pytest.raises(ValueError, match=match) as exc:
        lcmx_client.LcmxClient()
    out = capsys.readouterr()
    assert FAKE_KEY not in str(exc.value) + out.out + out.err


@pytest.mark.parametrize("key", [FAKE_KEY + "\nX-Leak: 1", "sk fake", "\tsk-fake", "sk-fake\x1b[0m", "sk-fake\x7f",
                                 "sk-fake\u2028x", FAKE_KEY + "\n", "sk-fake\u20ac", "sk-fake\u0100", "sk-fake\u00e9"])
def test_hosted_startup_refuses_key_outside_printable_ascii_without_echoing_it(monkeypatch, tmp_path, capsys, key):
    """A7/P7: only 0x21-0x7E. A newline forges a header line; a non-Latin-1 character (U+0100, U+20AC) makes the
    HTTP layer raise an error whose repr holds the whole bearer header."""
    monkeypatch.setenv("LCMX_REPO", str(tmp_path))
    _hosted_env(monkeypatch, LCM_EMBEDDING_API_KEY=key)
    with pytest.raises(ValueError, match="LCM_EMBEDDING_API_KEY must be printable ASCII") as exc:
        lcmx_client.LcmxClient()
    out = capsys.readouterr()
    assert "sk" not in str(exc.value).replace("LCM_EMBEDDING_API_KEY", "") + out.out + out.err


def test_hosted_startup_accepts_a_printable_ascii_key(monkeypatch, tmp_path):
    monkeypatch.setenv("LCMX_REPO", str(tmp_path))
    monkeypatch.setitem(sys.modules, "benchmarking.longmemeval", None)  # stop right after the startup checks
    _hosted_env(monkeypatch, LCM_EMBEDDING_API_KEY="sk-" + "".join(chr(c) for c in range(0x21, 0x7F)))
    with pytest.raises(ImportError):
        lcmx_client.LcmxClient()


# -- typed config (point 4) -----------------------------------------------------------------------------------

def test_config_set_bool_or_none_field_gets_a_real_bool(monkeypatch, tmp_path):
    from dataclasses import dataclass

    @dataclass
    class Config:  # the shape of LCM-X's `embedding_privacy_enabled: bool | None = None`
        database_path: str = ""
        embeddings_enabled: bool = True
        embedding_privacy_enabled: "bool | None" = None
        recall_label: "str | None" = None

    module = ModuleType("hermes_lcm.config")
    module.LCMConfig = Config
    package = ModuleType("hermes_lcm")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "hermes_lcm", package)
    monkeypatch.setitem(sys.modules, "hermes_lcm.config", module)
    monkeypatch.setenv("LCMX_CONFIG_SET", "embedding_privacy_enabled=false,recall_label=false")
    c = lcmx_client.LcmxClient.__new__(lcmx_client.LcmxClient)
    c.__dict__.update(_db_root=tmp_path, _embeddings=False)
    config = c._config("u")
    assert config.embedding_privacy_enabled is False
    assert config.recall_label == "false"  # a str | None field keeps the string


def test_hosted_privacy_off_is_applied_after_every_config_override(monkeypatch, tmp_path):
    """Round 5 item 2: an LCMX_CONFIG_SET changed after construction cannot turn the filter back on."""
    from dataclasses import dataclass

    @dataclass
    class Config:
        database_path: str = ""
        embeddings_enabled: bool = True
        embedding_provider: str = ""
        embedding_model: str = ""
        embedding_base_url: str = ""
        embedding_max_batch_items: int = 1000
        embedding_privacy_enabled: "bool | None" = None

    module = ModuleType("hermes_lcm.config")
    module.LCMConfig = Config
    package = ModuleType("hermes_lcm")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "hermes_lcm", package)
    monkeypatch.setitem(sys.modules, "hermes_lcm.config", module)
    c = lcmx_client.LcmxClient.__new__(lcmx_client.LcmxClient)
    c.__dict__.update(_db_root=tmp_path, _embeddings=True, _provider_name="openai-compatible", _model=MODEL,
                      _base_url=URL, _batch_size=10)
    for value in ("true", "1", "on"):
        monkeypatch.setenv("LCMX_CONFIG_SET", f"embedding_privacy_enabled={value}")
        assert c._config("u").embedding_privacy_enabled is False


# -- hosted provider on the real LCM-X class, fake transport ------------------------------------------------

class FakeHttp:
    """Scripted embeddings endpoint: ``script`` holds statuses (or "net") for the next requests, then 200s.
    A 200 returns one vector [n, 1.0] per input, n parsed from "doc-<n>" (0 for other text)."""

    def __init__(self):
        self.script, self.requests = [], []

    def __call__(self, *, url, payload, headers, timeout):
        from hermes_lcm.embedding_provider import HttpResponse

        self.requests.append({"url": url, "payload": payload, "headers": dict(headers)})
        step = self.script.pop(0) if self.script else 200
        if step == "net":
            raise urllib.error.URLError("connection refused")
        if step != 200:
            return HttpResponse(status=step, headers={}, body=b'{"error": {"message": "x"}}')
        data = [{"index": i, "embedding": [float(t.split("-")[1]) if t.startswith("doc-") else 0.0, 1.0]}
                for i, t in enumerate(payload["input"])]
        return HttpResponse(status=200, headers={}, body=json.dumps({"data": data}).encode())


@pytest.fixture
def http(monkeypatch):
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from benchmarking.longmemeval import _ensure_hermes_lcm_package

    _ensure_hermes_lcm_package()
    import hermes_lcm.embedding_provider as ep

    fake = FakeHttp()
    monkeypatch.setattr(ep, "_default_http_transport", fake)
    sleeps = []
    monkeypatch.setattr(lcmx_client, "_sleep", sleeps.append, raising=False)  # absent before the change
    fake.sleeps = sleeps
    return fake


@needs_repo
@pytest.mark.parametrize("spelling", ["openai-compatible", "openai", "OpenAI"])
def test_hosted_wiring_model_url_key_reach_provider_and_identity(monkeypatch, http, spelling):
    _hosted_env(monkeypatch, provider=spelling)
    c = lcmx_client.LcmxClient()
    from hermes_lcm.embedding_provider import (
        OpenAICompatibleProvider,
        probe_provider_availability,
    )

    assert isinstance(c._embedder._provider, OpenAICompatibleProvider)
    [warmup] = http.requests  # one startup warmup through the adapter's (query) provider
    assert warmup["url"] == URL + "/embeddings"
    assert warmup["payload"] == {"model": MODEL, "input": ["warmup"]}
    assert warmup["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
    assert (c._provider_name, c._model, c._dim) == ("openai-compatible", MODEL, 2)
    config = c._config("u")  # store config: what LCM-X's own providers and probe read
    assert (config.embedding_provider, config.embedding_model, config.embedding_base_url) == (
        "openai-compatible", MODEL, URL)
    assert type(config.embedding_max_batch_items) is int and config.embedding_max_batch_items == 10
    assert probe_provider_availability(config)["available"] is True
    assert config.embedding_privacy_enabled is False  # owner: privacy filter off on the hosted path (typed bool)
    from hermes_lcm.ingest_policy import embedding_privacy_revision

    assert embedding_privacy_revision(config) == "privacy:off"


@needs_repo
@pytest.mark.parametrize("script,sleeps", [([429], [1.0]), (["net"], [1.0]), ([503, 500], [1.0, 4.0])])
def test_hosted_retry_recovers_on_429_5xx_and_network(monkeypatch, http, script, sleeps):
    _hosted_env(monkeypatch)
    c = lcmx_client.LcmxClient()
    http.requests.clear()
    http.script[:] = script
    assert c._embedder.embed_query("doc-7") == [7.0, 1.0]
    assert len(http.requests) == len(script) + 1 and http.sleeps == sleeps


@needs_repo
@pytest.mark.parametrize("script,attempts,sleeps,match", [
    ([400], 1, [], r"failed \(400\)"),
    ([401], 1, [], r"failed \(401\)"),
    ([503, 503, 503], 3, [1.0, 4.0], r"failed \(503\)"),
    (["net", "net", "net"], 3, [1.0, 4.0], "network error"),
])
def test_hosted_retry_never_on_other_4xx_and_fails_as_today(monkeypatch, http, script, attempts, sleeps, match):
    _hosted_env(monkeypatch)
    c = lcmx_client.LcmxClient()
    from hermes_lcm.embedding_provider import EmbeddingProviderError

    http.requests.clear()
    http.script[:] = script
    with pytest.raises(EmbeddingProviderError, match=match):
        c._embedder.embed_query("doc-1")
    assert len(http.requests) == attempts and http.sleeps == sleeps


SESSION = [{"role": "user" if i % 2 == 0 else "assistant", "chat_time": "2023-05-01T09:00:00Z",
            "content": f"Message {i}: I adopted a beagle named Biscuit and we walk {i} miles in the park near the "
                       "river every single morning before work, rain or shine."} for i in range(14)]
# A second session: a turn under the conversational policy's 40-token floor and one long multi-chunk turn.
SESSION2 = [{"role": "user", "chat_time": "2023-05-08T18:30:00Z", "content": "ok thanks"},
            {"role": "assistant", "chat_time": "2023-05-08T18:31:00Z",
             "content": " ".join(f"Biscuit learned trick number {i} at the training class on Tuesday." for i in range(300))},
            {"role": "user", "chat_time": "2023-05-08T18:32:00Z",
             "content": "Biscuit turned two today and we celebrated with a small cake shaped like a bone at the park."}]


def _profiles(db):
    conn = sqlite3.connect(db)
    try:
        return sorted(conn.execute("SELECT task, provider, model_name, revision, dim, identity_hash "
                                   "FROM lcm_embedding_profile WHERE active=1").fetchall())
    finally:
        conn.close()


@needs_repo
@pytest.mark.parametrize("batch", [None, "4"])
def test_hosted_ingest_and_recall_use_product_identity_and_chunk_parity(monkeypatch, http, tmp_path, batch):
    """Labels and privacy from LCM-X: the store's profiles equal those LCM-X's own warmup registers on a fresh
    store. On a receipt-gate tree, over two sessions: the texts sent are exactly the local arm's chunk units plus
    each session summary, no request exceeds the batch size, recall is not degraded. On a tree without the gate:
    refusal after registration."""
    _hosted_env(monkeypatch, LCMX_EMBED_BATCH_SIZE=batch)
    c = lcmx_client.LcmxClient()
    import hermes_lcm.tools as tools
    from hermes_lcm import command
    from hermes_lcm.chunking import iter_message_chunks

    http.requests.clear()
    gate = hasattr(tools, "_lcm_recall_catchup_reason")
    if gate:
        c.add(SESSION, "u", session_key="s1")
        c.add(SESSION2, "u", session_key="s2")
    else:
        with pytest.raises(SystemExit, match="no_receipt_gate"):
            c.add(SESSION, "u", session_key="s1")
    # The product's own warmup on a fresh store yields the same identities, privacy revision included.
    fresh = tmp_path / "fresh" / "lcm.db"
    fresh.parent.mkdir()
    from hermes_lcm.store import MessageStore
    from hermes_lcm.vector_store import VectorStore

    MessageStore(str(fresh), ingest_protection_config=c._config("u")).close()
    VectorStore(str(fresh), config=c._config("u")).close()
    sent = [t for r in http.requests for t in r["payload"]["input"] if t != "warmup"]
    assert not isinstance(command._embedding_register_profiles(c._config("u"), str(fresh), allow_download=False), str)
    stored = _profiles(c._db_path("u"))
    assert stored == _profiles(fresh) and {row[0] for row in stored} == {"summary", "chunk"}
    assert {row[3] for row in stored} == {"privacy:off"}  # the product's own label for the filter off
    if not gate:
        return
    report = json.loads((c._user_dir("u") / "embedding_catchup.json").read_text())
    assert report["status"] == "complete" and report["chunk_backfill"] == "confirm_raw_text"
    assert report["externalized_payload_rows_store"] == 0  # nothing externalized in these sessions
    expected = []
    for session in (SESSION, SESSION2):
        rows = [{"store_id": i + 1, "role": m["role"], "content": m["content"]} for i, m in enumerate(session)]
        expected += [chunk.text for chunk in iter_message_chunks(rows, policy="conversational")]
        expected.append(c._summary([{"role": m["role"], "content": m["content"]} for m in session]))
    assert sorted(sent) == sorted(expected) and len(expected) > 14 + 2 + 2  # long turn split, short turn skipped
    conn = sqlite3.connect(c._db_path("u"))
    assert conn.execute("SELECT count(*) FROM lcm_chunk_meta").fetchone()[0] == len(expected) - 2
    assert conn.execute("SELECT count(*) FROM lcm_embedding_meta").fetchone()[0] == 2
    conn.close()
    assert max(len(r["payload"]["input"]) for r in http.requests) == int(batch or 10)
    assert "Biscuit" in c.search("what is my dog's name", "u", 5)
    assert json.loads((c._user_dir("u") / "recall_status.json").read_text())["degraded_payloads"] == 0


@needs_repo
@pytest.mark.parametrize("script,where", [([429], "registration"), ([200, 429], "summary pass"),
                                          ([200, 200, 429], "chunk backfill")])
def test_hosted_one_429_in_ingest_recovers_end_to_end(monkeypatch, http, script, where):
    """On a receipt-gate tree, one 429 at registration, in the summary pass or in the chunk backfill is retried
    through LCM-X's own calls; ingest completes and recall is not degraded."""
    _hosted_env(monkeypatch)
    c = lcmx_client.LcmxClient()
    import hermes_lcm.tools as tools

    if not hasattr(tools, "_lcm_recall_catchup_reason"):
        pytest.skip("tree without the catch-up receipt gate refuses the hosted path")
    http.requests.clear()
    http.script[:] = script
    c.add(SESSION, "u", session_key="s1")
    assert json.loads((c._user_dir("u") / "embedding_catchup.json").read_text())["status"] == "complete"
    assert http.sleeps == [1.0 if where == "registration" else 4.0]
    assert "Biscuit" in c.search("what is my dog's name", "u", 5)


F1_BLOB = __import__("base64").b64encode(__import__("hashlib").sha512(b"verify-1").digest() * 3).decode()
F1_TEXT = ("Session 0: my sister Ana moved to Lisbon and works at a bakery near the river.\nHere is the config blob you "
           "asked for:\n" + F1_BLOB[:64] + "\n" + F1_BLOB[64:128] + "\nthanks")


@needs_repo
def test_reviewer_f1_base64_probe_embeds_with_the_filter_off(monkeypatch, http):
    """Item 1: the reviewers' two 64-character base64 lines embed verbatim and catch-up completes. Item 3: the
    retired LCMX_EMBED_CHUNK_PARITY=0 no longer turns chunks off."""
    _hosted_env(monkeypatch, LCMX_EMBED_CHUNK_PARITY="0")
    c = lcmx_client.LcmxClient()
    import hermes_lcm.tools as tools

    if not hasattr(tools, "_lcm_recall_catchup_reason"):
        pytest.skip("tree without the catch-up receipt gate refuses the hosted path")
    http.requests.clear()
    c.add([{"role": "user", "content": F1_TEXT, "chat_time": "2023-05-01T09:00:00Z"},
           {"role": "assistant", "content": "Noted.", "chat_time": "2023-05-01T09:01:00Z"}], "u1", session_id="s0")
    report = json.loads((c._user_dir("u1") / "embedding_catchup.json").read_text())
    assert report["status"] == "complete" and report["outcome"] == "summaries=complete:1 chunks=complete:2"
    sent = " ".join(t for r in http.requests for t in r["payload"]["input"])
    assert F1_BLOB[:64] in sent and F1_BLOB[64:128] in sent


BLOB = __import__("base64").b64encode(__import__("hashlib").sha512(b"x").digest() * 200).decode()
BIG = " ".join(f"Line {i}: Ana described the bakery recipe step {i} with flour and a slow proof." for i in range(1500))


@needs_repo
def test_externalized_payload_rows_are_counted_read_only_from_the_store(monkeypatch, http, capsys):
    """A3, round 5: one read-only count per store of message rows holding LCM-X's externalized-payload placeholder.
    A base64 blob row (ingest placeholder, alone or inside text) and a very large turn (payload placeholder)
    count; a lone surrogate, plain long text and CRLF count 0; rereads are idempotent; two clients on one store
    (as two processes would) do not double count; one WARNING line."""
    _hosted_env(monkeypatch)
    c = lcmx_client.LcmxClient()
    import hermes_lcm.tools as tools

    if not hasattr(tools, "_lcm_recall_catchup_reason"):
        pytest.skip("tree without the catch-up receipt gate refuses the hosted path")

    def add(client, sid, text):
        client.add([{"role": "user", "content": text, "chat_time": "2023-05-02T09:00:00Z"},
                    {"role": "assistant", "content": "Noted, thanks a lot.", "chat_time": "2023-05-02T09:01:00Z"}],
                   "u", session_id=sid)
        return json.loads((client._user_dir("u") / "embedding_catchup.json").read_text())

    for sid, text in (("surrogate", "a lone \ud800 surrogate in a short turn"), ("crlf", "line one\r\nline two\r\n"),
                      ("plain", " ".join(["Ana bakes sourdough bread every Sunday morning."] * 120))):
        report = add(c, sid, text)
        assert report["status"] == "complete" and report["externalized_payload_rows_store"] == 0, sid
    assert add(c, "blob", BLOB)["externalized_payload_rows_store"] == 1
    assert add(c, "inline", "Here is the file:\n" + BLOB)["externalized_payload_rows_store"] == 2
    assert add(c, "big", BIG)["externalized_payload_rows_store"] == 3
    assert c._externalized_payload_rows("u") == c._externalized_payload_rows("u") == 3  # read-only, idempotent
    other = lcmx_client.LcmxClient()  # a second client on the same store (no shared counter state)
    monkeypatch.setattr(lcmx_client, "_CONFIG_SET_LOGGED", set())
    assert add(other, "plain2", "Ana also bakes rye bread on Saturdays.")["externalized_payload_rows_store"] == 3
    assert not (c._user_dir("u") / "ingest_rewritten_rows.json").exists()
    assert capsys.readouterr().err.count("externalized payload rows") == 2  # one per client process state


@needs_repo
def test_a_default_client_never_reuses_a_cached_hosted_provider(monkeypatch, http):
    """A1: hosted then default in one process; the default client gets its own provider."""
    _hosted_env(monkeypatch)
    hosted = lcmx_client.LcmxClient()
    import benchmarking.longmemeval as lme

    fake = FakeProvider()
    monkeypatch.setattr(lme, "resolve_harness_provider", lambda *a, **k: fake)
    for name in ("LCMX_EMBED_PROVIDER", "LCMX_EMBED_MODEL", "LCMX_EMBED_BASE_URL"):
        monkeypatch.delenv(name)
    default = lcmx_client.LcmxClient()
    assert default._embedder._provider is fake and hosted._embedder._provider is not fake
    assert (default._provider_name, default._base_url) == ("fastembed", None)
    _hosted_env(monkeypatch)
    assert lcmx_client.LcmxClient()._embedder is hosted._embedder  # same (provider, model, base_url): shared


# -- catch-up gate with fake LCM-X modules (point 2) -----------------------------------------------------------

@pytest.fixture
def gate(tmp_path, monkeypatch):
    """A hosted client whose catch-up runs against faked LCM-X maintenance, command and provider modules."""
    c = lcmx_client.LcmxClient.__new__(lcmx_client.LcmxClient)
    c.__dict__.update(_db_root=tmp_path, _embeddings=True, _provider_name="openai-compatible", _model=MODEL,
                      _embedder=SimpleNamespace(model_id=MODEL), _base_url=URL,
                      _chunk_policy="conversational", _batch_size=10)
    c._user_dir("u").mkdir()
    mods = {name: ModuleType(name) for name in ("hermes_lcm", "hermes_lcm.tools", "hermes_lcm.command",
                                                 "hermes_lcm.embedding_maintenance", "hermes_lcm.embedding_provider",
                                                 "hermes_lcm.externalize", "hermes_lcm.ingest_content")}
    mods["hermes_lcm.externalize"]._EXTERNALIZED_REF_RE = __import__("re").compile(r"\[Externalized payload:.*?;\s*ref=")
    mods["hermes_lcm.ingest_content"]._INGEST_PLACEHOLDER_RE = __import__("re").compile(r"\[Externalized LCM ingest")
    mods["hermes_lcm"].__path__ = []
    mods["hermes_lcm.tools"]._lcm_recall_catchup_reason = Mock()
    maintenance, command = mods["hermes_lcm.embedding_maintenance"], mods["hermes_lcm.command"]
    maintenance.run_incremental_embedding_pass = Mock(return_value="summaries=complete:0")
    maintenance._scan_frontier = Mock(return_value=None)
    command._embedding_read_connection = Mock()
    command._embedding_current_profile = command._chunk_current_profile = Mock(return_value=object())
    command._embedding_register_profiles = Mock()
    command._embedding_backfill_batch_size = lambda cap: cap
    command._chunk_backfill_run = Mock(return_value=dict(OK, embedded=3))
    command._embedding_backfill_summary_run = Mock(return_value=dict(OK, embedded=1))
    mods["hermes_lcm.embedding_provider"]._automatic_chunks_allowed = Mock(return_value=False)
    for name, module in mods.items():
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            setattr(mods["hermes_lcm"], name.split(".")[1], module)
    monkeypatch.setenv("LCMX_CATCHUP_MAX_PASSES", "3")
    sleeps = []
    monkeypatch.setattr(lcmx_client, "_sleep", sleeps.append, raising=False)
    seed(c)
    return SimpleNamespace(client=c, command=command, maintenance=maintenance, sleeps=sleeps,
                           chunks_allowed=mods["hermes_lcm.embedding_provider"]._automatic_chunks_allowed)


OK = {"embedded": 0, "remaining": 0, "failed": [], "error": None, "stop_reason": None, "privacy_withheld": 0}
FAILED = dict(OK, failed=[("1:0", "provider_error:OpenAI-compatible embedding request failed (429)")])
CONFIG = SimpleNamespace(embedding_max_batch_items=10)


def _drop_chunk_receipt(client):
    conn = sqlite3.connect(client._db_path("u"))
    with conn:
        conn.execute("DELETE FROM metadata WHERE key LIKE 'embedding_catchup:chunk:%'")
    conn.close()


def seed(client, summary_retry=0):
    conn = sqlite3.connect(client._db_path("u"))
    with conn:
        conn.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS messages(content TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS lcm_embedding_profile(identity_hash TEXT PRIMARY KEY, provider TEXT, "
                     "model_name TEXT, revision TEXT NOT NULL DEFAULT '', task TEXT, registered_at TEXT, "
                     "active INTEGER DEFAULT 1, archived_at TEXT NULL)")
        for task in ("summary", "chunk"):
            conn.execute("INSERT OR REPLACE INTO lcm_embedding_profile(identity_hash,task,provider,model_name,revision,"
                         "registered_at) VALUES(?,?,?,?,?,?)", (task + "-id", task, "openai-compatible", MODEL,
                                                                "privacy:v3:abc", "2026-11-19T00:00:00"))
            retry = summary_retry if task == "summary" else 0
            conn.execute("INSERT OR REPLACE INTO metadata VALUES(?,?)", (
                    f"embedding_catchup:{task}:{task}-id", json.dumps({"remaining": 0, "operator_retry": retry})))
    conn.close()


def _report(client):
    return json.loads((client._user_dir("u") / "embedding_catchup.json").read_text())


def test_parity_runs_product_chunk_backfill_with_opt_in_and_completes(gate):
    config = SimpleNamespace(embedding_max_batch_items=10)
    gate.client._catch_up("u", config)
    report = _report(gate.client)
    assert report["status"] == "complete" and report["outcome"] == "summaries=complete:0 chunks=complete:3"
    assert report["chunk_backfill"] == "confirm_raw_text"
    gate.command._chunk_backfill_run.assert_called_once_with(
        config, str(gate.client._db_path("u")), apply=True, limit=10, retry_uncertain=False,
        policy="conversational", confirm_raw_text=True, incremental=True)
    gate.command._embedding_register_profiles.assert_not_called()  # profiles already registered


def test_hosted_registers_profiles_through_product_warmup_once_missing(gate):
    gate.command._chunk_current_profile = Mock(return_value=None)
    gate.command._embedding_register_profiles.return_value = {"dim": 2}
    gate.client._catch_up("u", SimpleNamespace(embedding_max_batch_items=10))
    gate.command._embedding_register_profiles.assert_called_once()
    gate.command._embedding_register_profiles.return_value = "LCM embedding warmup\nstatus: error\nerror: boom"
    with pytest.raises(SystemExit, match="warmup failed: error: boom"):
        gate.client._catch_up("u", SimpleNamespace(embedding_max_batch_items=10))


@pytest.mark.parametrize("result", [
    "LCM chunk backfill\nstatus: refused\nerror: no chunk embedding profile",
    FAILED,
    {"embedded": 0, "remaining": 0, "failed": [], "error": "provider down", "stop_reason": None},
    {"embedded": 2, "remaining": 0, "failed": [], "error": None, "stop_reason": "lease_lost"},
])
def test_parity_aborts_on_any_chunk_backfill_failure(gate, result):
    gate.command._chunk_backfill_run.return_value = result
    with pytest.raises(SystemExit, match="catch-up failed: summaries=complete:0 chunks=error"):
        gate.client._catch_up("u", SimpleNamespace(embedding_max_batch_items=10))
    assert _report(gate.client)["status"] == "failed" and "chunk_error" in _report(gate.client)


def test_parity_refuses_a_summaries_only_catch_up(gate):
    """Parity on: no chunk receipt means not done, even where LCM-X runs no automatic chunk backfill."""
    _drop_chunk_receipt(gate.client)
    with pytest.raises(SystemExit, match="incomplete after 3"):
        gate.client._catch_up("u", SimpleNamespace(embedding_max_batch_items=10))


def test_hosted_always_requires_chunks_whatever_the_old_parity_switch_says(gate, monkeypatch):
    """F5: no LCMX_EMBED_CHUNK_PARITY any more; the hosted path always embeds chunks and requires them complete."""
    monkeypatch.setenv("LCMX_EMBED_CHUNK_PARITY", "0")
    gate.client._catch_up("u", CONFIG)
    assert _report(gate.client)["status"] == "complete"
    gate.command._chunk_backfill_run.assert_called_once()
    _drop_chunk_receipt(gate.client)
    with pytest.raises(SystemExit, match="incomplete after 3"):
        gate.client._catch_up("u", CONFIG)


@pytest.mark.parametrize("outcome", ["summaries=error", "error", "model_unavailable", "summaries=error chunks=complete:0"])
def test_hosted_error_parts_still_abort(gate, outcome):
    gate.maintenance.run_incremental_embedding_pass.return_value = outcome
    with pytest.raises(SystemExit, match="catch-up failed"):
        gate.client._catch_up("u", SimpleNamespace(embedding_max_batch_items=10))


def test_hosted_incomplete_summaries_never_complete(gate):
    gate.maintenance.run_incremental_embedding_pass.return_value = "summaries=partial:0"
    with pytest.raises(SystemExit, match="stalled|incomplete"):
        gate.client._catch_up("u", SimpleNamespace(embedding_max_batch_items=10))


def test_default_fastembed_gate_still_requires_chunks(gate):
    gate.client.__dict__.update(_base_url=None, _provider_name="fastembed")
    with pytest.raises(SystemExit, match="catch-up failed: summaries=complete:0"):
        gate.client._catch_up("u", object())
    gate.command._chunk_backfill_run.assert_not_called()


# -- round 3: document reruns, registration retry, withheld chunks, clean abort -------------------------------

def test_one_chunk_backfill_failure_recovers_on_a_retry_uncertain_rerun(gate):
    gate.command._chunk_backfill_run.side_effect = [FAILED, dict(OK, embedded=4)]
    gate.client._catch_up("u", CONFIG)
    assert _report(gate.client)["status"] == "complete" and gate.sleeps == [4.0]
    assert [call.kwargs["retry_uncertain"] for call in gate.command._chunk_backfill_run.call_args_list] == [False, True]
    assert _report(gate.client)["outcome"] == "summaries=complete:0 chunks=complete:4"


def test_persistent_chunk_failure_aborts_after_two_reruns(gate):
    gate.command._chunk_backfill_run.return_value = FAILED
    with pytest.raises(SystemExit, match="catch-up failed: summaries=complete:0 chunks=error"):
        gate.client._catch_up("u", CONFIG)
    assert gate.command._chunk_backfill_run.call_count == 3 and gate.sleeps == [4.0, 16.0]


def test_one_summary_failure_recovers_through_product_summary_backfill(gate):
    """A failed summary row (receipt operator_retry) is rerun with LCM-X's summary backfill, retry_uncertain."""
    seed(gate.client, summary_retry=1)
    passes = iter(["summaries=partial", "summaries=complete:0"])
    gate.maintenance.run_incremental_embedding_pass.side_effect = lambda db, config: next(passes)

    def rerun(config, db, **kwargs):
        seed(gate.client)  # the product clears operator_retry once the row embeds
        return dict(OK, embedded=1)

    gate.command._embedding_backfill_summary_run.side_effect = rerun
    gate.client._catch_up("u", CONFIG)
    assert _report(gate.client)["status"] == "complete" and gate.sleeps == [4.0]
    assert gate.command._embedding_backfill_summary_run.call_args.kwargs == {
        "apply": True, "limit": 10, "retry_uncertain": True}


def test_persistent_summary_failure_aborts_after_two_reruns(gate):
    seed(gate.client, summary_retry=1)
    gate.maintenance.run_incremental_embedding_pass.return_value = "summaries=partial"
    with pytest.raises(SystemExit, match="catch-up failed: summaries=failed_rows_remain"):
        gate.client._catch_up("u", CONFIG)
    assert gate.command._embedding_backfill_summary_run.call_count == 2 and gate.sleeps == [4.0, 16.0]


@pytest.mark.parametrize("errors,calls,sleeps", [
    (["error: OpenAI-compatible embedding request failed (429)"], 2, [1.0]),
    (["error: OpenAI-compatible embedding request failed (503)", "error: OpenAI-compatible network error: x"], 3,
     [1.0, 4.0]),
    (["error: OpenAI-compatible operation deadline exceeded"], 2, [1.0]),
])
def test_registration_retries_only_retryable_errors(gate, errors, calls, sleeps):
    gate.command._chunk_current_profile = Mock(return_value=None)
    gate.command._embedding_register_profiles.side_effect = [f"LCM embedding warmup\nstatus: error\n{e}"
                                                             for e in errors] + [{"dim": 2}]
    gate.client._catch_up("u", CONFIG)
    assert gate.command._embedding_register_profiles.call_count == calls and gate.sleeps == sleeps


@pytest.mark.parametrize("error,calls", [("error: OpenAI-compatible embedding request failed (400)", 1),
                                         ("error: OpenAI-compatible embedding request failed (401)", 1),
                                         ("error: OpenAI-compatible embedding request failed (429)", 3)])
def test_registration_aborts_on_non_retryable_or_after_three_attempts(gate, error, calls):
    gate.command._chunk_current_profile = Mock(return_value=None)
    gate.command._embedding_register_profiles.return_value = f"LCM embedding warmup\nstatus: error\n{error}"
    with pytest.raises(SystemExit, match="warmup failed"):
        gate.client._catch_up("u", CONFIG)
    assert gate.command._embedding_register_profiles.call_count == calls


def test_registration_failure_ends_in_a_clean_abort(gate):
    gate.command._embedding_read_connection = Mock(side_effect=sqlite3.OperationalError("database is locked"))
    with pytest.raises(SystemExit, match="catch-up failed: OperationalError; re-ingest"):
        gate.client._catch_up("u", CONFIG)
    assert lcmx_client._ABORT.is_set()


def test_a_withheld_chunk_is_an_unembedded_chunk_and_aborts(gate):
    """Filter off: nothing should be withheld; if LCM-X still reports one, parity fails. No withheld sidecar."""
    gate.command._chunk_backfill_run.return_value = dict(OK, embedded=2, privacy_withheld=1)
    with pytest.raises(SystemExit, match="chunks=error"):
        gate.client._catch_up("u", CONFIG)
    assert not (gate.client._user_dir("u") / "withheld_chunks.json").exists()


# -- round 4: error parts survive reruns (A4) ---------------------------------------------------------------------

@pytest.mark.parametrize("first", ["summaries=error", "summaries=refused", "summaries=partial:0 error"])
def test_a_summary_error_part_is_not_erased_by_a_clean_rerun(gate, first):
    seed(gate.client, summary_retry=1)
    passes = iter([first, "summaries=complete:0"])
    gate.maintenance.run_incremental_embedding_pass.side_effect = lambda db, config: next(passes)

    def rerun(config, db, **kwargs):
        seed(gate.client)
        return dict(OK, embedded=1)

    gate.command._embedding_backfill_summary_run.side_effect = rerun
    with pytest.raises(SystemExit, match="catch-up failed"):
        gate.client._catch_up("u", CONFIG)


@pytest.mark.parametrize("first", [{"error": "provider authentication failed; 401"},
                                   {"stop_reason": "lease_lost"}, {"stop_reason": "privacy_refused", "error": "x"}])
def test_a_chunk_error_is_not_erased_by_a_clean_rerun(gate, first):
    gate.command._chunk_backfill_run.side_effect = [dict(FAILED, **first), dict(OK, embedded=2)]
    with pytest.raises(SystemExit, match="chunks=error"):
        gate.client._catch_up("u", CONFIG)


# -- round 5: summary rerun stop reasons ---------------------------------------------------------------------------

def _summary_rerun_setup(gate, results):
    """One failed summary row; the incremental pass reports partial; reruns return ``results`` in order. A rerun
    clears the row only when it embedded without a stop."""
    seed(gate.client, summary_retry=1)
    passes = iter(["summaries=partial"] + ["summaries=complete:0"] * 5)
    gate.maintenance.run_incremental_embedding_pass.side_effect = lambda db, config: next(passes)
    queue = list(results)

    def rerun(config, db, **kwargs):
        result = queue.pop(0)
        if result.get("embedded") and not result.get("stop_reason"):
            seed(gate.client)
        return result

    gate.command._embedding_backfill_summary_run.side_effect = rerun


@pytest.mark.parametrize("stop", ["lease_lost", "identity_superseded", "process_exit", "privacy_refused", "other_new"])
def test_a_terminal_summary_rerun_stop_aborts_even_if_a_later_rerun_is_clean(gate, stop):
    _summary_rerun_setup(gate, [dict(OK, stop_reason=stop), dict(OK, embedded=1)])
    with pytest.raises(SystemExit, match="catch-up failed: summaries=error"):
        gate.client._catch_up("u", CONFIG)


def test_a_budget_stop_in_a_summary_rerun_continues(gate):
    """op_budget_exhausted is resumable (LCM-X stops before claiming the next batch; command.py:3883-3886)."""
    _summary_rerun_setup(gate, [dict(OK, stop_reason="op_budget_exhausted"), dict(OK, embedded=1)])
    gate.client._catch_up("u", CONFIG)
    assert _report(gate.client)["status"] == "complete" and gate.sleeps == [4.0, 16.0]


# -- default path golden (fake LCM-X modules) ----------------------------------------------------------------

class FakeProvider:
    model_id = "BAAI/bge-small-en-v1.5"
    dim = 4

    def __init__(self):
        self.document_calls = []

    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        return [[0.0] * 4 for _ in texts]


class FakeConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@pytest.mark.parametrize("provider", [None, "fastembed"])
def test_default_fastembed_path_golden(monkeypatch, tmp_path, provider):
    """Same resolver call, same unwrapped provider, one embed call per list, same store-config fields
    (golden from 05d424e6). Hosted-only variables set here must change nothing."""
    monkeypatch.setenv("LCMX_REPO", str(tmp_path))
    for name, value in (("LCMX_EMBED_PROVIDER", provider), ("LCMX_EMBED_BASE_URL", URL),
                        ("LCMX_EMBED_BATCH_SIZE", "3"), ("LCM_EMBEDDING_API_KEY", FAKE_KEY)):
        if value is not None:
            monkeypatch.setenv(name, value)
    fake, calls = FakeProvider(), []
    longmemeval = ModuleType("benchmarking.longmemeval")
    longmemeval._ensure_hermes_lcm_package = lambda: None
    longmemeval.deterministic_session_summary = lambda turns: ""
    longmemeval.resolve_harness_provider = lambda *a, **k: calls.append((a, k)) or fake
    benchmarking = ModuleType("benchmarking")
    benchmarking.__path__ = []
    package = ModuleType("hermes_lcm")
    package.__path__ = []
    config_module = ModuleType("hermes_lcm.config")
    config_module.LCMConfig = FakeConfig
    for name, module in (("benchmarking", benchmarking), ("benchmarking.longmemeval", longmemeval),
                         ("hermes_lcm", package), ("hermes_lcm.config", config_module)):
        monkeypatch.setitem(sys.modules, name, module)

    c = lcmx_client.LcmxClient()
    assert calls == [(("fastembed", "BAAI/bge-small-en-v1.5"), {})]
    assert type(c._embedder) is lcmx_client._LockedEmbedder and c._embedder._provider is fake
    assert (c._provider_name, c._model, c._dim) == ("fastembed", "BAAI/bge-small-en-v1.5", 4)
    c._embedder.embed_documents([f"t{i}" for i in range(25)])
    assert [len(batch) for batch in fake.document_calls] == [25]
    assert vars(c._config("u")) == {
        "database_path": str(tmp_path / "db" / "u" / "lcm.db"), "embeddings_enabled": True,
        "embedding_provider": "fastembed", "embedding_model": "BAAI/bge-small-en-v1.5",
    }
