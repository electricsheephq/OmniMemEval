import json
import os
import re
import shutil
import sqlite3
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from utils.checkpoint import atomic_json_dump

from .base_client import env_int, env_str, require_env

_DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"

# One warmed embedder per process (FastEmbed loads an ONNX model; ingestion
# runs several client instances on worker threads).
_EMBEDDER = None
_EMBEDDER_LOCK = threading.Lock()
_USER_LOCKS = {}
_USER_LOCKS_GUARD = threading.Lock()
_RECALL_LOCK = threading.RLock()
_ABORT = threading.Event()
_ABORT_MESSAGE = None
_ABORT_LOCK = threading.Lock()
_CONFIG_SET_LOGGED = set()


def _abort(message, *, rebuild=False):
    global _ABORT_MESSAGE
    with _ABORT_LOCK:
        if not _ABORT.is_set():
            _ABORT_MESSAGE = str(message) + ("; re-ingest with --clear / --restart-unit" if rebuild else "")
            _ABORT.set()
    raise SystemExit(_ABORT_MESSAGE)


def _check_abort():
    if _ABORT.is_set():
        _abort(_ABORT_MESSAGE)


def _safe(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-._") or "user"


def _user_lock(user_id):
    with _USER_LOCKS_GUARD:
        return _USER_LOCKS.setdefault(user_id, threading.Lock())


class _LockedEmbedder:
    """Serialize calls into the shared embedder across worker threads."""

    def __init__(self, provider):
        self._provider = provider
        self._lock = threading.Lock()

    @property
    def model_id(self):
        return self._provider.model_id

    @property
    def dim(self):
        return int(self._provider.dim)

    def __getattr__(self, name):
        return getattr(self._provider, name)

    def embed_documents(self, texts):
        with self._lock:
            return self._provider.embed_documents(texts)

    def embed_query(self, text):
        with self._lock:
            return self._provider.embed_query(text)


class LcmxClient:
    """LCM-X (hermes-lcm) in-process client for OmniMemEval.

    Runs a headless LCM-X store per ``user_id`` under
    ``$LCMX_DB_ROOT/<user_id>/`` using the same construction as the LCM-X
    memorybench bridge (benchmarks/qa-harness/src/providers/hermes-lcm/bridge):
    lossless message rows, conversational chunk embeddings, one deterministic
    session summary per ``add`` call, and the production ``lcm_recall`` tool
    for search. No LLM is used inside LCM-X.

    Each ``add`` call is one LCM-X session. Every message keeps its role and
    its ``chat_time`` as the host event time (``observed_at``), which
    ``lcm_recall`` returns as each message hit's ``event_time``.

    Configurable env vars:
        LCMX_REPO            – LCM-X checkout to import (required; never modified)
        LCMX_DB_ROOT         – root for per-user stores (required)
        LCMX_CONFIG          – "fleet" (default): fastembed embeddings on;
                               "off": full-text only
        LCMX_EMBED_PROVIDER  – embedding provider for fleet (default: fastembed)
        LCMX_EMBED_MODEL     – embedding model (default: BAAI/bge-small-en-v1.5)
        LCMX_EMBED_CACHE     – optional SQLite text->vector cache shared with the
                               LCM-X LongMemEval harness
        LCMX_SEARCH_SCOPE_BIAS – optional lcm_recall scope_bias override
        LCMX_CHUNK_POLICY    – LCM-X chunk policy: "conversational" (default; skips
                               turns under 40 tokens) or "full" (every message)
        LCMX_DELIVERY        – "full" (default): full stored message per hit;
                               "snippet": lcm_recall's 300-char snippet
        LCMX_CATCHUP_MAX_PASSES – maintenance pass limit (default: 500)
        LCMX_EXTRA_ENTRIES   – "1": render lcm_recall extra entries (extra=true,
                               recall_extra_entries) after the top_k hits as
                               their window text only; off by default
        FASTEMBED_CACHE_PATH / LCM_LONGMEMEVAL_FASTEMBED_CACHE – model cache
        TIKTOKEN_CACHE_DIR  – populated tokenizer cache (set by the job)
    """

    def __init__(self):
        repo = Path(require_env("LCMX_REPO")).expanduser().resolve()
        if not repo.is_dir():
            raise RuntimeError(f"LCMX_REPO does not exist: {repo}")
        self._db_root = Path(require_env("LCMX_DB_ROOT")).expanduser().resolve()
        self._db_root.mkdir(parents=True, exist_ok=True)

        mode = (env_str("LCMX_CONFIG", "fleet") or "fleet").lower()
        if mode not in ("fleet", "off"):
            raise ValueError(f"LCMX_CONFIG must be 'fleet' or 'off', got {mode!r}")
        self._embeddings = mode == "fleet"
        self._provider_name = (env_str("LCMX_EMBED_PROVIDER", "fastembed") or "fastembed").lower()
        self._model = env_str("LCMX_EMBED_MODEL", _DEFAULT_FASTEMBED_MODEL)
        self._scope_bias = env_str("LCMX_SEARCH_SCOPE_BIAS")
        self._top_k_cap = env_int("LCMX_TOPK_CAP", 25, min_value=1)
        self._delivery = (env_str("LCMX_DELIVERY", "full") or "full").lower()
        self._chunk_policy = (env_str("LCMX_CHUNK_POLICY", "conversational") or "conversational").lower()
        if self._chunk_policy not in ("conversational", "full"):
            raise ValueError(f"LCMX_CHUNK_POLICY must be 'conversational' or 'full', got {self._chunk_policy!r}")
        if self._delivery not in ("full", "snippet"):
            raise ValueError(f"LCMX_DELIVERY must be 'full' or 'snippet', got {self._delivery!r}")

        # Import LCM-X exactly the way its own LongMemEval harness does.
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        cache = env_str("LCMX_EMBED_CACHE")
        if cache:
            os.environ.setdefault("LCM_LONGMEMEVAL_EMBED_CACHE", cache)
        from benchmarking.longmemeval import (
            _ensure_hermes_lcm_package,
            deterministic_session_summary,
            resolve_harness_provider,
        )

        _ensure_hermes_lcm_package()
        self._summary = deterministic_session_summary
        self._embedder = None
        self._dim = 0
        if self._embeddings:
            global _EMBEDDER
            with _EMBEDDER_LOCK:
                if _EMBEDDER is None:
                    _EMBEDDER = _LockedEmbedder(
                        resolve_harness_provider(self._provider_name, self._model)
                    )
            self._embedder = _EMBEDDER
            self._dim = self._embedder.dim
            self._model = self._embedder.model_id

    # -- store helpers --------------------------------------------------------

    def _user_dir(self, user_id):
        return self._db_root / _safe(user_id)

    def _db_path(self, user_id):
        return self._user_dir(user_id) / "lcm.db"

    def _sessions_path(self, user_id):
        # session_id -> {"order", "time"}: the session order (recency prior, as
        # in the bridge) and the session event time for summary hits, which
        # carry no per-message observed_at.
        return self._user_dir(user_id) / "sessions.json"

    def _load_sessions(self, user_id):
        path = self._sessions_path(user_id)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _config(self, user_id):
        from hermes_lcm.config import LCMConfig

        if not self._embeddings:
            config = LCMConfig(database_path=str(self._db_path(user_id)),
                               embeddings_enabled=False)
        else:
            config = LCMConfig(
                database_path=str(self._db_path(user_id)),
                embeddings_enabled=True,
                embedding_provider=self._provider_name,
                embedding_model=self._model,
            )
        # R2 flag arms: LCMX_CONFIG_SET="field=value,field=value" sets LCM-X config fields
        # (bool/int/float/str coerced from the field's default). Unknown fields abort.
        for item in filter(None, (env_str("LCMX_CONFIG_SET", "") or "").split(",")):
            name, _, raw = item.partition("=")
            name, raw = name.strip(), raw.strip()
            if not hasattr(config, name):
                raise ValueError(f"LCMX_CONFIG_SET: unknown LCM-X config field {name!r}")
            default = getattr(config, name)
            if isinstance(default, bool):
                if raw.lower() not in ("1", "true", "yes", "on", "0", "false", "no", "off"):
                    raise ValueError(f"LCMX_CONFIG_SET: {name} needs a boolean, got {raw!r}")
                value = raw.lower() in ("1", "true", "yes", "on")
            elif isinstance(default, int):
                value = int(raw)
            elif isinstance(default, float):
                value = float(raw)
            else:
                value = raw
            setattr(config, name, value)
            key = f"config_set:{name}={value!r}"
            if key not in _CONFIG_SET_LOGGED:
                _CONFIG_SET_LOGGED.add(key)
                print(f"[lcmx] LCMX_CONFIG_SET applied: {name}={value!r}", file=sys.stderr)
        return config

    def _open(self, user_id):
        from hermes_lcm.dag import SummaryDAG
        from hermes_lcm.store import MessageStore
        from hermes_lcm.vector_store import VectorStore

        self._user_dir(user_id).mkdir(parents=True, exist_ok=True)
        config = self._config(user_id)
        db = str(self._db_path(user_id))
        store = MessageStore(db, ingest_protection_config=config)
        dag = SummaryDAG(db)
        vector_store = VectorStore(db, config=config)
        return config, store, dag, vector_store

    # -- add ------------------------------------------------------------------

    def add(self, messages, user_id, *args, **kwargs):
        """Ingest one session of messages losslessly.

        Args:
            messages: ``[{"role", "content", "chat_time"?}, ...]``.
            user_id: Store owner; one LCM-X database per user.
            session_key / session_id / conv_id: Optional session identifier
                (positional third argument also accepted). Without one, the
                session is named ``<user_id>-s<order>``.
            timestamp: Optional ISO-8601 event time used when a message has
                no ``chat_time``.
        """
        _check_abort()
        if not messages:
            return
        session_key = (
            kwargs.get("session_key")
            or kwargs.get("session_id")
            or kwargs.get("conv_id")
            or (args[0] if args and isinstance(args[0], str) else None)
        )
        default_time = kwargs.get("timestamp")
        if isinstance(default_time, (int, float)) and not isinstance(default_time, bool):
            from datetime import datetime, timezone

            default_time = datetime.fromtimestamp(default_time, tz=timezone.utc).isoformat()

        from hermes_lcm.chunking import iter_message_chunks
        from hermes_lcm.dag import SummaryNode
        from hermes_lcm.vector_store import EmbeddingIdentity

        with _user_lock(user_id):
            _check_abort()
            self._require_healthy_store(user_id)
            self._user_dir(user_id).mkdir(parents=True, exist_ok=True)
            atomic_json_dump({"status": "failed", "phase": "ingest"},
                             self._user_dir(user_id) / "embedding_catchup.json", indent=2)
            sessions = self._load_sessions(user_id)
            order = len(sessions) + 1
            session_id = str(session_key) if session_key else f"{_safe(user_id)}-s{order}"
            if session_id in sessions:
                session_id = f"{session_id}#{order}"

            rows_in = []
            for m in messages:
                event_time = m.get("chat_time") or default_time
                row = {
                    "role": str(m.get("role", "user")),
                    "content": str(m.get("content", "")),
                }
                if event_time:
                    row["timestamp"] = event_time
                rows_in.append(row)
            session_time = next((r["timestamp"] for r in rows_in if r.get("timestamp")), None)

            config, store, dag, vector_store = self._open(user_id)
            try:
                store_ids = store.append_batch(
                    session_id, rows_in, source="benchmark", conversation_id=session_id
                )
                summary_text = self._summary(
                    [{"role": r["role"], "content": r["content"]} for r in rows_in]
                )
                node_id = dag.add_node(
                    SummaryNode(
                        session_id=session_id,
                        depth=0,
                        summary=summary_text,
                        token_count=len(summary_text.split()),
                        source_token_count=sum(len(r["content"].split()) for r in rows_in),
                        source_type="messages",
                        created_at=float(order),
                    )
                )
                if self._embeddings:
                    vector_store.register_profile(self._model, self._provider_name, self._dim)
                    identity = vector_store.capture_identity(
                        self._model, provider=self._provider_name
                    )
                    vector_store.register_profile(
                        self._model, self._provider_name, self._dim, task="chunk"
                    )
                    chunk_identity = EmbeddingIdentity.canonical(
                        self._provider_name, self._model, "", self._dim,
                        "float32", "little", "chunk",
                    )
                    rows = [
                        {"store_id": sid, "role": r["role"], "content": r["content"]}
                        for sid, r in zip(store_ids, rows_in)
                    ]
                    chunks = list(iter_message_chunks(rows, policy=self._chunk_policy))
                    if chunks:
                        vectors = self._embedder.embed_documents([c.text for c in chunks])
                        for chunk, vector in zip(chunks, vectors):
                            vector_store.record_chunk_embedding(
                                chunk.chunk_id, self._model, vector,
                                store_id=chunk.store_id, chunk_index=chunk.chunk_index,
                                char_start=chunk.char_start, char_end=chunk.char_end,
                                token_estimate=chunk.token_estimate, identity=chunk_identity,
                            )
                    summary_vector = self._embedder.embed_documents([summary_text])[0]
                    vector_store.record_embedding(
                        str(node_id), "summary", self._model, summary_vector, identity=identity
                    )
                sessions[session_id] = {"order": order, "time": session_time}
                atomic_json_dump(sessions, self._sessions_path(user_id))
            except Exception as exc:  # noqa: BLE001 - make ingestion failures fatal
                # Ingest exceptions abort both arms: the pending marker would refuse a re-add.
                _abort(f"LCM-X ingest failed: {type(exc).__name__}", rebuild=True)
            finally:
                vector_store.close()
                dag.close()
                store.close()

            self._catch_up(user_id, config)

    def _require_healthy_store(self, user_id):
        path = self._user_dir(user_id) / "embedding_catchup.json"
        if path.exists():
            marker = json.loads(path.read_text())
            if marker.get("status") == "failed":
                _abort(f"LCM-X store failed during {marker.get('phase', 'catch-up')}", rebuild=True)

    def _catchup_receipts(self, user_id):
        # Read only receipts for the active identities, never an older profile.
        db = self._db_path(user_id).resolve()
        conn = sqlite3.connect(f"{db.as_uri()}?mode=ro", uri=True)
        try:
            records = {}
            for task in ("summary", "chunk"):
                row = conn.execute(
                    "SELECT m.value FROM lcm_embedding_profile p JOIN metadata m "
                    "ON m.key = 'embedding_catchup:' || ? || ':' || p.identity_hash "
                    "WHERE p.active=1 AND p.archived_at IS NULL "
                    "AND replace(p.task, '+prefix', '')=? AND p.provider=? "
                    "AND p.model_name=? AND CASE WHEN instr(p.revision, '+prefix:')=0 "
                    "THEN '' ELSE substr(p.revision, instr(p.revision, '+prefix:')) END=? "
                    "ORDER BY p.registered_at DESC LIMIT 1",
                    (task, task, self._provider_name, self._model,
                     getattr(self._embedder, "identity_revision_suffix", "")),
                ).fetchone()
                records[task] = json.loads(row[0]) if row else {}
            return records
        finally:
            conn.close()

    def _catch_up(self, user_id, config):
        import hermes_lcm.tools as lcm_tools

        path = self._user_dir(user_id) / "embedding_catchup.json"
        report = {"status": "not_applicable", "passes": 0}
        if not hasattr(lcm_tools, "_lcm_recall_catchup_reason") or not self._embeddings:
            report["reason"] = "no_receipt_gate" if self._embeddings else "embeddings_disabled"
            atomic_json_dump(report, path, indent=2)
            return
        from hermes_lcm import command
        from hermes_lcm import embedding_maintenance as maintenance

        report["status"] = "failed"
        limit = env_int("LCMX_CATCHUP_MAX_PASSES", 500, min_value=1)
        db = str(self._db_path(user_id))
        idle, count, retry_deadline = 0, 0, None

        def progress_state(receipts):
            return ({k: {f: v for f, v in r.items() if f != "updated_at"}
                     for k, r in receipts.items()}, maintenance._scan_frontier(db, command))

        try:
            atomic_json_dump(report, path, indent=2)
            previous = progress_state(self._catchup_receipts(user_id))
            while count < limit:
                _check_abort()
                outcome = maintenance.run_incremental_embedding_pass(db, config)
                report.update(passes=report["passes"] + 1, outcome=outcome)
                if outcome in {"lease_held", "model_preparing"} or str(outcome).endswith("=lease_held"):
                    retry_deadline = retry_deadline or time.monotonic() + 600
                    if time.monotonic() >= retry_deadline:
                        _abort("LCM-X catch-up retry exceeded 600s", rebuild=True)
                    time.sleep(min(5, retry_deadline - time.monotonic()))
                    continue
                count += 1
                retry_deadline = None
                receipts = self._catchup_receipts(user_id)
                report["receipts"] = receipts
                complete = re.fullmatch(r"summaries=complete:\d+ chunks=complete:\d+", str(outcome))
                if complete and all(r.get("remaining") == 0 and not r.get("operator_retry") for r in receipts.values()):
                    report["status"] = "complete"
                    return
                snapshot = progress_state(receipts)
                writes = sum(int(n) for n in re.findall(r":(\d+)(?: |$)", str(outcome)))
                idle = idle + 1 if not writes and snapshot == previous else 0
                previous = snapshot
                if idle >= 20:
                    _abort("LCM-X catch-up stalled for 20 passes", rebuild=True)
                # The pass does not raise on failures. Reject refusals and errors;
                # only bounded partial/complete work may need another pass.
                if not re.fullmatch(r"summaries=(?:complete|partial):\d+ chunks=(?:complete|partial):\d+", str(outcome)):
                    _abort(f"LCM-X embedding catch-up failed: {outcome}", rebuild=True)
            _abort(f"LCM-X embedding catch-up incomplete after {limit} passes", rebuild=True)
        except Exception as exc:  # noqa: BLE001 - make maintenance failures fatal
            _abort(f"LCM-X embedding catch-up failed: {type(exc).__name__}", rebuild=True)
        finally:
            atomic_json_dump(report, path, indent=2)

    def set_search_results_dir(self, path):
        self._search_status_path = Path(path) / "lcmx_recall_status.json"

    def _record_search_payload(self, user_id, payload):
        reason = payload.get("degraded_reason")
        coverage = payload.get("provenance", {}).get("coverage", {})
        degraded = "error" in payload or self._embeddings and bool(
            payload.get("degraded") or reason or payload.get("timeout")
            or any(coverage.get(arm) in {"none", "disabled"} for arm in ("summary", "chunk"))
        )
        with _user_lock(user_id):
            path = self._user_dir(user_id) / "recall_status.json"
            report = json.loads(path.read_text()) if path.exists() else {
                "payloads": 0, "degraded_payloads": 0,
            }
            report["payloads"] += 1
            report["degraded_payloads"] += int(degraded)
            atomic_json_dump(report, path, indent=2)
        # Count issued payloads cumulatively across resumes/retries, retaining aborted-run degradation.
        path = getattr(self, "_search_status_path", self._db_root / "search_status.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        stage = json.loads(path.read_text()) if path.exists() else {"stage": "search", "payloads": 0, "degraded_payloads": 0}
        stage["payloads"] += 1
        stage["degraded_payloads"] += int(degraded)
        atomic_json_dump(stage, path, indent=2)
        if degraded and self._embeddings:
            # Stage wrappers can turn Exception into a skipped question. This
            # invalidates the whole benchmark stage, even with skip flags set.
            _abort(
                f"LCM-X degraded recall: {report['degraded_payloads']} degraded payload(s): {reason}"
            )

    # -- search ---------------------------------------------------------------

    def _log_recall_call(self, user_id, query, config, payload):
        """Opt-in per-call arm proof (LCMX_RECALL_CALL_LOG=1): one JSONL line per lcm_recall call, next to the
        stage's lcmx_recall_status.json, with the arm label (LCMX_ARM), the resolved value of every config field
        named in LCMX_CONFIG_SET or LCMX_LOG_CONFIG_FIELDS ("<absent>" when the tree has no such field), the
        provenance policy markers, and the hit store ids. No query or hit text is logged."""
        if (env_str("LCMX_RECALL_CALL_LOG", "") or "") not in ("1", "true", "yes"):
            return
        import hashlib

        fields = [item.partition("=")[0].strip()
                  for item in (env_str("LCMX_CONFIG_SET", "") or "").split(",") if item.strip()]
        fields += [f.strip() for f in (env_str("LCMX_LOG_CONFIG_FIELDS", "") or "").split(",") if f.strip()]
        prov = payload.get("provenance") or {}
        from datetime import datetime, timezone

        line = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            "user_id": user_id,
            "query_sha256": hashlib.sha256(str(query).encode("utf-8")).hexdigest(),
            "arm": env_str("LCMX_ARM", "") or "",
            "config": {f: getattr(config, f, "<absent>") for f in dict.fromkeys(fields)},
            "forget_withheld": [r.get("store_id") if isinstance(r, dict) else r
                                for r in prov.get("forget_withheld") or []],
            "forget_withheld_receipts": prov.get("forget_withheld") or [],
            "forget_unavailable": prov.get("forget_unavailable"),
            "delivery_order": prov.get("delivery_order"),
            "hit_store_ids": [h.get("store_id") for h in payload.get("hits", [])],
            "degraded": bool(payload.get("degraded")),
        }
        path = getattr(self, "_search_status_path", self._db_root / "search_status.json").parent / "lcmx_recall_calls.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with _RECALL_LOCK, open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, default=str) + "\n")

    def search(self, query, user_id, top_k):
        """Run ``lcm_recall`` and return one ``[event_time] text`` line per hit."""
        _check_abort()
        self._require_healthy_store(user_id)
        if not self._db_path(user_id).exists():
            if self._embeddings:
                _abort("LCM-X semantic search store missing; ingest first", rebuild=True)
            return ""
        import hermes_lcm.tools as lcm_tools

        config, store, dag, vector_store = self._open(user_id)
        try:
            engine = SimpleNamespace(
                _config=config,
                _store=store,
                _dag=dag,
                _hermes_home=str(self._user_dir(user_id)),
                # Disjoint from every ingested session so the scope prior never
                # lifts an evidence session.
                current_session_id=f"__lcmx_omnimemeval_probe__{_safe(user_id)}",
            )
            if self._embeddings:
                cache_key = (self._provider_name, str(self._embedder.model_id).strip())
                engine._lcm_embedding_provider_cache = (cache_key, self._embedder)
            args = {"query": query, "limit": min(int(top_k), self._top_k_cap)}
            if self._scope_bias:
                args["scope_bias"] = float(self._scope_bias)
            with _RECALL_LOCK:
                _check_abort()
                payload = json.loads(lcm_tools.lcm_recall(args, engine=engine))
                self._record_search_payload(user_id, payload)
                self._log_recall_call(user_id, query, config, payload)
            # LCMX_EXTRA_ENTRIES=1 (lever 9): extra entries are rendered after the
            # top_k hits from their window alone; off, the hit list is untouched.
            hits, extras = payload.get("hits", []), []
            if (env_str("LCMX_EXTRA_ENTRIES", "") or "") in ("1", "true", "yes"):
                extras = [h for h in hits if h.get("extra") is True]
                hits = [h for h in hits if h.get("extra") is not True]
            # Recall snippets are cut at 300 chars; deliver the full stored
            # message for message hits (LCMX_DELIVERY=snippet restores the cut).
            full_rows = {}
            if self._delivery == "full":
                ids = [h.get("store_id") for h in hits
                       if isinstance(h.get("store_id"), int)]
                full_rows = store.get_batch(ids)
        finally:
            vector_store.close()
            dag.close()
            store.close()

        if "error" in payload:
            raise RuntimeError(f"lcm_recall error: {payload['error']}")
        sessions = self._load_sessions(user_id)
        lines = []
        for hit in hits[:top_k]:
            row = full_rows.get(hit.get("store_id")) if hit.get("kind") != "summary" else None
            if row and row.get("content"):
                text = " ".join(str(row["content"]).split())
            else:
                # Drop the FTS match-highlight markers; keep the stored text.
                snippet = str(hit.get("snippet") or "").replace(">>>", "").replace("<<<", "")
                text = " ".join(snippet.split())
            if not text:
                continue
            event_time = hit.get("event_time") or (
                sessions.get(str(hit.get("session_id"))) or {}
            ).get("time")
            prefix = f"[{event_time}] " if event_time else ""
            if hit.get("kind") == "summary":
                prefix += "(session summary) "
            lines.append(prefix + text)
        for hit in extras:  # bounded by lcm-x (at most N); not cut at top_k
            text = " ".join(str(hit.get("window") or "").split())
            if not text:
                continue
            event_time = hit.get("event_time") or (
                sessions.get(str(hit.get("session_id"))) or {}
            ).get("time")
            prefix = f"[{event_time}] " if event_time else ""
            if hit.get("kind") == "summary":
                prefix += "(session summary) "
            lines.append(prefix + text)
        return "\n".join(lines)

    # -- delete ---------------------------------------------------------------

    def delete_all(self, user_id):
        with _user_lock(user_id):
            shutil.rmtree(self._user_dir(user_id), ignore_errors=True)

    def delete(self, user_id):
        self.delete_all(user_id)
