"""Offline evidence delivery for saved LongMemEval, LoCoMo and PMv2 searches."""
import json
from itertools import chain, zip_longest

BUDGETS = (4096, 8192, 16384)
WARNING = ("Verbatim-only undercounts evidence carried by summaries and must not gate "
           "any flip that changes the summary/raw mix.")


def evidence(row):
    if "context" in row:
        from locomo.locomo_responses import _dedup_shared_context
        return ([row["evidence_turns"].get(key, "").partition(": ")[2] for key in row["evidence"]],
                _dedup_shared_context(row["context"]))
    if "row_idx" in row:
        turns = row["related_conversation_snippet"]
        turns = json.loads(turns) if isinstance(turns, str) else turns
        return [turn["content"] for turn in turns], row["search_context"]
    return [turn.partition(" : ")[2] for turn in row["answer_evidences"]], row["search_context"]


def classes(turns, context, hits, sources, encoding):
    tokens = encoding.encode_ordinary(context)
    for budget in BUDGETS:
        cut = encoding.decode(tokens[:budget], errors="ignore")
        # Positions are zero-based delivered lines. Count only complete summary lines.
        lines = cut.count("\n") + (cut == context or context[len(cut):].startswith("\n"))
        covered = {source for hit in hits if hit["kind"] == "summary" and 0 <= hit["position"] < lines
                   for source in hit["source_store_ids"]}
        text = " ".join(cut.split())
        for mode in ("strict", "lenient"):
            labels = []
            for turn, source in zip(turns, sources, strict=True):
                gold = " ".join(turn.split())
                spans = [gold] if mode == "strict" or len(gold) <= 300 else (
                    gold[i:i + 300] for i in range(len(gold) - 299))
                labels.append("verbatim" if any(span and span in text for span in spans)
                              else "summary" if source in covered else "none")
            yield (budget, mode), labels


def score(data, encoding):
    rows = {}
    for row in chain.from_iterable(data.values()):
        qid = str(row.get("question_id", row.get("row_idx")))
        if ("answer_evidences" in row and qid.endswith("_abs")) or row.get("evidence") == []:
            continue
        turns, context = evidence(row)
        if qid == "None" or qid in rows or not turns:
            raise ValueError(f"Invalid question ID or missing gold evidence for {qid}")
        if row.get("reflect_answer"):
            context = ""
        hits = row.get("delivered_hits", [])
        sources = row.get("evidence_source_store_ids", [None] * len(turns))
        rows[qid] = (str(row["category"]), dict(classes(turns, context, hits, sources, encoding)),
                     hits or context.splitlines())
    return rows


def pair(a, b):
    for qid in sorted(a.keys() | b.keys()):
        left, right = a.get(qid, (None, {}, [])), b.get(qid, (None, {}, []))
        if left[:2] != right[:2]:
            yield qid, next((hits for hits in zip_longest(left[2], right[2]) if hits[0] != hits[1]), None)
