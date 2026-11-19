"""Offline evidence delivery for saved LongMemEval, LoCoMo and PMv2 searches."""
import json
from argparse import ArgumentParser
from itertools import chain, zip_longest
from pathlib import Path

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
        # LME gold sessions are extracted in set order, which can vary between runs.
        turns, sources = zip(*sorted(zip(turns, sources, strict=True), key=lambda item: item[0]))
        rows[qid] = (str(row["category"]), dict(classes(turns, context, hits, sources, encoding)),
                     hits or context.splitlines())
    return rows


def pair(a, b):
    for qid in sorted(a.keys() | b.keys()):
        left, right = a.get(qid, (None, {}, [])), b.get(qid, (None, {}, []))
        if left[:2] != right[:2]:
            yield qid, next((hits for hits in zip_longest(left[2], right[2]) if hits[0] != hits[1]), None)


def aggregates(rows):
    groups = {None: []}
    for category, scores, _ in rows.values():
        groups[None].append(scores)
        groups.setdefault(category, []).append(scores)
    for category in sorted(groups, key=lambda name: (name is not None, name)):
        questions = groups[category]
        for budget in BUDGETS:
            for mode in ("strict", "lenient"):
                labels = [scores[budget, mode] for scores in questions]
                items = sum(map(len, labels))
                result = dict(category=category, budget_tokens=budget, mode=mode,
                              questions=len(questions), evidence_items=items)
                for name, delivered in (("verbatim", ("verbatim",)),
                                        ("verbatim_plus_summary", ("verbatim", "summary"))):
                    hits = [sum(label in delivered for label in question) for question in labels]
                    result[name] = dict(evidence_recall=sum(hits) / items if items else None,
                                        all_evidence_delivered=sum(n == len(q) for n, q in zip(hits, labels))
                                        / len(questions) if questions else None)
                yield result


def write_reports(prefix, rows, arm):
    results = list(aggregates(rows))
    report = dict(arm=arm, summary_coverage="pending-adapter-hook", warning=WARNING, results=results)
    Path(f"{prefix}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [f"# Evidence delivery ({arm})", "", f"> **Warning:** {WARNING} (`summary_coverage: pending-adapter-hook`)", "",
             "| Category | Tokens | Match | Questions | Items | Verbatim recall | Verbatim all | V+S recall | V+S all |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        rates = [rate for key in ("verbatim", "verbatim_plus_summary") for rate in row[key].values()]
        cells = [row["category"] if row["category"] is not None else "**Overall**", row["budget_tokens"],
                 row["mode"], row["questions"], row["evidence_items"]]
        cells += [f"{rate:.2%}" if rate is not None else "—" for rate in rates]
        lines.append("| " + " | ".join(map(str, cells)) + " |")
    Path(f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    scorer = commands.add_parser("score", help="write JSON and Markdown evidence reports")
    scorer.add_argument("search", type=Path)
    scorer.add_argument("--arm", required=True)
    scorer.add_argument("--output-prefix", required=True, type=Path)
    pairing = commands.add_parser("pair", help="compare A and A' by question ID")
    pairing.add_argument("search", type=Path)
    pairing.add_argument("replica", type=Path)
    args = parser.parse_args(argv)
    from utils.nlp_metrics import get_encoding
    encoding = get_encoding()
    rows = score(json.loads(args.search.read_text(encoding="utf-8")), encoding)
    if args.command == "score":
        write_reports(args.output_prefix, rows, args.arm)
        return 0
    replica = score(json.loads(args.replica.read_text(encoding="utf-8")), encoding)
    discordant = dict(pair(rows, replica))
    print(json.dumps(dict(summary_coverage="pending-adapter-hook", warning=WARNING,
                          discordant_questions=len(discordant), first_differing_hits=discordant), indent=2))
    return int(bool(discordant))


if __name__ == "__main__":
    raise SystemExit(main())
