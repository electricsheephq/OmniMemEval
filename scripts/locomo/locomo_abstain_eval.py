"""LoCoMo binary abstain judge (category-5 / adversarial mode).

Reads results/locomo/<lib>-<version>/<lib>_locomo_responses.json (produced with LOCOMO_QA_MODE=cat5 or all) and asks
the EVAL model one question per answer: does the answer abstain (says the information is not available / not in
memory / cannot be determined) or does it assert an answer? For category-5 items it also asks whether an asserted
answer matches the item's adversarial_answer.

Scoring: category 5 PASS = abstains; FAIL = asserts any answer. Categories 1-4 are recorded too (mode "all"), so a
both-ways check can count answerable items that the judge wrongly scores as abstain.

The judge prompt is pinned: its sha256 (system + user template) is printed and stored in the output, and
--expect-prompt-sha refuses to run when it differs. Output: <lib>_locomo_abstain_judged.json (per item + summary).
Resumable: items already judged successfully are kept.
"""
import argparse
import asyncio
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.env import load_env  # noqa: E402
from utils.checkpoint import atomic_json_dump  # noqa: E402

ABSTAIN_JUDGE_SYSTEM = "You label whether an answer abstains. Reply with JSON only."
ABSTAIN_JUDGE_PROMPT = """A memory assistant answered a question about past conversations.

Question: {question}
Assistant answer: {answer}
Reference wrong answer (may be "(none)"): {adversarial_answer}

Decide two things.
1. "abstains": true if the answer declines to give an answer because the information is not available, not in
   memory, not mentioned, unknown, or cannot be determined. An answer that states any specific answer to the question,
   even with hedging, is false. An answer that only says the premise is wrong without asserting an alternative is true.
2. "asserts_reference": true only if the answer asserts the reference wrong answer or something with the same
   meaning; false otherwise (false whenever the reference is "(none)").

Reply with exactly: {{"abstains": true|false, "asserts_reference": true|false}}"""


JUDGE_TEMPERATURE = 0
JUDGE_MAX_TOKENS = 256
# Amendment 1 section 4 (judge model fallback): an OpenAI-compatible endpoint that rejects max_tokens (the frontier shim)
# runs with LOCOMO_ABSTAIN_MAX_TOKENS=none, which omits the field; the instrument block records max_tokens=null.
# LOCOMO_ABSTAIN_TAG suffixes the judged-output file so a second judge model never resumes from another's items.
# LOCOMO_ABSTAIN_SETTINGS_NOTE is copied verbatim into the instrument block (for example "temperature sent, not applied").
if (os.getenv("LOCOMO_ABSTAIN_MAX_TOKENS") or "").strip().lower() == "none":
    JUDGE_MAX_TOKENS = None
OUT_TAG = (os.getenv("LOCOMO_ABSTAIN_TAG") or "").strip()
if OUT_TAG and not OUT_TAG.replace("-", "").replace("_", "").isalnum():
    raise SystemExit("LOCOMO_ABSTAIN_TAG must be alphanumeric with - or _")


def verdict_pass(label: dict) -> bool:
    """Frozen mapping (fixed lever-3 bars): PASS iff abstains is true and asserts_reference is false."""
    return label["abstains"] is True and label["asserts_reference"] is False


def harness_sha() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# v2 (RETIRED by Amendment 1; was a candidate, never pinned): v1 scored two answers that give the
# answer while noting a missing detail ("...no specific city was given"; "...then she read it in 2022") as abstaining.
ABSTAIN_JUDGE_PROMPT_V2 = """A memory assistant answered a question about past conversations.

Question: {question}
Assistant answer: {answer}
Reference wrong answer (may be "(none)"): {adversarial_answer}

Decide two things.
1. "abstains": true only if the answer gives NO answer to the question: it says the information is not available,
   not in memory, not mentioned, unknown, or cannot be determined, and states nothing that answers the question.
   An answer that only says the premise is wrong, without asserting an alternative, is true.
   It is false whenever the answer states any specific content that answers the question (a date, time, place, name,
   item, reason, description), even if it is hedged ("probably", "I think", "if X then"), even if it also says some
   detail is missing or unspecified, and even if it is combined with an abstention phrase.
2. "asserts_reference": true only if the answer asserts the reference wrong answer or something with the same
   meaning; false otherwise (false whenever the reference is "(none)").

Reply with exactly: {{"abstains": true|false, "asserts_reference": true|false}}"""
# v3 (lever-3 bars Amendment 1 revision 3, sha256 9d446c7c...): v1 with the definitions replaced by the amendment's
# rule 1. Nothing else changes (system prompt, schema, model settings, frozen mapping). v2 is RETIRED (record only).
ABSTAIN_JUDGE_PROMPT_V3 = """A memory assistant answered a question about past conversations.

Question: {question}
Assistant answer: {answer}
Reference wrong answer (may be "(none)"): {adversarial_answer}

Decide two things.
1. "abstains": true when the answer gives no answer for the person or thing the question names. It may say that the
   information is not in memory or not specified. It may also correct the question's premise by attributing the fact
   to a different person, named or identified by an unambiguous role ("the other speaker"). Such an attributed fact is
   not an answer, provided the response gives no answer for the person the question names. A premise correction
   abstains only if it says the fact is not known for, or does not belong to, the person the question names.
   Otherwise the attributed fact is offered as the answer, and the response asserts.
   "abstains" is false when the answer gives any answer content for the person or thing the question names:
   - a plain answer;
   - a hedged one ("probably X");
   - one mixed with an abstention phrase ("not specified, but likely X");
   - a partial one with a note that a detail is missing ("X, though no city was given");
   - another person's fact presented as the named person's, or given with no person identified;
   - another person's fact offered as the answer to the question, even with its attribution ("Answer: Sweden (that
     was Caroline)").
2. "asserts_reference": true only when the answer asserts the reference wrong answer (or something with the same
   meaning) as the answer for the person or thing the question names. A fact attributed to a different person, and not
   offered as the answer, does not count as asserting the reference, even when the reference's words appear. Always
   false when the reference is "(none)".

Reply with exactly: {{"abstains": true|false, "asserts_reference": true|false}}"""
PROMPTS = {"v1": ABSTAIN_JUDGE_PROMPT, "v2": ABSTAIN_JUDGE_PROMPT_V2, "v3": ABSTAIN_JUDGE_PROMPT_V3}
PROMPT_VERSION = os.getenv("LOCOMO_ABSTAIN_PROMPT_VERSION", "v1")


def prompt_sha256(version=None) -> str:
    text = PROMPTS[version or PROMPT_VERSION]
    return hashlib.sha256((ABSTAIN_JUDGE_SYSTEM + "\n\x00\n" + text).encode("utf-8")).hexdigest()


def parse_label(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in judge reply: {text[:200]!r}")
    obj = json.loads(text[start:end + 1])
    out = {}
    for key in ("abstains", "asserts_reference"):
        value = obj.get(key)
        if not isinstance(value, bool):
            raise ValueError(f"judge reply field {key!r} is not a boolean: {text[:200]!r}")
        out[key] = value
    return out


def item_key(group_id: str, idx: int, question: str) -> str:
    return f"{group_id}#{idx}#{hashlib.sha256(question.encode('utf-8')).hexdigest()[:12]}"


async def judge_one(client, model, rec, semaphore):
    adv = rec.get("adversarial_answer")
    prompt = PROMPTS[PROMPT_VERSION].format(question=rec.get("question") or "", answer=rec.get("answer") or "",
                                         adversarial_answer="(none)" if adv in (None, "") else adv)
    async with semaphore:
        resp = await client.chat.completions.create(
            model=model, temperature=JUDGE_TEMPERATURE,
            **({} if JUDGE_MAX_TOKENS is None else {"max_tokens": JUDGE_MAX_TOKENS}),
            messages=[{"role": "system", "content": ABSTAIN_JUDGE_SYSTEM}, {"role": "user", "content": prompt}])
    label = parse_label(resp.choices[0].message.content or "")
    return {**label, "pass": verdict_pass(label)}


def summarize(items: dict) -> dict:
    ok = [v for v in items.values() if v.get("status") == "success"]
    cat5 = [v for v in ok if v.get("category") == 5]
    other = [v for v in ok if v.get("category") != 5]
    return {
        "judged": len(ok), "failed": sum(1 for v in items.values() if v.get("status") != "success"),
        "cat5_n": len(cat5), "cat5_pass": sum(v["pass"] for v in cat5),
        "cat5_pass_abstain": sum(v["abstains"] for v in cat5),
        "cat5_asserts_adversarial": sum((not v["abstains"]) and v["asserts_reference"] for v in cat5),
        "cat5_pass_rate": round(sum(v["pass"] for v in cat5) / len(cat5), 4) if cat5 else None,
        "cat1_4_n": len(other), "cat1_4_scored_abstain": sum(v["abstains"] for v in other),
    }


def instrument(model, sha):
    return {"judge_model": model, "judge_base_url": os.getenv("EVAL_BASE_URL"),
            "judge_extra_body": os.getenv("EVAL_EXTRA_BODY"), "temperature": JUDGE_TEMPERATURE,
            "max_tokens": JUDGE_MAX_TOKENS, "prompt_version": PROMPT_VERSION, "prompt_sha256": sha, "harness_sha": harness_sha(),
            "output_tag": OUT_TAG or None, "settings_note": os.getenv("LOCOMO_ABSTAIN_SETTINGS_NOTE") or None,
            "mapping": "PASS iff abstains is true and asserts_reference is false"}


async def canned_async(path, out_path, llm_workers, expect_sha):
    """Score a canned set: every row has question, answer, adversarial_answer and expect_abstains /
    expect_asserts_reference / expect_pass. Exit 0 only when every item is right on all three."""
    sha = prompt_sha256()
    if expect_sha and expect_sha != sha:
        raise SystemExit(f"abstain judge prompt sha256 {sha} != expected {expect_sha}")
    load_env()
    from utils.llm_client import create_async_openai_client
    client, model = create_async_openai_client("EVAL")
    raw = open(path, "rb").read()
    rows = [json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()]
    sem = asyncio.Semaphore(llm_workers)
    out = []

    async def run(r):
        try:
            got = await judge_one(client, model, r, sem)
            err = None
        except Exception as exc:  # a parse failure is a failed (wrong) unit
            got, err = None, f"{type(exc).__name__}: {exc}"[:300]
        ok = got is not None and (got["abstains"], got["asserts_reference"], got["pass"]) == (
            r["expect_abstains"], r["expect_asserts_reference"], r["expect_pass"])
        out.append({"id": r["id"], "kind": r.get("kind"), "got": got, "error": err, "right": ok,
                    "verdict_right": got is not None and got["pass"] == r["expect_pass"]})

    await asyncio.gather(*(run(r) for r in rows))
    out.sort(key=lambda x: x["id"])
    res = {**instrument(model, sha), "canned_file": path, "canned_sha256": hashlib.sha256(raw).hexdigest(),
           "n": len(out), "right": sum(x["right"] for x in out), "verdict_right": sum(x["verdict_right"] for x in out),
           "wrong": [x for x in out if not x["right"]], "items": out}
    atomic_json_dump(res, out_path, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "items"}, indent=1))
    return 0 if res["right"] == res["n"] else 1


async def main_async(lib, version, llm_workers, expect_sha):
    sha = prompt_sha256()
    print(f"[ABSTAIN JUDGE] prompt sha256={sha}")
    if expect_sha and expect_sha != sha:
        raise SystemExit(f"abstain judge prompt sha256 {sha} != expected {expect_sha}")
    load_env()
    from utils.llm_client import create_async_openai_client
    client, model = create_async_openai_client("EVAL")
    print(f"[EVAL] model={model}")
    rdir = f"results/locomo/{lib}-{version}"
    responses = json.load(open(f"{rdir}/{lib}_locomo_responses.json"))
    out_path = f"{rdir}/{lib}_locomo_abstain_judged{'' if PROMPT_VERSION == 'v1' else '_' + PROMPT_VERSION}{'_' + OUT_TAG if OUT_TAG else ''}.json"
    prior = json.load(open(out_path)).get("items", {}) if os.path.exists(out_path) else {}
    items, todo = {}, []
    for group_id, recs in responses.items():
        for idx, rec in enumerate(recs if isinstance(recs, list) else [recs]):
            key = item_key(group_id, idx, str(rec.get("question") or ""))
            if prior.get(key, {}).get("status") == "success":
                items[key] = prior[key]
            elif str(rec.get("status") or "success") != "success":
                items[key] = {"group_id": group_id, "category": rec.get("category"), "status": "answer_failed"}
            else:
                todo.append((key, group_id, rec))
    sem = asyncio.Semaphore(llm_workers)

    async def run(key, group_id, rec):
        base = {"group_id": group_id, "category": rec.get("category"), "question": rec.get("question"),
                "answer": rec.get("answer"), "adversarial_answer": rec.get("adversarial_answer")}
        try:
            items[key] = {**base, **await judge_one(client, model, rec, sem), "status": "success"}
        except Exception as exc:  # recorded, retried on the next resume
            items[key] = {**base, "status": "failed", "error": f"{type(exc).__name__}: {exc}"[:300]}

    await asyncio.gather(*(run(*t) for t in todo))
    summary = summarize(items)
    atomic_json_dump({**instrument(model, sha),
                      "qa_mode": os.getenv("LOCOMO_QA_MODE") or "default", "summary": summary, "items": items},
                     out_path, indent=2)
    print(json.dumps(summary, indent=1))
    print(f"📁 Results saved to: {out_path}")
    return 1 if summary["failed"] else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lib", default="lcmx")
    ap.add_argument("--version")
    ap.add_argument("--canned", help="score a canned JSONL set instead of a run's responses")
    ap.add_argument("--canned-out")
    ap.add_argument("--llm-workers", type=int, default=4)
    ap.add_argument("--expect-prompt-sha", default="")
    a = ap.parse_args()
    if a.canned:
        sys.exit(asyncio.run(canned_async(a.canned, a.canned_out or a.canned + ".scored.json", a.llm_workers,
                                          a.expect_prompt_sha)))
    if not a.version:
        ap.error("--version is required unless --canned is given")
    sys.exit(asyncio.run(main_async(a.lib, a.version, a.llm_workers, a.expect_prompt_sha)))


if __name__ == "__main__":
    main()
