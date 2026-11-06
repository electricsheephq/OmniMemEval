import asyncio
import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "locomo"))

from locomo import locomo_common, locomo_abstain_eval as ae  # noqa: E402

QAS = [{"question": f"q{c}", "category": c} for c in (1, 2, 3, 4, 5)]


def test_default_mode_excludes_category_5(monkeypatch):
    monkeypatch.delenv("LOCOMO_QA_MODE", raising=False)
    assert [q["category"] for q in locomo_common.filter_eval_qas(QAS)] == [1, 2, 3, 4]


def test_cat5_and_all_modes(monkeypatch):
    monkeypatch.setenv("LOCOMO_QA_MODE", "cat5")
    assert [q["category"] for q in locomo_common.filter_eval_qas(QAS)] == [5]
    monkeypatch.setenv("LOCOMO_QA_MODE", "all")
    assert [q["category"] for q in locomo_common.filter_eval_qas(QAS)] == [1, 2, 3, 4, 5]


def test_unknown_mode_refused(monkeypatch):
    monkeypatch.setenv("LOCOMO_QA_MODE", "cat6")
    with pytest.raises(ValueError):
        locomo_common.filter_eval_qas(QAS)


def test_prompt_sha_is_pinned():
    # Changing the judge prompt must be deliberate: update this pin and the gating packet together.
    assert ae.prompt_sha256() == "fa3f056ae87e2916126881b88bbb674f8e373f7fed4386de085d34d75b2cc473"
    assert len(ae.prompt_sha256()) == 64


def test_parse_label_strict():
    assert ae.parse_label('x {"abstains": true, "asserts_reference": false} y') == {"abstains": True,
                                                                                     "asserts_reference": False}
    with pytest.raises(ValueError):
        ae.parse_label('{"abstains": "yes", "asserts_reference": false}')
    with pytest.raises(ValueError):
        ae.parse_label("no json")


def test_summary_counts():
    items = {
        "a": {"category": 5, "status": "success", "abstains": True, "asserts_reference": False},
        "b": {"category": 5, "status": "success", "abstains": False, "asserts_reference": True},
        "c": {"category": 2, "status": "success", "abstains": True, "asserts_reference": False},
        "d": {"category": 5, "status": "failed"},
    }
    s = ae.summarize(items)
    assert (s["cat5_n"], s["cat5_pass_abstain"], s["cat5_asserts_adversarial"]) == (2, 1, 1)
    assert (s["cat1_4_n"], s["cat1_4_scored_abstain"], s["failed"]) == (1, 1, 1)


def test_judge_one_sends_reference_or_none():
    sent = []

    class C:
        class chat:
            class completions:
                @staticmethod
                async def create(**kw):
                    sent.append(kw["messages"][1]["content"])
                    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                        content=json.dumps({"abstains": False, "asserts_reference": True})))])

    sem = asyncio.Semaphore(1)
    asyncio.run(ae.judge_one(C, "m", {"question": "q", "answer": "a", "adversarial_answer": "wrong"}, sem))
    asyncio.run(ae.judge_one(C, "m", {"question": "q", "answer": "a"}, sem))
    assert "Reference wrong answer (may be \"(none)\"): wrong" in sent[0]
    assert "Reference wrong answer (may be \"(none)\"): (none)" in sent[1]
