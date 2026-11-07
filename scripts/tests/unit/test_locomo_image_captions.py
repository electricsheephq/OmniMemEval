"""LoCoMo image captions: opt-in env (LOCOMO_IMAGE_CAPTIONS), default off and byte-identical."""

import json

import pytest
from locomo import locomo_ingestion as ing

SESSION = [
    {"speaker": "A", "dia_id": "D1:1", "text": "Look at this!", "img_url": ["u"], "blip_caption": "a photo of a dog",
     "query": "dog"},
    {"speaker": "B", "dia_id": "D1:2", "text": "Cute."},
]


class FakeClient:
    def __init__(self):
        self.calls = []

    def add(self, messages, user_id, **kw):
        self.calls.append((user_id, json.loads(json.dumps(messages))))


def meta():
    return {"session_date": "1:56 pm on 8 May, 2023", "conv_idx": 0, "session_key": "session_1",
            "speaker_a_user_id": "ua", "speaker_b_user_id": "ub", "speaker_a": "A", "speaker_b": "B"}


@pytest.mark.parametrize("value", [None, "", "0", "false", "no"])
def test_off_returns_same_object_and_same_ingest(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(ing.IMAGE_CAPTION_ENV, raising=False)
    else:
        monkeypatch.setenv(ing.IMAGE_CAPTION_ENV, value)
    assert ing.apply_image_captions(SESSION) is SESSION
    c = FakeClient()
    ing.ingest_session(c, SESSION, "lcmx", meta())
    texts = [m["content"] for _, msgs in c.calls for m in msgs]
    assert not any("description of attached image" in x for x in texts)
    assert any(x.startswith("A: Look at this!") for x in texts)


def test_on_appends_caption_once_and_drops_key(monkeypatch):
    monkeypatch.setenv(ing.IMAGE_CAPTION_ENV, "1")
    out = ing.apply_image_captions(SESSION)
    assert out[0]["text"] == "Look at this! (description of attached image: a photo of a dog)"
    assert "blip_caption" not in out[0] and out[0]["img_url"] == ["u"]
    assert out[1] == SESSION[1]
    assert SESSION[0]["text"] == "Look at this!" and "blip_caption" in SESSION[0]  # input not mutated
    c = FakeClient()
    ing.ingest_session(c, SESSION, "lcmx", meta())
    texts = [m["content"] for _, msgs in c.calls for m in msgs]
    assert any(x.startswith("A: Look at this! (description of attached image: a photo of a dog)") for x in texts)
    assert all(x.count("description of attached image") <= 1 for x in texts)
