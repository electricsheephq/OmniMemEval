"""LoCoMo zep graph / graphiti paths read the dataset's caption key `blip_caption` (was `blip_captions`, never present)."""

from locomo import locomo_ingestion as ing

SESSION = [
    {"speaker": "A", "dia_id": "D1:1", "text": "Look at this!", "blip_caption": "a photo of a dog"},
    {"speaker": "B", "dia_id": "D1:2", "text": "Cute."},
]


def meta(**kw):
    return {"session_date": "1:56 pm on 8 May, 2023", "conv_idx": 0, "session_key": "session_1",
            "speaker_a_user_id": "ua", "speaker_b_user_id": "ub", "speaker_a": "A", "speaker_b": "B", **kw}


class Recorder:
    def __init__(self):
        self.graph, self.adds = [], []

    def sdk_graph_add(self, **kw):
        self.graph.append(kw["data"])

    def add(self, messages, user_id, **kw):
        self.adds.append((messages, kw))


def test_zep_graph_includes_caption():
    r = Recorder()
    ing.ingest_session(r, SESSION, "zep", meta(group_id="g1"))
    assert r.graph[0] == "A: Look at this! (description of attached image: a photo of a dog)"
    assert r.graph[1] == "B: Cute."


def test_graphiti_transcript_includes_caption():
    r = Recorder()
    ing.ingest_session(r, SESSION, "graphiti", meta())
    blob = repr(r.adds)
    assert "A: Look at this! (description of attached image: a photo of a dog)" in blob
    assert "B: Cute. (description" not in blob
