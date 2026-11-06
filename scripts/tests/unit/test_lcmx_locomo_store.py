import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from locomo import locomo_search  # noqa: E402


class FakeClient:
    def __init__(self):
        self.calls = []

    def search(self, query, user_id, top_k):
        self.calls.append((user_id, top_k))
        return "\n".join(f"hit{i}" for i in range(top_k))


def test_dual_default_searches_both_speakers(monkeypatch):
    monkeypatch.delenv("LCMX_LOCOMO_STORE", raising=False)
    c = FakeClient()
    ctx, _, _, raw = locomo_search.lcmx_search(c, "q", "ua", "ub", 3, "A", "B")
    assert c.calls == [("ua", 3), ("ub", 3)]
    assert ctx.count("hit0") == 2


def test_single_store_fills_slots_with_distinct_hits(monkeypatch):
    monkeypatch.setenv("LCMX_LOCOMO_STORE", "single")
    c = FakeClient()
    ctx, _, _, raw = locomo_search.lcmx_search(c, "q", "ua", "ub", 3, "A", "B")
    assert c.calls == [("ua", 6)]
    for i in range(6):
        assert ctx.count(f"hit{i}") == 1
    assert ctx.index("hit2") < ctx.index("B") < ctx.index("hit3")
