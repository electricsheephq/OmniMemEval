import os
import sys
from dataclasses import dataclass

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from client_factory import lcmx_client  # noqa: E402


@dataclass
class FakeConfig:
    database_path: str = ""
    embeddings_enabled: bool = False
    recall_honour_forget: bool = False
    recall_limit: int = 20
    weight: float = 0.5
    recall_delivery_order: str = "rank"


def _config(monkeypatch, value):
    import types
    mod = types.ModuleType("hermes_lcm.config")
    mod.LCMConfig = FakeConfig
    monkeypatch.setitem(sys.modules, "hermes_lcm", types.ModuleType("hermes_lcm"))
    monkeypatch.setitem(sys.modules, "hermes_lcm.config", mod)
    monkeypatch.setenv("LCMX_CONFIG_SET", value)
    client = lcmx_client.LcmxClient.__new__(lcmx_client.LcmxClient)
    client._embeddings = False
    client._db_root = lcmx_client.Path("/nonexistent")
    return client._config("u")


def test_sets_typed_fields(monkeypatch):
    c = _config(monkeypatch, "recall_honour_forget=true,recall_limit=30,weight=0.25,recall_delivery_order=time")
    assert (c.recall_honour_forget, c.recall_limit, c.weight, c.recall_delivery_order) == (True, 30, 0.25, "time")


def test_unknown_field_and_bad_bool_raise(monkeypatch):
    with pytest.raises(ValueError):
        _config(monkeypatch, "no_such_field=1")
    with pytest.raises(ValueError):
        _config(monkeypatch, "recall_honour_forget=ture")
