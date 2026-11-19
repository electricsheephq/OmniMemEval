"""Frozen R1 splits, indexed by the original dataset positions."""

import hashlib
import json
import os
from pathlib import Path

from utils.checkpoint import atomic_json_dump
from utils.r1_run import bind_run


def select_indices(rows, id_key, results_dir, version):
    directory = Path(results_dir)
    bind_run(directory, version)
    split = os.getenv("R1_SPLIT")
    if split is None:
        return range(len(rows)) if id_key else rows
    if split not in {"dev", "confirm"}:
        raise ValueError("R1_SPLIT must be dev or confirm")
    ids = rows[id_key] if id_key else {pid: pid for pid in rows}
    members = {"dev": {}, "confirm": {}}
    for index, value in ids.items():
        identity = str(value)
        digest = hashlib.sha256(identity.encode()).hexdigest()
        members["dev" if int(digest, 16) % 100 < 40 else "confirm"][index] = identity
    for name, rows in members.items():
        identities = sorted(set(rows.values()))
        digest = hashlib.sha256(json.dumps(identities).encode()).hexdigest()
        print(f"{directory.parent.name} {name}: {len(rows)} rows, {len(identities)} unique IDs")
        atomic_json_dump({"ids": identities, "sha256": digest, "rows": len(rows)},
                         directory / f"r1_split_{name}.json", indent=2)
    return list(members[split])
