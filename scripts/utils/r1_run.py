"""Opt-in R1 store and checkpoint settings shared by ingestion and search."""

import json
import os
import subprocess
from pathlib import Path


def store_version(version):
    override = os.getenv("R1_STORE_VERSION")
    if override == version:
        raise ValueError("A/A' requires R1_STORE_VERSION to differ from --version")
    return override or version


def bind_run(directory, version):
    """Keep cached work tied to the R1 settings that produced it."""
    switches = {name: os.getenv(f"R1_{name}") for name in ("SPLIT", "EVIDENCE", "LOCAL_RNG", "STORE_VERSION")}
    settings = dict(switches, STORE_VERSION=store_version(version))
    marker = directory / "r1_settings.json"
    if marker.exists():
        if json.loads(marker.read_text()) != settings:
            raise ValueError(f"R1 settings differ from cached work in {directory}; use a fresh --version")
    elif any(switches.values()):
        if (directory / "tmp").exists() or (directory / "success_records.txt").exists():
            raise ValueError(f"Existing R1 checkpoints in {directory}; use a fresh --version")
        directory.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(settings))


def pin_build(results_dir):
    if os.getenv("R1_BUILD_PIN") == "1":
        repo = Path(os.environ["LCMX_REPO"]).expanduser().resolve()
        git = ["git", "-C", str(repo)]
        head = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(git + ["status", "--porcelain"], text=True))
        with (Path(results_dir) / "experiment_config.sh").open("a") as config:
            config.write(f"lcmx_commit={head}\nlcmx_dirty={int(dirty)}\n")
