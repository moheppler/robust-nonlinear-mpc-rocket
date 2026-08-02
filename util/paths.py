"""Repo-anchored output locations, so results are written/read independent of the cwd."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def results_dir(name: str) -> Path:
    d = REPO_ROOT / "results" / name
    d.mkdir(parents=True, exist_ok=True)
    return d
