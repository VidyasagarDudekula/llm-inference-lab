# bench/logger.py
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Union

DEFAULT_LOG_PATH = Path("reports") / "metrics.jsonl"


def _utc_timestamp_seconds() -> str:
    """
    Returns UTC time in ISO format without timezone offset, matching:
    '2026-02-05T10:00:00'
    """
    return datetime.utcnow().replace(microsecond=0).isoformat()


def _find_repo_root(start: Path) -> Optional[Path]:
    """
    Walk upward looking for a .git directory to locate the repo root.
    Returns None if not found.
    """
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / ".git").exists():
            return p
    return None


def get_git_commit_hash(repo_dir: Optional[Union[str, Path]] = None, short: bool = True) -> str:
    """
    Try to get the current git commit hash.
    - First checks common CI env vars
    - Then runs `git rev-parse ...`
    Returns 'unknown' if it can't determine it.
    """
    # 1) CI / env fallbacks (common)
    for key in ("GIT_COMMIT", "GITHUB_SHA", "CI_COMMIT_SHA"):
        v = os.environ.get(key)
        if v:
            return v[:7] if short else v

    # 2) Try git command
    cwd: Optional[str] = None
    if repo_dir is not None:
        cwd = str(Path(repo_dir).resolve())
    else:
        # try to infer repo root from this file location
        maybe_root = _find_repo_root(Path(__file__).parent)
        if maybe_root is not None:
            cwd = str(maybe_root)

    cmd = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _json_default(obj: Any) -> Any:
    """
    Make a best-effort to serialize common numeric scalar types without importing them.
    - torch / numpy scalars often have .item()
    - pathlib.Path -> str
    """
    if isinstance(obj, Path):
        return str(obj)

    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass

    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def log_metric(
    metric_name: str,
    value: float,
    unit: str,
    config: Optional[Mapping[str, Any]] = None,
    *,
    log_path: Union[str, Path] = DEFAULT_LOG_PATH,
    commit_hash: Optional[str] = None,
    timestamp: Optional[str] = None,
    repo_dir: Optional[Union[str, Path]] = None,
    fsync: bool = False,
) -> dict:
    """
    Append one JSON object per line into reports/metrics.jsonl.

    Each line schema:
    {
      "timestamp": "2026-02-05T10:00:00",
      "commit_hash": "a1b2c3d",
      "metric_name": "matmul_f32",
      "value": 0.009,
      "unit": "ms",
      "config": {...}
    }
    """
    entry = {
        "timestamp": timestamp or _utc_timestamp_seconds(),
        "commit_hash": commit_hash or get_git_commit_hash(repo_dir=repo_dir, short=True),
        "metric_name": str(metric_name),
        "value": float(value),
        "unit": str(unit),
        "config": dict(config or {}),
    }

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Use binary append so we do exactly one write() call for the whole line.
    line = (json.dumps(entry, ensure_ascii=False, separators=(",", ":"), default=_json_default) + "\n").encode("utf-8")
    with open(path, "ab") as f:  # <- append mode for JSONL
        f.write(line)
        f.flush()
        if fsync:
            os.fsync(f.fileno())

    return entry
