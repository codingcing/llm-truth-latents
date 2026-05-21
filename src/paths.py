"""Resolve where pipeline artifacts (activations, labels, figures) live.

Lets the same notebooks run on Colab and locally without editing path cells.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_COLAB_RESULTS = "/content/drive/MyDrive/abstention-geometry/results"
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _on_colab() -> bool:
    return "google.colab" in sys.modules


def results_dir() -> str:
    """Return the artifact directory for the current environment.

    Resolution order:
      1. The RESULTS_DIR environment variable, if set (always wins). Set this on
         your Mac to point at the Drive-synced folder, e.g.
         ~/Library/CloudStorage/GoogleDrive-<you>/My Drive/abstention-geometry/results
      2. On Colab: a Google Drive path. Drive must be mounted first.
      3. Locally: the repo's results/ directory.

    The directory is created if missing. On Colab, raises RuntimeError if Drive is
    not mounted, rather than silently writing to ephemeral runtime storage.

    Returns:
        Absolute path with a trailing separator (safe to concatenate filenames).
    """
    override = os.environ.get("RESULTS_DIR")
    if override:
        path = Path(override).expanduser()
    elif _on_colab():
        if not Path("/content/drive/MyDrive").exists():
            raise RuntimeError(
                "Google Drive is not mounted. Run this first:\n"
                "  from google.colab import drive; drive.mount('/content/drive')"
            )
        # If the notebook is run from inside the Drive-synced repo, keep results
        # next to the repo so they sync straight back to your machine — works no
        # matter where under My Drive the folder lives. Otherwise (repo cloned to
        # ephemeral /content) fall back to the fixed Drive path.
        if str(_REPO_ROOT).startswith("/content/drive"):
            path = _REPO_ROOT / "results"
        else:
            path = Path(_COLAB_RESULTS)
    else:
        path = _REPO_ROOT / "results"

    path.mkdir(parents=True, exist_ok=True)
    return str(path) + os.sep
