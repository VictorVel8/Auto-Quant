"""Git commit lookup shared by manifests."""

import subprocess


def git_commit() -> str:
    """HEAD commit hash, or "unknown" outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() or "unknown"
