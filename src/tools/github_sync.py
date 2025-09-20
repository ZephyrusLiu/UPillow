"""Utility CLI for pushing and pulling this repository from GitHub.

This module wraps a couple of frequently used ``git`` commands with
lightweight error handling so that users can invoke them via

```
python -m src.tools.github_sync push
```

or through the convenience shell script located in ``scripts/``.  The
script intentionally keeps the feature surface minimal and defers to the
standard Git credential helper configured on the user's machine.  As a
result, it works with SSH keys, cached HTTPS credentials, or environment
variables such as ``GITHUB_TOKEN`` when used with the Git credential
manager.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable


class GitSyncError(RuntimeError):
    """Raised when a wrapped ``git`` command fails."""


def _find_repo_root(start: Path | None = None) -> Path:
    """Locate the repository root containing a ``.git`` directory."""

    current = start or Path.cwd()
    for path in (current, *current.parents):
        if (path / ".git").exists():
            return path
    raise GitSyncError(
        "Unable to locate the repository root. Please run this command inside the project directory."
    )


def _run_git(args: Iterable[str], cwd: Path) -> None:
    """Execute ``git`` with ``args`` in ``cwd`` and convert failures to ``GitSyncError``."""

    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        message_parts = ["Git command failed:", "git " + " ".join(args)]
        if stdout:
            message_parts.append("STDOUT:\n" + stdout)
        if stderr:
            message_parts.append("STDERR:\n" + stderr)
        raise GitSyncError("\n".join(message_parts))


def _detect_branch(cwd: Path) -> str:
    """Return the name of the currently checked-out branch."""

    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise GitSyncError("Unable to determine the current branch name.")
    branch = result.stdout.strip()
    if not branch:
        raise GitSyncError("The current branch name is empty; is the repository in a detached HEAD state?")
    return branch


def _do_push(remote: str, branch: str | None, set_upstream: bool, cwd: Path) -> None:
    target_branch = branch or _detect_branch(cwd)
    args = ["push", remote, target_branch]
    if set_upstream:
        args.insert(1, "--set-upstream")
    _run_git(args, cwd)


def _do_pull(remote: str, branch: str | None, cwd: Path) -> None:
    target_branch = branch or _detect_branch(cwd)
    _run_git(["fetch", remote], cwd)
    _run_git(["pull", "--ff-only", remote, target_branch], cwd)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push or pull the repository using Git.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common_args = {
        "--remote": {
            "default": "origin",
            "help": "Git remote name to use (default: origin).",
        },
        "--branch": {
            "default": None,
            "help": "Branch to push or pull. Defaults to the currently checked-out branch.",
        },
    }

    push_parser = subparsers.add_parser("push", help="Push the current branch to the remote.")
    push_parser.add_argument("--remote", **common_args["--remote"])
    push_parser.add_argument("--branch", **common_args["--branch"])
    push_parser.add_argument(
        "--set-upstream",
        action="store_true",
        help="Use --set-upstream when pushing (handy for first push of a branch).",
    )

    pull_parser = subparsers.add_parser("pull", help="Fetch and fast-forward merge from the remote.")
    pull_parser.add_argument("--remote", **common_args["--remote"])
    pull_parser.add_argument("--branch", **common_args["--branch"])

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        repo_root = _find_repo_root()
        if args.command == "push":
            _do_push(args.remote, args.branch, args.set_upstream, repo_root)
            print(f"Pushed branch '{args.branch or _detect_branch(repo_root)}' to remote '{args.remote}'.")
        elif args.command == "pull":
            _do_pull(args.remote, args.branch, repo_root)
            print(
                f"Fetched and fast-forwarded branch '{args.branch or _detect_branch(repo_root)}' from remote '{args.remote}'."
            )
        else:  # pragma: no cover - argparse enforces valid command
            raise GitSyncError(f"Unknown command: {args.command}")
    except GitSyncError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())