#!/usr/bin/env python3
"""
Mirrors specified folder into into a GitLab clone.
Requires ssh keys to be set up for the user.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run_git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        text=True,
    )


def ensure_repo(dest: Path, remote_repo: str) -> None:
    git_meta = dest / ".git"
    if git_meta.exists():
        run_git(dest, "pull")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    subprocess.run(
        ["git", "clone", remote_repo, str(dest)],
        check=True,
    )


def clear(repo: Path) -> None:
    for child in repo.iterdir():
        if child.name == ".git" or child.name == "no_index":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def copy_tree_merge(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise SystemExit(f"Source is not a directory: {src}")
    shutil.copytree(src, dst, dirs_exist_ok=True)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Copy ml-app into a gitlab clone and push to GitLab.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=root / "no_index" / "oops",
        help="Path to the mirror git working tree (default: no_index/oops next to this script).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "ml-app",
        help="Folder to copy into the mirror (default: ml-app next to this script).",
    )
    parser.add_argument(
        "--gitignore",
        type=Path,
        help="Path to the .gitignore file to copy into the mirror root.",
    )
    parser.add_argument(
        "-m",
        "--message",
        required=True,
        help="Commit message for the mirror repository.",
    )
    parser.add_argument(
        "--git-repo",
        required=True,
        type=str,
        help="Remote repo",
    )
    args = parser.parse_args()

    dest = args.dest.resolve()
    source = args.source.resolve()
    gitignore_src = args.gitignore.resolve()
    remote_repo = args.git_repo

    if not gitignore_src.is_file():
        raise SystemExit(f".gitignore source not found: {gitignore_src}")

    ensure_repo(dest, remote_repo)
    clear(dest)
    copy_tree_merge(source, dest)
    shutil.copy2(gitignore_src, dest / ".gitignore")

    run_git(dest, "add", "-A")
    commit = run_git(dest, "commit", "-m", args.message, check=False)
    if commit.returncode != 0:
        print("git commit did not create a commit (working tree clean or hook).", file=sys.stderr)
    run_git(dest, "push")


if __name__ == "__main__":
    main()
