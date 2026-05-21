"""`jimm-ci-pr` — list the commits in a PR and (optionally) apply the
per-SHA approval label that `jimm-ci-run` looks for.

The per-SHA approval workflow requires the maintainer to apply a label
named `<prefix><short-sha>` to a PR. Without help, that means copying
the head SHA out of the GitHub UI and editing a label name. This tool
prints a table of every commit in the PR with its short SHA, local
timestamp, author email, message subject, and the exact label string
that approves it — and with `--apply`, creates and applies the label
for the current head in one shot.
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .config import Config
from .github_app import GitHubApp

_PR_URL_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>\d+)"
)


@dataclass
class PrRef:
    owner: str
    repo: str
    number: int

    @property
    def fullname(self) -> str:
        return f"{self.owner}/{self.repo}"


def parse_pr_arg(arg: str, *, default_owner: str, default_repo: str) -> PrRef:
    m = _PR_URL_RE.match(arg.strip())
    if m:
        return PrRef(m["owner"], m["repo"], int(m["number"]))
    if arg.isdigit():
        return PrRef(default_owner, default_repo, int(arg))
    raise SystemExit(
        f"jimm-ci-pr: expected a PR URL or PR number, got {arg!r}"
    )


def _parse_iso(stamp: str) -> datetime:
    # GitHub returns 'Z'-suffixed timestamps. fromisoformat handles those
    # since Python 3.11.
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def _format_local(stamp: str) -> str:
    try:
        dt = _parse_iso(stamp).astimezone()
        return dt.strftime("%Y-%m-%d %H:%M %Z")
    except ValueError:
        return stamp


def _author_email(commit: dict[str, Any]) -> str:
    author = (commit.get("commit") or {}).get("author") or {}
    return author.get("email") or author.get("name") or "?"


def _subject(commit: dict[str, Any]) -> str:
    msg = (commit.get("commit") or {}).get("message") or ""
    return msg.splitlines()[0] if msg else ""


def _render(commits: list[dict[str, Any]], label_prefix: str,
            head_sha: str | None) -> str:
    if not commits:
        return "no commits found"
    rows: list[tuple[str, str, str, str, str, bool]] = []
    for c in commits:
        sha = c.get("sha", "")
        short = sha[:7]
        stamp = ((c.get("commit") or {}).get("committer") or {}).get("date", "")
        when = _format_local(stamp)
        label = f"{label_prefix}{short}"
        author = _author_email(c)
        subject = _subject(c)
        is_head = bool(head_sha) and sha == head_sha
        rows.append((short, when, label, author, subject, is_head))

    headers = ("sha", "committed", "label", "author", "subject")
    widths = [len(h) for h in headers]
    for short, when, label, author, subject, _ in rows:
        for i, val in enumerate((short, when, label, author, subject)):
            widths[i] = max(widths[i], len(val))

    def fmt_row(vals: tuple[str, ...]) -> str:
        return "  ".join(v.ljust(w) for v, w in zip(vals, widths)).rstrip()

    out = [fmt_row(headers), fmt_row(tuple("-" * w for w in widths))]
    for short, when, label, author, subject, is_head in rows:
        line = fmt_row((short, when, label, author, subject))
        if is_head:
            line += "  <- HEAD"
        out.append(line)
    return "\n".join(out)


async def _inspect_async(args: argparse.Namespace) -> int:
    cfg = Config.from_env()
    pr_ref = parse_pr_arg(args.pr, default_owner=cfg.repo_owner,
                           default_repo=cfg.repo_name)
    gh = GitHubApp(cfg.app_id, cfg.installation_id, cfg.private_key)
    try:
        commits = await gh.list_pull_commits(pr_ref.fullname, pr_ref.number)
        head_sha = commits[-1].get("sha") if commits else None
        print(_render(commits, cfg.approval_label_prefix, head_sha))

        if args.apply:
            if not head_sha:
                print("jimm-ci-pr: no commits, nothing to label", file=sys.stderr)
                return 1
            label = f"{cfg.approval_label_prefix}{head_sha[:7]}"
            await gh.ensure_label(
                pr_ref.fullname, label,
                description=f"Approve commit {head_sha[:7]} for jimm-ci",
            )
            await gh.add_labels_to_issue(pr_ref.fullname, pr_ref.number, [label])
            print(f"\napplied label `{label}` to PR #{pr_ref.number}")
        return 0
    finally:
        await gh.aclose()


def cli_main() -> None:
    parser = argparse.ArgumentParser(
        prog="jimm-ci-pr",
        description=(
            "Print commits in a PR with their per-SHA approval label strings; "
            "optionally apply the label for the current head."
        ),
    )
    parser.add_argument(
        "pr",
        help="PR URL (https://github.com/owner/repo/pull/N) or bare PR number",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="create and apply the approval label for the current PR head",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_inspect_async(args)))
