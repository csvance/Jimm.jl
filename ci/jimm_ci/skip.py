"""`jimm-ci-skip` — mark commits as deliberately not tested.

Posts a completed Check Run with `conclusion=skipped` for each
`jimm-ci / <family>` on the given commits, which makes `jimm-ci-run`
treat them as already-handled and stop reconsidering them. Use this to
clear out a backlog of master commits you have decided not to
retroactively test, or to permanently exclude a one-off commit.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date

from .config import Config
from .github_app import GitHubApp
from .path_filter import ALL_FAMILIES
from .runner import check_name, discover_jobs

LOG = logging.getLogger("jimm_ci")


async def _mark_skipped(
    gh: GitHubApp, repo: str, sha: str, families: tuple[str, ...],
) -> None:
    today = date.today().isoformat()
    output = {
        "title": "Skipped",
        "summary": (
            f"Marked skipped by `jimm-ci-skip` on {today}. "
            "Re-running CI on this commit requires deleting this check or "
            "pushing a new commit on top."
        ),
    }
    for family in families:
        name = check_name(family, "")
        await gh.create_check_run(
            repo, sha, name,
            status="completed", conclusion="skipped", output=output,
        )
        LOG.info("skipped %s on %s", name, sha[:8])


async def _skip_async(args: argparse.Namespace) -> int:
    cfg = Config.from_env()
    gh = GitHubApp(cfg.app_id, cfg.installation_id, cfg.private_key)
    try:
        targets: list[tuple[str, tuple[str, ...]]] = []

        if args.all_pending:
            jobs = await discover_jobs(cfg, gh)
            if not jobs:
                print("nothing to skip")
                return 0
            for job in jobs:
                targets.append((job.head_sha, job.families))

        for sha in args.shas:
            if len(sha) < 7:
                print(f"jimm-ci-skip: SHA {sha!r} is too short", flush=True)
                return 1
            targets.append((sha, ALL_FAMILIES))

        if not targets:
            print("jimm-ci-skip: nothing to do; pass SHAs or --all-pending")
            return 1

        for sha, families in targets:
            print(f"skipping {sha[:12]}: {','.join(families)}")
            if not args.dry_run:
                await _mark_skipped(gh, cfg.repo_fullname, sha, families)
        return 0
    finally:
        await gh.aclose()


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def cli_main() -> None:
    parser = argparse.ArgumentParser(
        prog="jimm-ci-skip",
        description=(
            "Mark commits as `skipped` so jimm-ci-run treats them as already "
            "handled. Pass SHAs explicitly or use --all-pending."
        ),
    )
    parser.add_argument(
        "shas", nargs="*",
        help="commit SHAs to mark skipped (any length >= 7)",
    )
    parser.add_argument(
        "--all-pending", action="store_true",
        help="run discovery and mark every currently-candidate job as skipped",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print what would be skipped without posting check runs",
    )
    args = parser.parse_args()
    _configure_logging()
    raise SystemExit(asyncio.run(_skip_async(args)))
