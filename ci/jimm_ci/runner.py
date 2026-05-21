"""Manual CLI runner for Jimm.jl CI.

`jimm-ci-run` queries GitHub for candidate commits — every open PR
labeled with a per-SHA approval marker, plus master commits in the last
30 days that have no completed `jimm-ci` Check Run — and dispatches each
through the `Builder` (git worktree + `Pkg.test` + Checks API reporting).
No webhooks, no public listener; invocation is human-driven over SSH.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .config import Config
from .github_app import GitHubApp
from .path_filter import ALL_FAMILIES, REPRESENTATIVE_VARIANT, families_for_paths

LOG = logging.getLogger("jimm_ci")
MAX_OUTPUT_TEXT = 60_000  # GitHub Checks API limit is 65535; leave headroom.
MASTER_LOOKBACK_DAYS = 30
CHECK_NAME_PREFIX = "jimm-ci / "

# Maps test family → Python sidecar that dumps its parity fixtures.
# Mirrors the family resolution in scripts/test_variant.sh.
_FAMILY_SIDECAR: dict[str, str] = {
    "bit":        "test/parity/dump_resnetv2_bit_io.py",
    "resnet":     "test/parity/dump_resnet_io.py",
    "convnext":   "test/parity/dump_convnext_io.py",
    "convnextv2": "test/parity/dump_convnextv2_io.py",
}


@dataclass
class Job:
    head_sha: str
    base_sha: str
    families: tuple[str, ...]
    full_sweep: bool
    label: str
    pr_number: int | None = None
    pr_title: str | None = None
    check_runs: dict[str, int] = field(default_factory=dict)


def _read_tail(log_path: Path, *, limit: int = MAX_OUTPUT_TEXT) -> str:
    if not log_path.exists():
        return ""
    data = log_path.read_bytes()
    if len(data) <= limit:
        return data.decode("utf-8", errors="replace")
    head = b"[... log truncated ...]\n"
    return (head + data[-(limit - len(head)):]).decode("utf-8", errors="replace")


async def _run_subprocess(
    cmd: list[str], *, env: dict[str, str], log_path: Path,
    cwd: Path | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        log.write(f"$ {' '.join(cmd)}\n".encode())
        log.flush()
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
            cwd=str(cwd) if cwd else None,
        )
        try:
            return await proc.wait()
        except asyncio.CancelledError:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=10)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
            raise


class Builder:
    def __init__(self, cfg: Config, app: GitHubApp) -> None:
        self.cfg = cfg
        self.app = app

    async def _ensure_mirror(self) -> None:
        mirror = self.cfg.mirror_dir
        if not mirror.exists():
            mirror.parent.mkdir(parents=True, exist_ok=True)
            await self._git(
                ["git", "clone", "--mirror", f"https://github.com/{self.cfg.repo_fullname}.git", str(mirror)],
                cwd=mirror.parent, log_path=self.cfg.log_dir / "mirror-init.log",
            )
        else:
            await self._git(
                ["git", "fetch", "--prune", "origin"],
                cwd=mirror, log_path=self.cfg.log_dir / "mirror-fetch.log",
            )

    async def _git(self, cmd: list[str], *, cwd: Path, log_path: Path) -> None:
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        rc = await _run_subprocess(cmd, env=env, log_path=log_path)
        if rc != 0:
            raise RuntimeError(f"git failed: {' '.join(cmd)} (cwd={cwd}, rc={rc})")

    async def _worktree(self, sha: str) -> Path:
        wt = self.cfg.workspace_dir / sha
        if wt.exists():
            shutil.rmtree(wt)
        wt.parent.mkdir(parents=True, exist_ok=True)
        await self._git(
            ["git", "-C", str(self.cfg.mirror_dir), "worktree", "add", "--detach", str(wt), sha],
            cwd=self.cfg.mirror_dir, log_path=self.cfg.log_dir / sha / "worktree.log",
        )
        self._link_parity_dir(wt)
        return wt

    def _link_parity_dir(self, wt: Path) -> None:
        """Point the worktree's `data/parity/` at the persistent fixtures dir.

        Parity fixtures are gitignored, so a fresh worktree starts with no
        `data/parity/` content. Symlinking the whole directory means dumps
        written here persist across worktrees and check-runs.
        """
        self.cfg.parity_dir.mkdir(parents=True, exist_ok=True)
        data = wt / "data"
        data.mkdir(parents=True, exist_ok=True)
        link = data / "parity"
        if link.is_symlink() or link.exists():
            if link.is_dir() and not link.is_symlink():
                shutil.rmtree(link)
            else:
                link.unlink()
        link.symlink_to(self.cfg.parity_dir)

    async def _drop_worktree(self, wt: Path) -> None:
        try:
            await self._git(
                ["git", "-C", str(self.cfg.mirror_dir), "worktree", "remove", "--force", str(wt)],
                cwd=self.cfg.mirror_dir, log_path=self.cfg.log_dir / wt.name / "worktree-remove.log",
            )
        except Exception:
            LOG.exception("worktree remove failed for %s", wt)
            if wt.exists():
                shutil.rmtree(wt, ignore_errors=True)

    def _env_for_family(self, family: str, variant: str) -> dict[str, str]:
        env = {
            **os.environ,
            "JULIA_NUM_THREADS": os.environ.get("JULIA_NUM_THREADS", "4"),
            "HF_HUB_CACHE": str(self.cfg.hf_cache),
            "JULIA_DEPOT_PATH": str(self.cfg.julia_depot),
            "JIMM_TEST_FAMILIES": family,
            "JIMM_TEST_VARIANTS": variant,
        }
        if self.cfg.hf_token:
            env["HF_TOKEN"] = self.cfg.hf_token
            env["HUGGING_FACE_HUB_TOKEN"] = self.cfg.hf_token
        return env

    def _env_for_sidecar(self) -> dict[str, str]:
        """Shared persistent venv + HF cache for every Python dump invocation."""
        env = {
            **os.environ,
            "UV_PROJECT_ENVIRONMENT": str(self.cfg.python_env),
            "HF_HUB_CACHE": str(self.cfg.hf_cache),
        }
        if self.cfg.hf_token:
            env["HF_TOKEN"] = self.cfg.hf_token
            env["HUGGING_FACE_HUB_TOKEN"] = self.cfg.hf_token
        return env

    async def _ensure_fixtures(
        self, job: Job, wt: Path, family: str, variant: str,
    ) -> None:
        """Run the family's Python sidecar to dump any missing fixtures.

        PR-scope (`variant != ""`): dump only that variant if its `.h5` is
        absent. Full sweep (`variant == ""`): pass `--all` to dump every
        variant the sidecar knows about, which the sidecar itself
        skips/overwrites as appropriate. `infra` has no sidecar.
        """
        sidecar = _FAMILY_SIDECAR.get(family)
        if not sidecar:
            return

        if variant:
            fixture = self.cfg.parity_dir / f"{variant}_io.h5"
            if fixture.exists():
                LOG.info("fixture %s already present", fixture.name)
                return
            args = ["--variant", variant]
        else:
            args = ["--all"]

        log_path = self.cfg.log_dir / job.head_sha / f"{family}-dump.log"
        cmd = ["uv", "run", "--project", str(wt),
               "python", sidecar, *args]
        env = self._env_for_sidecar()
        LOG.info("dumping parity fixtures: %s", " ".join(cmd))
        rc = await _run_subprocess(cmd, env=env, log_path=log_path, cwd=wt)
        if rc != 0:
            raise RuntimeError(
                f"parity dump for {family}/{variant or 'all'} failed (rc={rc}); "
                f"see {log_path}"
            )

    async def run(self, job: Job) -> None:
        LOG.info("starting job %s sha=%s families=%s sweep=%s",
                 job.label, job.head_sha, job.families, job.full_sweep)
        await self._ensure_mirror()
        wt = await self._worktree(job.head_sha)
        try:
            for family in job.families:
                variant = "" if job.full_sweep else REPRESENTATIVE_VARIANT.get(family, "")
                await self._run_family(job, wt, family, variant)
        finally:
            await self._drop_worktree(wt)

    async def _run_family(self, job: Job, wt: Path, family: str, variant: str) -> None:
        name = check_name(family, variant)
        log_rel = f"{job.head_sha}/{family}.log"
        log_path = self.cfg.log_dir / log_rel

        check = await self.app.create_check_run(
            self.cfg.repo_fullname, job.head_sha, name, status="in_progress",
        )
        job.check_runs[family] = check.id
        LOG.info("check_run %s id=%s", name, check.id)

        env = self._env_for_family(family, variant)
        cmd = [
            str(self.cfg.julia_binary), "--project=.", "-e",
            "using Pkg; Pkg.instantiate(); Pkg.test()",
        ]

        rc = 1
        try:
            await self._ensure_fixtures(job, wt, family, variant)
            rc = await _run_subprocess(cmd, env=env, log_path=log_path, cwd=wt)
        except asyncio.CancelledError:
            await self.app.complete_check_run(
                self.cfg.repo_fullname, check.id, conclusion="cancelled",
                output={"title": f"{family} cancelled",
                        "summary": "Build was cancelled.",
                        "text": _read_tail(log_path)},
            )
            raise
        except Exception as exc:
            LOG.exception("subprocess failed for %s", family)
            await self.app.complete_check_run(
                self.cfg.repo_fullname, check.id, conclusion="failure",
                output={"title": f"{family} errored",
                        "summary": f"Service-level error: {exc!s}",
                        "text": _read_tail(log_path)},
            )
            return

        conclusion = "success" if rc == 0 else "failure"
        await self.app.complete_check_run(
            self.cfg.repo_fullname, check.id, conclusion=conclusion,
            output={"title": f"{family} {'passed' if rc == 0 else 'failed'}",
                    "summary": f"Exit code {rc}. Variant: `{variant or 'all'}`. Sweep: `{job.full_sweep}`.",
                    "text": _read_tail(log_path)},
        )


def check_name(family: str, variant: str) -> str:
    return f"{CHECK_NAME_PREFIX}{family}" + (f" ({variant})" if variant else "")


def _has_completed_check(check_runs: list[dict], family: str) -> bool:
    """True if any completed `jimm-ci / <family>...` Check Run exists for the commit."""
    prefix = f"{CHECK_NAME_PREFIX}{family}"
    for run in check_runs:
        name = run.get("name", "")
        if not (name == prefix or name.startswith(prefix + " ")):
            continue
        if run.get("status") == "completed":
            return True
    return False


def _missing_families(
    families: tuple[str, ...], check_runs: list[dict],
) -> tuple[str, ...]:
    return tuple(f for f in families if not _has_completed_check(check_runs, f))


async def discover_jobs(cfg: Config, gh: GitHubApp) -> list[Job]:
    """Find every untested commit eligible to run on this invocation."""
    jobs: list[Job] = []

    # ── Open PRs (interactive y/n prompt happens in cli_main) ───────────
    pulls = await gh.list_open_pulls(cfg.repo_fullname)
    for pr in pulls:
        number = pr.get("number")
        head = pr.get("head", {}) or {}
        base = pr.get("base", {}) or {}
        head_sha = head.get("sha")
        base_sha = base.get("sha")
        if not head_sha or not base_sha or not isinstance(number, int):
            continue

        head_repo = (head.get("repo") or {}).get("full_name")
        base_repo = (base.get("repo") or {}).get("full_name")
        if head_repo and base_repo and head_repo != base_repo:
            LOG.info("PR #%s skipped: fork (%s)", number, head_repo)
            continue

        try:
            paths = await gh.compare(cfg.repo_fullname, base_sha, head_sha)
        except Exception:
            LOG.exception("PR #%s compare failed", number)
            continue
        families = families_for_paths(paths)
        if not families:
            LOG.info("PR #%s: no families touched, skipping", number)
            continue

        check_runs = await gh.list_check_runs(cfg.repo_fullname, head_sha)
        todo = _missing_families(families, check_runs)
        if not todo:
            LOG.info("PR #%s already tested (sha=%s)", number, head_sha[:8])
            continue
        jobs.append(Job(
            head_sha=head_sha, base_sha=base_sha, families=todo,
            full_sweep=False, pr_number=number,
            pr_title=pr.get("title") or "",
            label=f"pr-{number}@{head_sha[:8]}",
        ))

    # ── master commits in the last 30 days ──────────────────────────────
    since = (datetime.now(timezone.utc) - timedelta(days=MASTER_LOOKBACK_DAYS)) \
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        commits = await gh.list_commits(cfg.repo_fullname, sha="master", since=since)
    except Exception:
        LOG.exception("listing master commits failed")
        commits = []

    # newest-first from the API; process oldest-first so check-runs stack chronologically.
    for commit in reversed(commits):
        sha = commit.get("sha")
        if not isinstance(sha, str):
            continue
        parents = commit.get("parents") or []
        parent_sha = parents[0].get("sha") if parents else sha

        try:
            check_runs = await gh.list_check_runs(cfg.repo_fullname, sha)
        except Exception:
            LOG.exception("list_check_runs failed for %s", sha[:8])
            continue
        # Master commits run the full sweep, so any commit missing *any* family
        # is a candidate. We test all four families if any are missing rather
        # than partial, to keep behavior consistent with the old push handler.
        if all(_has_completed_check(check_runs, f) for f in ALL_FAMILIES):
            continue

        jobs.append(Job(
            head_sha=sha, base_sha=parent_sha, families=ALL_FAMILIES,
            full_sweep=True, label=f"master@{sha[:8]}",
        ))

    return jobs


def _print_jobs(jobs: list[Job]) -> None:
    if not jobs:
        print("no jobs to run")
        return
    for j in jobs:
        scope = "full-sweep" if j.full_sweep else "representative"
        print(f"{j.label}: sha={j.head_sha[:12]} families={','.join(j.families)} "
              f"scope={scope}"
              + (f" pr=#{j.pr_number}" if j.pr_number else ""))


async def _ask_yes_no(prompt: str) -> bool:
    """Synchronous-input y/n prompt that doesn't block the event loop."""
    loop = asyncio.get_running_loop()
    while True:
        answer = (await loop.run_in_executor(None, input, prompt)).strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no", ""):
            return False
        print("  please answer y or n", flush=True)


async def _post_skipped(
    gh: GitHubApp, repo: str, sha: str, families: tuple[str, ...],
) -> None:
    today = date.today().isoformat()
    output = {
        "title": "Skipped",
        "summary": (
            f"Cancelled by `jimm-ci-run` prompt on {today}. "
            "Push a new commit on the PR to re-prompt."
        ),
    }
    for family in families:
        await gh.create_check_run(
            repo, sha, check_name(family, ""),
            status="completed", conclusion="skipped", output=output,
        )


async def _confirm_pr_jobs(
    gh: GitHubApp, cfg: Config, jobs: list[Job],
) -> list[Job]:
    """Prompt y/n on each PR job. `n` posts skipped check runs and drops the job."""
    if not sys.stdin.isatty():
        raise RuntimeError(
            "jimm-ci-run needs a TTY to prompt for PR confirmation; "
            "run it from an interactive shell."
        )
    kept: list[Job] = []
    pr_jobs = [j for j in jobs if j.pr_number is not None]
    other_jobs = [j for j in jobs if j.pr_number is None]
    if not pr_jobs:
        return jobs
    print(f"\n{len(pr_jobs)} pull request(s) waiting on confirmation:\n",
          flush=True)
    for job in pr_jobs:
        title = job.pr_title or "(no title)"
        print(f"  PR #{job.pr_number}: {title}")
        print(f"    head:     {job.head_sha[:12]}")
        print(f"    families: {','.join(job.families)}")
        prompt = "    run? [y/N]: "
        if await _ask_yes_no(prompt):
            kept.append(job)
            print("    → queued", flush=True)
        else:
            print("    → cancelled (posting skipped check runs)", flush=True)
            try:
                await _post_skipped(
                    gh, cfg.repo_fullname, job.head_sha, job.families,
                )
            except Exception:
                LOG.exception("PR #%s: posting skipped check runs failed",
                              job.pr_number)
        print(flush=True)
    return kept + other_jobs


async def _run_async(args: argparse.Namespace) -> int:
    cfg = Config.from_env()
    gh = GitHubApp(cfg.app_id, cfg.installation_id, cfg.private_key)
    try:
        jobs = await discover_jobs(cfg, gh)
        if args.dry_run:
            _print_jobs(jobs)
            return 0
        if not jobs:
            print("no jobs to run")
            return 0
        jobs = await _confirm_pr_jobs(gh, cfg, jobs)
        if not jobs:
            print("no jobs left after confirmation")
            return 0
        builder = Builder(cfg, gh)
        for job in jobs:
            try:
                await builder.run(job)
            except Exception:
                LOG.exception("job %s failed at builder level", job.label)
        return 0
    finally:
        await gh.aclose()


def _configure_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("JIMM_CI_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def cli_main() -> None:
    parser = argparse.ArgumentParser(
        prog="jimm-ci-run",
        description="Run Jimm.jl CI for untested approved commits.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the jobs that would be executed and exit without running them",
    )
    args = parser.parse_args()
    _configure_logging()
    raise SystemExit(asyncio.run(_run_async(args)))
