"""FastAPI webhook listener and build worker for Jimm.jl CI."""
from __future__ import annotations

import asyncio
import hmac
import logging
import os
import shutil
import signal
import subprocess
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse

from .config import Config
from .github_app import GitHubApp
from .path_filter import REPRESENTATIVE_VARIANT, families_for_paths
from .state import State

LOG = logging.getLogger("jimm_ci")
MAX_OUTPUT_TEXT = 60_000  # GitHub Checks API limit is 65535; leave headroom.


@dataclass
class Job:
    head_sha: str
    base_sha: str
    families: tuple[str, ...]
    full_sweep: bool
    label: str
    delivery_guid: str | None = None
    pr_number: int | None = None
    check_runs: dict[str, int] = field(default_factory=dict)


def verify_signature(secret: bytes, body: bytes, header: str | None) -> bool:
    if not header or not header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(secret, body, sha256).hexdigest()
    return hmac.compare_digest(expected, header)


def _read_tail(log_path: Path, *, limit: int = MAX_OUTPUT_TEXT) -> str:
    if not log_path.exists():
        return ""
    data = log_path.read_bytes()
    if len(data) <= limit:
        return data.decode("utf-8", errors="replace")
    head = b"[... log truncated; full log at details_url ...]\n"
    return (head + data[-(limit - len(head)):]).decode("utf-8", errors="replace")


async def _run_subprocess(cmd: list[str], *, env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        log.write(f"$ {' '.join(cmd)}\n".encode())
        log.flush()
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
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
        return wt

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
        name = f"jimm-ci / {family}" + (f" ({variant})" if variant else "")
        log_rel = f"{job.head_sha}/{family}.log"
        log_path = self.cfg.log_dir / log_rel
        details_url = f"{self.cfg.public_base_url.rstrip('/')}/logs/{log_rel}"

        check = await self.app.create_check_run(
            self.cfg.repo_fullname, job.head_sha, name,
            status="in_progress", details_url=details_url,
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
            rc = await _run_subprocess(cmd, env=env, log_path=log_path)
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


class App:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.gh = GitHubApp(cfg.app_id, cfg.installation_id, cfg.private_key)
        self.builder = Builder(cfg, self.gh)
        self.queue: asyncio.Queue[Job] = asyncio.Queue()
        self.worker_task: asyncio.Task[None] | None = None
        self.catchup_task: asyncio.Task[None] | None = None
        self.state = State(cfg.state_dir / "completed.json")
        # Track GUIDs currently in queue or being built so a redelivery that
        # arrives before the original finishes does not start a duplicate.
        self._in_flight: set[str] = set()

    async def start(self, *, catch_up: bool = True) -> None:
        self.worker_task = asyncio.create_task(self._worker(), name="jimm-ci-worker")
        if catch_up:
            self.catchup_task = asyncio.create_task(
                self._catch_up(), name="jimm-ci-catchup",
            )

    async def stop(self) -> None:
        for t in (self.catchup_task, self.worker_task):
            if t and not t.done():
                t.cancel()
        for t in (self.catchup_task, self.worker_task):
            if t:
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        await self.gh.aclose()

    async def _worker(self) -> None:
        while True:
            job = await self.queue.get()
            try:
                await self.builder.run(job)
            except asyncio.CancelledError:
                LOG.info("worker cancelled mid-job %s", job.label)
                raise
            except Exception:
                LOG.exception("job %s failed at builder level", job.label)
            finally:
                if job.delivery_guid:
                    self.state.mark_completed(job.delivery_guid)
                    self._in_flight.discard(job.delivery_guid)
                self.queue.task_done()

    def is_handled(self, guid: str) -> bool:
        """True if this delivery is already completed or currently in flight."""
        return self.state.is_completed(guid) or guid in self._in_flight

    async def _catch_up(self) -> None:
        """On startup, ask GitHub to redeliver any webhook we missed."""
        try:
            deliveries = await self.gh.iter_deliveries()
        except Exception:
            LOG.exception("catch-up: list deliveries failed; skipping")
            return
        if not deliveries:
            LOG.info("catch-up: no deliveries returned by GitHub")
            return
        wanted_events = {"pull_request", "push"}
        requested = 0
        for d in deliveries:
            guid = d.get("guid")
            event = d.get("event")
            did = d.get("id")
            if not guid or event not in wanted_events or not isinstance(did, int):
                continue
            if self.is_handled(guid):
                continue
            try:
                await self.gh.redeliver(did)
                requested += 1
                LOG.info("catch-up: requested redelivery event=%s id=%s guid=%s",
                         event, did, guid)
            except Exception:
                LOG.exception("catch-up: redelivery failed id=%s", did)
        LOG.info("catch-up: scanned %d deliveries, requested %d redeliveries",
                 len(deliveries), requested)

    async def enqueue_pull_request(
        self, payload: dict[str, Any], delivery_guid: str | None,
    ) -> str:
        pr = payload["pull_request"]
        action = payload["action"]
        if action not in {"opened", "synchronize", "reopened"}:
            return f"ignored action={action}"

        head_repo = pr["head"]["repo"]["full_name"]
        base_repo = pr["base"]["repo"]["full_name"]
        head_sha = pr["head"]["sha"]
        base_sha = pr["base"]["sha"]
        pr_number = pr["number"]

        if head_repo != base_repo:
            await self._post_fork_skip(head_sha, pr_number)
            return f"fork PR #{pr_number} skipped"

        paths = await self.gh.compare(self.cfg.repo_fullname, base_sha, head_sha)
        families = families_for_paths(paths)
        if not families:
            LOG.info("PR #%s: no families touched, skipping", pr_number)
            return f"no families touched"
        job = Job(
            head_sha=head_sha, base_sha=base_sha, families=families,
            full_sweep=False, pr_number=pr_number,
            label=f"pr-{pr_number}@{head_sha[:8]}", delivery_guid=delivery_guid,
        )
        if delivery_guid:
            self._in_flight.add(delivery_guid)
        await self.queue.put(job)
        return f"queued pr-{pr_number} families={families}"

    async def enqueue_push(
        self, payload: dict[str, Any], delivery_guid: str | None,
    ) -> str:
        ref = payload.get("ref", "")
        if ref != "refs/heads/master":
            return f"ignored ref={ref}"
        head_sha = payload["after"]
        base_sha = payload["before"]
        if base_sha == "0000000000000000000000000000000000000000":
            families = ("infra", "bit", "convnext", "convnextv2")
        else:
            paths = await self.gh.compare(self.cfg.repo_fullname, base_sha, head_sha)
            families = families_for_paths(paths)
        if not families:
            LOG.info("push %s: no families touched, skipping", head_sha[:8])
            return f"no families touched"
        job = Job(
            head_sha=head_sha, base_sha=base_sha, families=families,
            full_sweep=True, label=f"master@{head_sha[:8]}",
            delivery_guid=delivery_guid,
        )
        if delivery_guid:
            self._in_flight.add(delivery_guid)
        await self.queue.put(job)
        return f"queued push families={families}"

    async def _post_fork_skip(self, head_sha: str, pr_number: int) -> None:
        try:
            await self.gh.create_check_run(
                self.cfg.repo_fullname, head_sha, "jimm-ci / external-pr",
                status="completed", output={
                    "title": "External PR: manual review required",
                    "summary": (
                        f"PR #{pr_number} is from a fork. CI is skipped automatically. "
                        "A maintainer must push the branch to the main repository "
                        "(or use the future /ci-run comment trigger) to run tests."
                    ),
                },
            )
        except httpx.HTTPError:
            LOG.exception("failed to post external-pr check")


def build_app(cfg: Config) -> FastAPI:
    state = App(cfg)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await state.start()
        try:
            yield
        finally:
            await state.stop()

    api = FastAPI(lifespan=lifespan)

    @api.get("/health", response_class=PlainTextResponse)
    async def health() -> str:
        return "ok\n"

    @api.post("/webhook")
    async def webhook(
        request: Request,
        x_github_event: str = Header(default=""),
        x_hub_signature_256: str | None = Header(default=None),
        x_github_delivery: str | None = Header(default=None),
    ) -> Response:
        body = await request.body()
        if not verify_signature(cfg.webhook_secret, body, x_hub_signature_256):
            raise HTTPException(status_code=401, detail="bad signature")
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"bad json: {exc!s}")

        if x_github_event == "ping":
            return PlainTextResponse("pong\n")

        if x_github_delivery and state.is_handled(x_github_delivery):
            return PlainTextResponse(f"already handled {x_github_delivery}\n")

        if x_github_event == "pull_request":
            result = await state.enqueue_pull_request(payload, x_github_delivery)
        elif x_github_event == "push":
            result = await state.enqueue_push(payload, x_github_delivery)
        else:
            result = f"ignored event={x_github_event}"
        return PlainTextResponse(result + "\n")

    @api.get("/logs/{sha}/{family}.log", response_class=PlainTextResponse)
    async def get_log(sha: str, family: str) -> PlainTextResponse:
        # Defend against path traversal even though FastAPI's path matching
        # already disallows slashes in single segments.
        if "/" in sha or "/" in family or ".." in sha or ".." in family:
            raise HTTPException(status_code=400, detail="bad path")
        path = cfg.log_dir / sha / f"{family}.log"
        if not path.exists():
            raise HTTPException(status_code=404, detail="no such log")
        return PlainTextResponse(path.read_text(errors="replace"))

    return api


def _configure_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("JIMM_CI_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> None:
    _configure_logging()
    cfg = Config.from_env()
    uvicorn.run(
        build_app(cfg), host=cfg.listen_host, port=cfg.listen_port,
        log_config=None, access_log=False,
    )


async def _nightly_async() -> None:
    cfg = Config.from_env()
    app = App(cfg)
    await app.start(catch_up=False)
    try:
        head = await app.gh.get_default_branch_head(cfg.repo_fullname, "master")
        job = Job(
            head_sha=head, base_sha=head,
            families=("infra", "bit", "convnext", "convnextv2"),
            full_sweep=True, label=f"nightly@{head[:8]}",
        )
        await app.queue.put(job)
        await app.queue.join()
    finally:
        await app.stop()


def nightly_main() -> None:
    _configure_logging()
    asyncio.run(_nightly_async())
