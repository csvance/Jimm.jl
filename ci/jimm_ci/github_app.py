"""Minimal GitHub App client.

Handles JWT minting, installation token caching, the slice of the Checks
API we need, the Compare API for path-filter input, and small helpers for
listing PRs / commits / check-runs and applying labels. Uses httpx for
async I/O and PyJWT for RS256 signing. No other GitHub libraries.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx
import jwt

LOG = logging.getLogger(__name__)
API = "https://api.github.com"


@dataclass
class CheckRun:
    id: int
    name: str
    head_sha: str
    html_url: str


class GitHubApp:
    def __init__(
        self,
        app_id: int,
        installation_id: int,
        private_key: str,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.app_id = app_id
        self.installation_id = installation_id
        self.private_key = private_key
        self._client = client or httpx.AsyncClient(timeout=30.0)
        self._token: str | None = None
        self._token_exp: float = 0.0

    async def aclose(self) -> None:
        await self._client.aclose()

    def _mint_jwt(self) -> str:
        now = int(time.time())
        return jwt.encode(
            {"iat": now - 60, "exp": now + 540, "iss": str(self.app_id)},
            self.private_key,
            algorithm="RS256",
        )

    async def installation_token(self) -> str:
        now = time.time()
        if self._token and now < self._token_exp - 300:
            return self._token
        app_jwt = self._mint_jwt()
        r = await self._client.post(
            f"{API}/app/installations/{self.installation_id}/access_tokens",
            headers={
                "Authorization": f"Bearer {app_jwt}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        r.raise_for_status()
        data = r.json()
        self._token = data["token"]
        self._token_exp = now + 3300
        return self._token

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        token = await self.installation_token()
        headers = kwargs.pop("headers", {})
        headers.setdefault("Authorization", f"Bearer {token}")
        headers.setdefault("Accept", "application/vnd.github+json")
        headers.setdefault("X-GitHub-Api-Version", "2022-11-28")
        r = await self._client.request(method, f"{API}{path}", headers=headers, **kwargs)
        if r.status_code == 401:
            self._token = None
            self._token_exp = 0
            token = await self.installation_token()
            headers["Authorization"] = f"Bearer {token}"
            r = await self._client.request(method, f"{API}{path}", headers=headers, **kwargs)
        r.raise_for_status()
        return r

    async def _paginated(self, path: str, *, per_page: int = 100,
                         max_pages: int = 20) -> list[dict[str, Any]]:
        sep = "&" if "?" in path else "?"
        url: str | None = f"{path}{sep}per_page={per_page}"
        out: list[dict[str, Any]] = []
        pages = 0
        while url and pages < max_pages:
            if url.startswith("http"):
                token = await self.installation_token()
                r = await self._client.get(
                    url,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/vnd.github+json",
                        "X-GitHub-Api-Version": "2022-11-28",
                    },
                )
                r.raise_for_status()
            else:
                r = await self._request("GET", url)
            page = r.json()
            if not isinstance(page, list):
                break
            out.extend(page)
            next_link = r.links.get("next") if hasattr(r, "links") else None
            url = next_link["url"] if next_link else None
            pages += 1
        return out

    async def create_check_run(
        self,
        repo: str,
        head_sha: str,
        name: str,
        *,
        status: str = "in_progress",
        conclusion: str | None = None,
        details_url: str | None = None,
        output: dict[str, Any] | None = None,
    ) -> CheckRun:
        body: dict[str, Any] = {"name": name, "head_sha": head_sha, "status": status}
        if conclusion is not None:
            body["conclusion"] = conclusion
        if details_url is not None:
            body["details_url"] = details_url
        if output is not None:
            body["output"] = output
        r = await self._request("POST", f"/repos/{repo}/check-runs", json=body)
        data = r.json()
        return CheckRun(
            id=data["id"], name=data["name"], head_sha=data["head_sha"],
            html_url=data["html_url"],
        )

    async def complete_check_run(
        self,
        repo: str,
        check_run_id: int,
        *,
        conclusion: str,
        output: dict[str, Any] | None = None,
    ) -> None:
        body: dict[str, Any] = {"status": "completed", "conclusion": conclusion}
        if output is not None:
            body["output"] = output
        await self._request("PATCH", f"/repos/{repo}/check-runs/{check_run_id}", json=body)

    async def compare(self, repo: str, base: str, head: str) -> list[str]:
        """Return the list of changed file paths between base and head."""
        r = await self._request("GET", f"/repos/{repo}/compare/{base}...{head}")
        data = r.json()
        return [f["filename"] for f in data.get("files", [])]

    async def get_default_branch_head(self, repo: str, branch: str = "master") -> str:
        r = await self._request("GET", f"/repos/{repo}/branches/{branch}")
        return r.json()["commit"]["sha"]

    async def list_open_pulls(self, repo: str) -> list[dict[str, Any]]:
        """List open PRs with head/base, labels, and fork status."""
        return await self._paginated(f"/repos/{repo}/pulls?state=open")

    async def list_commits(self, repo: str, *, sha: str, since: str) -> list[dict[str, Any]]:
        """List commits on `sha` (branch or commit) committed at or after `since`.

        `since` must be an ISO-8601 timestamp (e.g. `2026-04-20T00:00:00Z`).
        """
        return await self._paginated(
            f"/repos/{repo}/commits?sha={sha}&since={since}",
        )

    async def list_check_runs(self, repo: str, ref: str) -> list[dict[str, Any]]:
        """Return all check runs for `ref`. Paginated under `check_runs`."""
        out: list[dict[str, Any]] = []
        per_page = 100
        page = 1
        while True:
            r = await self._request(
                "GET",
                f"/repos/{repo}/commits/{ref}/check-runs"
                f"?per_page={per_page}&page={page}",
            )
            data = r.json()
            runs = data.get("check_runs", []) if isinstance(data, dict) else []
            out.extend(runs)
            if len(runs) < per_page:
                return out
            page += 1
            if page > 20:
                return out

