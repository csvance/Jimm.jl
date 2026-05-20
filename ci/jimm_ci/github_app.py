"""Minimal GitHub App client.

Handles JWT minting, installation token caching, the small slice of the
Checks API we need, and the Compare API for path-filter input. Uses httpx
for async I/O and PyJWT for RS256 signing. No other GitHub libraries.
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

    async def _app_request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Request authenticated as the App itself (JWT), not as an installation.

        Used by webhook-delivery endpoints (`/app/hook/deliveries`) which are
        scoped to the app, not to a specific installation. `url` may be a full
        URL (used to follow Link-header pagination) or an API path.
        """
        if not url.startswith("http"):
            url = f"{API}{url}"
        app_jwt = self._mint_jwt()
        headers = kwargs.pop("headers", {})
        headers.setdefault("Authorization", f"Bearer {app_jwt}")
        headers.setdefault("Accept", "application/vnd.github+json")
        headers.setdefault("X-GitHub-Api-Version", "2022-11-28")
        r = await self._client.request(method, url, headers=headers, **kwargs)
        r.raise_for_status()
        return r

    async def create_check_run(
        self,
        repo: str,
        head_sha: str,
        name: str,
        *,
        status: str = "in_progress",
        details_url: str | None = None,
        output: dict[str, Any] | None = None,
    ) -> CheckRun:
        body: dict[str, Any] = {"name": name, "head_sha": head_sha, "status": status}
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

    async def iter_deliveries(
        self, *, per_page: int = 100, max_pages: int = 20,
    ) -> list[dict[str, Any]]:
        """Return webhook delivery summaries newest-first across `max_pages` pages.

        Each entry contains id, guid, delivered_at, event, action, status_code,
        and redelivery flag. The full payload is not included; that requires a
        per-delivery GET, which we never need because we re-trigger delivery via
        the attempts endpoint instead.
        """
        out: list[dict[str, Any]] = []
        url: str | None = f"/app/hook/deliveries?per_page={per_page}"
        pages = 0
        while url and pages < max_pages:
            r = await self._app_request("GET", url)
            page = r.json()
            if not isinstance(page, list):
                break
            out.extend(page)
            next_link = r.links.get("next") if hasattr(r, "links") else None
            url = next_link["url"] if next_link else None
            pages += 1
        return out

    async def redeliver(self, delivery_id: int) -> None:
        await self._app_request(
            "POST", f"/app/hook/deliveries/{delivery_id}/attempts",
        )
