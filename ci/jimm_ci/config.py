from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    app_id: int
    installation_id: int
    private_key: str
    webhook_secret: bytes

    repo_owner: str
    repo_name: str

    state_dir: Path
    julia_depot: Path
    hf_cache: Path
    mirror_dir: Path
    workspace_dir: Path
    log_dir: Path

    julia_binary: Path
    hf_token: str | None

    listen_host: str
    listen_port: int
    public_base_url: str

    @property
    def repo_fullname(self) -> str:
        return f"{self.repo_owner}/{self.repo_name}"

    @classmethod
    def from_env(cls) -> Config:
        state = Path(os.environ.get("JIMM_CI_STATE_DIR", "/var/lib/jimm-ci"))
        secret_path = Path(os.environ["JIMM_CI_WEBHOOK_SECRET_FILE"])
        key_path = Path(os.environ["JIMM_CI_PRIVATE_KEY_FILE"])

        hf_token = os.environ.get("HF_TOKEN")
        hf_token_file = os.environ.get("JIMM_CI_HF_TOKEN_FILE")
        if hf_token is None and hf_token_file:
            hf_token = Path(hf_token_file).read_text().strip()

        return cls(
            app_id=int(os.environ["JIMM_CI_APP_ID"]),
            installation_id=int(os.environ["JIMM_CI_INSTALLATION_ID"]),
            private_key=key_path.read_text(),
            webhook_secret=secret_path.read_bytes().strip(),
            repo_owner=os.environ["JIMM_CI_REPO_OWNER"],
            repo_name=os.environ["JIMM_CI_REPO_NAME"],
            state_dir=state,
            julia_depot=Path(os.environ.get("JULIA_DEPOT_PATH", state / "julia-depot")),
            hf_cache=Path(os.environ.get("HF_HUB_CACHE", state / "hf-cache")),
            mirror_dir=state / "mirror.git",
            workspace_dir=state / "work",
            log_dir=state / "logs",
            julia_binary=Path(os.environ.get("JIMM_CI_JULIA", "/usr/local/bin/julia")),
            hf_token=hf_token,
            listen_host=os.environ.get("JIMM_CI_HOST", "127.0.0.1"),
            listen_port=int(os.environ.get("JIMM_CI_PORT", "8080")),
            public_base_url=os.environ.get(
                "JIMM_CI_PUBLIC_URL", "https://jimm-ci.invalid"
            ),
        )
