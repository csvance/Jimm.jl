# Jimm.jl self-hosted CI

`jimm-ci` is a small FastAPI service that runs the Jimm.jl parity test suite on
a self-hosted Linux VM in response to GitHub events. It is intentionally
narrower than a general-purpose runner: it understands only the Jimm.jl
repository layout, the four test families defined in `path_filter.py`, and the
Checks API surface needed to report results back to GitHub.

The service is **driven by GitHub App webhooks**, not by GitHub Actions. A
GitHub App installed on the repo posts `pull_request` and `push` events to
`/webhook`; on restart the service queries the App's webhook delivery history
and asks GitHub to redeliver anything that was missed while the VM was offline.

```
                ┌───────────────────────────────────────────────┐
                │ GitHub                                        │
                │   ├── App webhooks (pull_request, push)       │
                │   ├── Checks API (results posted here)        │
                │   └── Compare + Deliveries APIs               │
                └─────────────────────┬─────────────────────────┘
                                      │  HTTPS
                          ┌───────────▼────────────┐
                          │ Cloudflare Tunnel       │
                          │ (jimm-ci.<domain>)      │
                          └───────────┬────────────┘
                                      │  http://127.0.0.1:8080
                          ┌───────────▼────────────┐
                          │ uvicorn + FastAPI       │
                          │  POST /webhook          │
                          │  GET  /health           │
                          │  GET  /logs/<sha>/...   │
                          └───────────┬────────────┘
                                      │  asyncio.Queue
                          ┌───────────▼────────────┐
                          │ Builder                 │
                          │  git mirror + worktree  │
                          │  julia Pkg.test         │
                          │  → Checks API           │
                          └─────────────────────────┘
```

## Layout

```
ci/
├── pyproject.toml          # uv-managed Python project, entry points
├── jimm_ci/
│   ├── server.py           # FastAPI app, worker queue, Builder
│   ├── github_app.py       # JWT + installation token + Checks/Compare/Deliveries
│   ├── path_filter.py      # changed paths → test families
│   ├── state.py            # durable record of handled delivery GUIDs
│   └── config.py           # env-var-driven configuration
└── systemd/
    ├── jimm-ci.service           # listener
    ├── jimm-ci-nightly.service   # nightly full sweep
    ├── jimm-ci-nightly.timer     # 07:00 daily
    └── cloudflared.service       # public ingress
```

## Architecture

### `server.py`

The FastAPI app exposes three routes:

* `POST /webhook` verifies the `X-Hub-Signature-256` HMAC against
  `JIMM_CI_WEBHOOK_SECRET_FILE`, parses the JSON body, deduplicates by
  `X-GitHub-Delivery`, and either enqueues a `Job` or returns an `ignored`
  message. `ping` events return `pong`.
* `GET /health` returns `ok` for the tunnel healthcheck.
* `GET /logs/{sha}/{family}.log` serves the raw build log so that the
  `details_url` on each Check Run resolves to readable output.

A single background worker drains the queue serially. For each job it:

1. Ensures a bare mirror exists at `/var/lib/jimm-ci/mirror.git` and runs
   `git fetch --prune` to pull the latest refs.
2. Creates a detached `git worktree` at `/var/lib/jimm-ci/work/<sha>`.
3. For each family in the job, posts an `in_progress` Check Run, runs
   `julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'` with
   `JIMM_TEST_FAMILIES` / `JIMM_TEST_VARIANTS` set, and completes the Check
   Run with the tail of the log (capped at ~60 KB to stay under the GitHub
   limit).
4. Removes the worktree.

### `github_app.py`

Minimal async client built on `httpx` and `pyjwt`. Mints an RS256 JWT from the
private key, exchanges it for an installation access token (cached until ~5
minutes before expiry), and uses that token for repo-scoped calls (Checks,
Compare). Webhook-delivery endpoints (`/app/hook/deliveries`) are App-scoped,
so they sign with the JWT directly via `_app_request`.

### `path_filter.py`

Maps a list of changed paths to a tuple of test families. Families are
`infra`, `bit`, `convnext`, `convnextv2`. Paths under `src/Layers/`,
`src/Interop/`, `ci/`, or top-level files like `src/Jimm.jl`, `Project.toml`,
`Manifest.toml`, and `test/runtests.jl` are treated as **shared** and promote
to all families. `REPRESENTATIVE_VARIANT` picks a single variant per family
for PR-scope runs; pushes to `master` and the nightly sweep run the full
variant set instead.

When adding a new family, update both `_FAMILY_PREFIXES` / `_FAMILY_EXACT` /
`ALL_FAMILIES` / `REPRESENTATIVE_VARIANT` here **and** the root `CLAUDE.md`
checklist, which calls this file out explicitly.

### `state.py`

A small JSON-on-disk record at `/var/lib/jimm-ci/completed.json` containing
the webhook delivery GUIDs that have been fully processed. It backs two
properties:

* **Deduplication.** If GitHub redelivers a webhook (manually or as part of
  the startup catch-up), the same delivery cannot trigger a duplicate build.
* **Startup catch-up.** On boot the service calls `iter_deliveries` to list
  the App's recent webhook deliveries newest-first, then `redeliver`s any
  `pull_request` / `push` event whose GUID is not in `completed.json`. GitHub
  retains delivery history for 30 days; a VM offline longer than that simply
  loses anything older.

History is capped at 2,000 entries to avoid unbounded growth. Writes are
atomic via `tmpfile + os.replace`.

### `config.py`

All configuration is environment-driven. Secrets are loaded from files
rather than env strings so that systemd's `Environment=` lines remain
non-sensitive. Defaults assume `/var/lib/jimm-ci/` as the state root and
`/usr/local/bin/julia` as the Julia binary.

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `JIMM_CI_APP_ID` | yes | — | GitHub App ID |
| `JIMM_CI_INSTALLATION_ID` | yes | — | App installation on the repo |
| `JIMM_CI_PRIVATE_KEY_FILE` | yes | — | Path to App PEM |
| `JIMM_CI_WEBHOOK_SECRET_FILE` | yes | — | Path to webhook shared secret |
| `JIMM_CI_REPO_OWNER` | yes | — | e.g. `cvance` |
| `JIMM_CI_REPO_NAME` | yes | — | e.g. `Jimm.jl` |
| `JIMM_CI_STATE_DIR` | no | `/var/lib/jimm-ci` | Mirror, worktrees, logs, depot |
| `JIMM_CI_HF_TOKEN_FILE` | no | — | HuggingFace token for parity weights |
| `JIMM_CI_PUBLIC_URL` | no | `https://jimm-ci.invalid` | Used to build `details_url` |
| `JIMM_CI_HOST` | no | `127.0.0.1` | Listener bind address |
| `JIMM_CI_PORT` | no | `8080` | Listener port |
| `JIMM_CI_JULIA` | no | `/usr/local/bin/julia` | Julia binary path |
| `JULIA_NUM_THREADS` | no | `4` | Forwarded to test jobs |

## Job lifecycle

**Pull request events.** Only `opened`, `synchronize`, and `reopened` enqueue
a job. The service calls the GitHub Compare API for the PR's base...head to
list changed paths, feeds them through `families_for_paths`, and runs each
touched family with its `REPRESENTATIVE_VARIANT`. Pull requests opened from
**forks** are skipped automatically: the service posts a single
`jimm-ci / external-pr` check explaining that a maintainer must push the
branch to the main repository to run tests.

**Push events.** Only pushes to `refs/heads/master` enqueue a job. The
changed-path list comes from comparing `before...after`. Master pushes run as
a **full sweep**: every variant in every touched family, with no
`REPRESENTATIVE_VARIANT` filter.

**Nightly.** `jimm-ci-nightly.timer` fires daily at 07:00 (with a 15-minute
randomized delay). The oneshot service calls `nightly_main`, which queries
the current `master` HEAD and runs the full sweep across all four families,
then exits. The bare mirror and Julia depot are shared safely with the live
listener since `git fetch` is atomic and `Pkg.test` is read-only on the depot.

**Per-family Check Runs.** Each family produces its own Check Run named
`jimm-ci / <family>` (or `jimm-ci / <family> (<variant>)` for PR-scope
runs). The `details_url` points at the streaming log served by the service.

## Setup on Debian 13

These steps assume a fresh KVM Debian 13 VM with internet access and the
ability to point a DNS name at it via Cloudflare. Run everything as `root`
unless noted.

### 1. System packages and `gh-runner` user

```bash
apt-get update
apt-get install -y --no-install-recommends \
    build-essential git ca-certificates curl jq tar xz-utils \
    python3 python3-venv

adduser --system --group --home /home/gh-runner --shell /bin/bash gh-runner
install -d -o gh-runner -g gh-runner /home/gh-runner/.cloudflared
```

### 2. Install `uv` (Python project runner)

`uv` is invoked from the systemd unit via `uv run --project ...`, so it must
be on the default `PATH` for the `gh-runner` user. The simplest path is the
official installer:

```bash
sudo -u gh-runner bash -lc 'curl -LsSf https://astral.sh/uv/install.sh | sh'
ln -sf /home/gh-runner/.local/bin/uv /usr/local/bin/uv
```

### 3. Install Julia

`JIMM_CI_JULIA` defaults to `/usr/local/bin/julia`; install a matching
official build:

```bash
JULIA_VER=1.11.2   # adjust to match Project.toml compat
cd /tmp
curl -fLO "https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_VER%.*}/julia-${JULIA_VER}-linux-x86_64.tar.gz"
tar -C /opt -xzf "julia-${JULIA_VER}-linux-x86_64.tar.gz"
ln -sf "/opt/julia-${JULIA_VER}/bin/julia" /usr/local/bin/julia
```

### 4. Clone the repository and create state directories

```bash
install -d -o gh-runner -g gh-runner /opt/jimm-ci
sudo -u gh-runner git clone https://github.com/<OWNER>/Jimm.jl.git /opt/jimm-ci/Jimm.jl

install -d -o gh-runner -g gh-runner \
    /var/lib/jimm-ci \
    /var/lib/jimm-ci/julia-depot \
    /var/lib/jimm-ci/hf-cache
```

The service creates `mirror.git`, `work/`, and `logs/` under
`/var/lib/jimm-ci/` on first run.

### 5. Register the GitHub App

In the GitHub UI (Settings → Developer settings → GitHub Apps → New GitHub
App):

* **Homepage URL:** anything (the repo URL is fine).
* **Webhook URL:** `https://jimm-ci.<your-domain>/webhook` (the hostname you
  will route through Cloudflare in step 7).
* **Webhook secret:** generate a long random string. Keep it; you will write
  it to disk in step 6.
* **Permissions (repository):**
  * Checks: **Read & write**
  * Contents: **Read-only**
  * Metadata: **Read-only**
  * Pull requests: **Read-only**
* **Subscribe to events:** Pull request, Push.
* **Where can this app be installed:** Only on this account.

After creation:

1. Click **Generate a private key**; save the downloaded `.pem`.
2. Note the **App ID** from the App settings page.
3. **Install** the App on the target repository.
4. Open the App's **Install App → Configure** view; the URL contains the
   `installation_id`. (Or fetch it via the API: `GET
   /repos/{owner}/{repo}/installation`.)

### 6. Place secrets on disk

```bash
install -d -m 0750 -o root -g gh-runner /etc/jimm-ci

# App private key
install -m 0640 -o root -g gh-runner /path/to/jimm-ci.<id>.private-key.pem \
    /etc/jimm-ci/app.pem

# Webhook secret (no trailing newline — the loader does .strip())
printf '%s' '<the-secret-you-set-on-the-app>' > /etc/jimm-ci/webhook-secret
chmod 0640 /etc/jimm-ci/webhook-secret
chown root:gh-runner /etc/jimm-ci/webhook-secret

# HuggingFace token used by parity tests to fetch weights
printf '%s' '<hf_xxx...>' > /etc/jimm-ci/hf-token
chmod 0640 /etc/jimm-ci/hf-token
chown root:gh-runner /etc/jimm-ci/hf-token
```

### 7. Configure the Cloudflare Tunnel

Install `cloudflared` from the official repository, then create the tunnel
as the `gh-runner` user so the credentials file lands in
`/home/gh-runner/.cloudflared/`:

```bash
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg \
    -o /usr/share/keyrings/cloudflare-main.gpg
echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared bookworm main' \
    > /etc/apt/sources.list.d/cloudflared.list
apt-get update
apt-get install -y cloudflared

sudo -u gh-runner cloudflared tunnel login          # opens browser-free auth flow
sudo -u gh-runner cloudflared tunnel create jimm-ci
sudo -u gh-runner cloudflared tunnel route dns jimm-ci jimm-ci.<your-domain>
```

Write the tunnel config (replace `<tunnel-uuid>` with the value printed by
`tunnel create`):

```bash
cat > /home/gh-runner/.cloudflared/config.yml <<'YAML'
tunnel: <tunnel-uuid>
credentials-file: /home/gh-runner/.cloudflared/<tunnel-uuid>.json

ingress:
  - hostname: jimm-ci.<your-domain>
    service: http://127.0.0.1:8080
  - service: http_status:404
YAML
chown gh-runner:gh-runner /home/gh-runner/.cloudflared/config.yml
```

### 8. Install systemd units

The three units committed under `ci/systemd/` contain `REPLACE_WITH_*`
placeholders. Copy them into place and fill in the values you collected in
step 5:

```bash
cp /opt/jimm-ci/Jimm.jl/ci/systemd/jimm-ci.service          /etc/systemd/system/
cp /opt/jimm-ci/Jimm.jl/ci/systemd/jimm-ci-nightly.service  /etc/systemd/system/
cp /opt/jimm-ci/Jimm.jl/ci/systemd/jimm-ci-nightly.timer    /etc/systemd/system/
cp /opt/jimm-ci/Jimm.jl/ci/systemd/cloudflared.service      /etc/systemd/system/

sed -i \
    -e 's/REPLACE_WITH_APP_ID/123456/' \
    -e 's/REPLACE_WITH_INSTALLATION_ID/7890123/' \
    -e 's/REPLACE_WITH_OWNER/<owner>/' \
    -e 's/REPLACE_WITH_DOMAIN/<your-domain>/' \
    /etc/systemd/system/jimm-ci.service \
    /etc/systemd/system/jimm-ci-nightly.service

systemctl daemon-reload
systemctl enable --now jimm-ci.service
systemctl enable --now cloudflared.service
systemctl enable --now jimm-ci-nightly.timer
```

### 9. Smoke test

```bash
# Listener is up
curl -fsS https://jimm-ci.<your-domain>/health

# Watch live logs
journalctl -u jimm-ci -f
```

In the GitHub App's **Advanced → Recent Deliveries** panel, pick any past
delivery and click **Redeliver**. Within a few seconds the service log
should show the webhook arriving, a job being enqueued, and a Check Run
appearing on the corresponding commit.

## Operations

### Logs

* Per-build logs: `/var/lib/jimm-ci/logs/<sha>/<family>.log`. Also served
  via `https://jimm-ci.<your-domain>/logs/<sha>/<family>.log` so that the
  Check Run `details_url` resolves from anywhere.
* Service logs: `journalctl -u jimm-ci [-f]` for the listener,
  `journalctl -u cloudflared` for the tunnel, `journalctl -u
  jimm-ci-nightly` for the nightly sweep.

### Restart / redeploy

After pulling new code into `/opt/jimm-ci/Jimm.jl`:

```bash
sudo -u gh-runner git -C /opt/jimm-ci/Jimm.jl pull --ff-only
systemctl restart jimm-ci.service
```

On startup the catch-up loop will replay any webhooks that arrived while
the service was down.

### Kick a nightly manually

```bash
systemctl start jimm-ci-nightly.service
journalctl -u jimm-ci-nightly -f
```

### Rotate the webhook secret

1. Generate a new secret and update **both** `/etc/jimm-ci/webhook-secret`
   and the GitHub App's **Webhook → Secret** field in the same window. Any
   delivery sent between the two writes will fail signature verification
   (return `401`) and end up in the App's redelivery queue, where the
   listener will pick it up after the rotation completes.
2. `systemctl restart jimm-ci.service` so the new secret is loaded.

### Rotate the App private key

1. Generate a new key in the GitHub App settings; download the `.pem`.
2. Replace `/etc/jimm-ci/app.pem` (keep `0640 root:gh-runner`).
3. `systemctl restart jimm-ci.service`.
4. Revoke the old key in GitHub once the listener is healthy.

### Clear stuck delivery state

If a delivery is wedged (for example, you want to force a rerun of an
already-completed GUID), stop the service, edit
`/var/lib/jimm-ci/completed.json` to remove the GUID, and start the service
again. The catch-up loop will see the GUID as unhandled and request
redelivery.

### Adding a new test family

The CI's family routing lives in `ci/jimm_ci/path_filter.py`. When a new
model family is added under `src/Models/<Family>/` with a matching
`test/test_<family>.jl`, update `_FAMILY_PREFIXES`, `_FAMILY_EXACT`,
`ALL_FAMILIES`, and `REPRESENTATIVE_VARIANT` in that file, then redeploy.
The root `CLAUDE.md` repeats this checklist; keep both in sync.

## Security notes

* Every webhook is verified with HMAC-SHA256 against the secret in
  `JIMM_CI_WEBHOOK_SECRET_FILE` before the JSON body is parsed. Failed
  verification returns `401` and no work is queued.
* The listener binds `127.0.0.1` only; the Cloudflare Tunnel is the sole
  public ingress, which means the public surface is exactly the routes
  defined on the FastAPI app (`/webhook`, `/health`, `/logs/...`).
* Pull requests from forks never run user-controlled code on the runner.
  They are short-circuited with a single `jimm-ci / external-pr` Check Run.
  A maintainer must push the branch to the main repository to opt it in.
* The systemd unit hardens the service with `NoNewPrivileges`,
  `ProtectSystem=strict`, `ProtectHome=read-only`, `PrivateTmp`,
  `ProtectKernelTunables`, `ProtectKernelModules`,
  `ProtectControlGroups`, `RestrictNamespaces`, `RestrictRealtime`, and
  `LockPersonality`. The only writable paths are `/var/lib/jimm-ci` and
  `/opt/jimm-ci/Jimm.jl`.
* Secrets live under `/etc/jimm-ci/` with `0640 root:gh-runner`. The
  `gh-runner` account is a system user with no login shell beyond what
  systemd invokes.
