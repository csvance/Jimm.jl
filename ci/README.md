# Jimm.jl self-hosted CI

`jimm-ci` is two small CLI tools that run the Jimm.jl parity test suite
on a self-hosted Linux VM and report results back to GitHub via the
Checks API. There is **no webhook listener, no public endpoint, and no
domain name involved** — every run is started by the maintainer, by
hand, over SSH.

* `jimm-ci-run` — finds every commit eligible to be tested (open PRs
  bearing a per-SHA approval label, plus master commits in the last 30
  days with no completed `jimm-ci` check) and runs the suite against
  each, posting Check Run results to GitHub.
* `jimm-ci-pr` — prints the commits in a PR with the exact label string
  that approves each one, and (with `--apply`) creates and applies the
  label for the PR head in one command.
* `jimm-ci-skip` — posts `conclusion=skipped` Check Runs on one or more
  commits so the runner stops considering them. Use it to clear a
  backlog of commits you've decided not to retroactively test
  (`--all-pending`) or to permanently exclude a single SHA.

```
                 ┌──────────────────────────────────────┐
                 │ GitHub                               │
                 │   ├── Pulls / Commits / Compare      │
                 │   └── Checks API                     │
                 └────────────┬─────────────────────────┘
                              │ HTTPS (outbound only)
                  ┌───────────▼─────────────┐
                  │ jimm-ci-run / jimm-ci-pr│
                  │   (launched over SSH)   │
                  │  git mirror + worktree  │
                  │  julia Pkg.test         │
                  └─────────────────────────┘
```

## Approval model

A pull request is approved for CI by attaching a label of the form
`ci-approved-<short-sha>`, where `<short-sha>` is the first seven hex
characters of the PR head commit. The label is **per-SHA**: if a new
commit is pushed to the PR, the encoded SHA stops matching the head and
`jimm-ci-run` skips the PR until a label with the new short SHA is
applied. This prevents an approved PR from silently running tests on an
unreviewed follow-up commit.

Master commits are not gated — merged code is presumed reviewed at merge
time.

The label prefix is configurable via `JIMM_CI_APPROVAL_LABEL_PREFIX`
(default `ci-approved-`).

## Layout

```
ci/
├── pyproject.toml          # uv-managed Python project, entry points
└── jimm_ci/
    ├── runner.py           # jimm-ci-run: discover + build loop
    ├── inspector.py        # jimm-ci-pr: PR commit table + --apply
    ├── github_app.py       # JWT + installation token + Checks/Compare/Pulls/Labels
    ├── path_filter.py      # changed paths → test families
    └── config.py           # env-var-driven configuration
```

## What the runner does

On every invocation, `jimm-ci-run`:

1. Lists open PRs, drops fork PRs, drops PRs without a
   `ci-approved-<head_sha[:7]>` label.
2. For each remaining PR, calls Compare to map changed paths → test
   families, then queries the Checks API for the head commit and keeps
   only families with no completed `jimm-ci / <family>` Check Run yet.
   PR-scope jobs use `REPRESENTATIVE_VARIANT` per family.
3. Lists master commits in the last 30 days; for each commit missing
   any of the four `jimm-ci` Check Runs, queues a full-sweep job.
4. Processes the resulting job list serially. For each family in a job:
   posts an `in_progress` Check Run, runs
   `julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'`
   inside a detached `git worktree` with
   `JIMM_TEST_FAMILIES` / `JIMM_TEST_VARIANTS` set, then completes the
   Check Run with the tail of the build log (capped at ~60 KB).

`--dry-run` prints the discovered jobs and exits without invoking Julia.

## Configuration

All configuration is environment-driven. Secrets are loaded from files,
not env strings.

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `JIMM_CI_APP_ID` | yes | — | GitHub App ID |
| `JIMM_CI_INSTALLATION_ID` | yes | — | App installation on the repo |
| `JIMM_CI_PRIVATE_KEY_FILE` | yes | — | Path to App PEM |
| `JIMM_CI_REPO_OWNER` | yes | — | e.g. `cvance` |
| `JIMM_CI_REPO_NAME` | yes | — | e.g. `Jimm.jl` |
| `JIMM_CI_STATE_DIR` | no | `/var/lib/jimm-ci` | Mirror, worktrees, logs, depot |
| `JIMM_CI_HF_TOKEN_FILE` | no | — | HuggingFace token for parity weights |
| `JIMM_CI_JULIA` | no | `/usr/local/bin/julia` | Julia binary path |
| `JIMM_CI_APPROVAL_LABEL_PREFIX` | no | `ci-approved-` | Prefix the runner looks for on PR labels |
| `JULIA_NUM_THREADS` | no | `4` | Forwarded to test jobs |

## Setup on Debian 13

Assumes a fresh Debian 13 VM with internet access. Run everything as
`root` unless noted.

### 1. System packages and `ci` user

```bash
apt-get update
apt-get install -y --no-install-recommends \
    build-essential git ca-certificates curl jq tar xz-utils \
    python3 python3-venv

adduser --system --group --home /home/ci --shell /bin/bash ci
```

### 2. Install `uv` (Python project runner)

```bash
sudo -u ci bash -lc 'curl -LsSf https://astral.sh/uv/install.sh | sh'
ln -sf /home/ci/.local/bin/uv /usr/local/bin/uv
```

### 3. Install Julia via juliaup

```bash
sudo -u ci bash -lc 'curl -fsSL https://install.julialang.org | sh -s -- --yes'
ln -sf /home/ci/.juliaup/bin/julia /usr/local/bin/julia
```

### 4. Clone the repository and create state directories

```bash
install -d -o ci -g ci /opt/jimm-ci
sudo -u ci git clone https://github.com/<OWNER>/Jimm.jl.git /opt/jimm-ci/Jimm.jl

install -d -o ci -g ci \
    /var/lib/jimm-ci \
    /var/lib/jimm-ci/julia-depot \
    /var/lib/jimm-ci/hf-cache
```

The runner creates `mirror.git`, `work/`, and `logs/` under
`/var/lib/jimm-ci/` on first invocation.

### 5. Register the GitHub App

In the GitHub UI (Settings → Developer settings → GitHub Apps → New
GitHub App):

* **Homepage URL:** anything (the repo URL is fine).
* **Webhook:** uncheck "Active". The app does not need a webhook URL or
  secret.
* **Permissions (repository):**
  * Checks: **Read & write**
  * Contents: **Read-only**
  * Issues: **Read & write** (only required if you want
    `jimm-ci-pr --apply` to create and apply labels)
  * Metadata: **Read-only**
  * Pull requests: **Read & write** if you want `--apply` to work,
    otherwise **Read-only**
* **Subscribe to events:** none.
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
install -d -m 0750 -o root -g ci /etc/jimm-ci

# App private key
install -m 0640 -o root -g ci /path/to/jimm-ci.<id>.private-key.pem \
    /etc/jimm-ci/app.pem

# HuggingFace token used by parity tests to fetch weights
printf '%s' '<hf_xxx...>' > /etc/jimm-ci/hf-token
chmod 0640 /etc/jimm-ci/hf-token
chown root:ci /etc/jimm-ci/hf-token
```

### 7. Configure the shell environment

The `ci` user needs the GitHub App configuration in its environment
whenever it runs the CLI. The cleanest place is `/home/ci/.profile` (or
`/home/ci/.bashrc`):

```bash
sudo -u ci tee -a /home/ci/.profile <<'SH'
export JIMM_CI_APP_ID=123456
export JIMM_CI_INSTALLATION_ID=7890123
export JIMM_CI_PRIVATE_KEY_FILE=/etc/jimm-ci/app.pem
export JIMM_CI_HF_TOKEN_FILE=/etc/jimm-ci/hf-token
export JIMM_CI_REPO_OWNER=<owner>
export JIMM_CI_REPO_NAME=Jimm.jl
export JIMM_CI_JULIA=/usr/local/bin/julia
export JULIA_NUM_THREADS=4
SH
```

## Running CI

Everything below runs as the `ci` user.

### Dry-run discovery

```bash
ssh ci@<vm> 'cd /opt/jimm-ci/Jimm.jl/ci && uv run jimm-ci-run --dry-run'
```

Prints the jobs that would execute and exits. Useful to confirm an
approval label has taken effect before paying for the actual run.

### Execute pending jobs

```bash
ssh ci@<vm> 'cd /opt/jimm-ci/Jimm.jl/ci && uv run jimm-ci-run'
```

Drains every discovered job serially. Re-running immediately is a no-op:
the Check Runs created on the first invocation make the runner consider
each commit as already tested.

### Approve a PR for CI

```bash
ssh ci@<vm> 'cd /opt/jimm-ci/Jimm.jl/ci && uv run jimm-ci-pr <pr-url>'
```

Prints a table of commits, the label string that approves each, and
marks the current head. Add `--apply` to create the label (if it does
not exist in the repo yet) and attach it to the PR:

```bash
ssh ci@<vm> 'cd /opt/jimm-ci/Jimm.jl/ci && uv run jimm-ci-pr <pr-url> --apply'
```

Once a PR head is labeled, the next `jimm-ci-run` picks it up.

### Skip pending commits without testing them

```bash
ssh ci@<vm> 'cd /opt/jimm-ci/Jimm.jl/ci && uv run jimm-ci-skip --all-pending'
```

Posts a `skipped` Check Run for every family on every currently-pending
commit so the runner stops reconsidering them. Pass explicit SHAs to
skip a specific subset (`jimm-ci-skip <sha> [<sha> ...]`), or add
`--dry-run` to preview which commits would be marked.

## Operations

### Logs

Per-build logs are at `/var/lib/jimm-ci/logs/<sha>/<family>.log`. The
tail of each log is also embedded into the corresponding GitHub Check
Run output (capped at ~60 KB to stay under the GitHub limit), so the
GitHub UI is usually enough for spot-checking; SSH into the VM only when
you need the full log.

### Redeploy

```bash
ssh ci@<vm> 'sudo -u ci git -C /opt/jimm-ci/Jimm.jl pull --ff-only'
```

No service to restart; the next `jimm-ci-run` invocation will pick up
the new code automatically.

### Forcing a re-test of an already-tested commit

The runner skips any commit whose `jimm-ci / <family>` Check Runs are
all already `completed`. To force a re-test:

1. In the GitHub UI, delete the existing Check Runs for that commit
   (Checks tab → ⋯ → **Re-run** triggers a fresh run via the API too,
   but the simplest path is just to re-create the App's check runs).
2. Or, push a no-op follow-up commit and label it.

### Rotate the App private key

1. Generate a new key in the GitHub App settings; download the `.pem`.
2. Replace `/etc/jimm-ci/app.pem` (keep `0640 root:ci`).
3. Revoke the old key in the GitHub UI once a `jimm-ci-pr --dry-run` is
   confirmed to still authenticate.

### Adding a new test family

The CI's family routing lives in `ci/jimm_ci/path_filter.py`. When a
new model family is added under `src/Models/<Family>/` with a matching
`test/test_<family>.jl`, update `_FAMILY_PREFIXES`, `_FAMILY_EXACT`,
`ALL_FAMILIES`, and `REPRESENTATIVE_VARIANT` in that file. The root
`CLAUDE.md` repeats this checklist; keep both in sync.

## Security notes

* The VM never accepts inbound connections from anyone but you over SSH.
  GitHub talks to it only through outbound HTTPS calls made by the CLI.
* Pull requests from forks are filtered out client-side in
  `jimm-ci-run` and never produce a Check Run. A maintainer must push
  the branch to the main repository to opt it in.
* The per-SHA approval label is the only thing standing between an
  attacker-pushed commit and code execution on the VM. Always review
  the diff at the SHA you are about to label, not just "the PR."
* Secrets live under `/etc/jimm-ci/` with `0640 root:ci`. The `ci`
  account is a system user.
