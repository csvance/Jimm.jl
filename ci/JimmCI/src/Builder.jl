module BuilderMod

using Logging

using ..ConfigMod
using ..GitHubAppMod
using ..PathFilter
using ..Jobs

export Builder, run_job

const LOG = Logging.global_logger()

# Maps test family → Python sidecar that dumps its parity fixtures.
# Mirrors the family resolution in scripts/test_variant.sh.
const _FAMILY_SIDECAR = Dict{String,String}(
    "bit"        => "test/parity/dump_resnetv2_bit_io.py",
    "resnet"     => "test/parity/dump_resnet_io.py",
    "convnext"   => "test/parity/dump_convnext_io.py",
    "convnextv2" => "test/parity/dump_convnextv2_io.py",
)

struct Builder
    cfg::Config
    gh::GitHubApp
end

# ── Build-cancel token shared with the TUI ────────────────────────────
# Named `BuildCancel` (not `CancelToken`) to avoid colliding with
# Tachikoma's own `CancelToken`, which cancels task-queue tasks. Ours
# signals the running subprocess to exit; the task itself unwinds
# normally so check_runs are still posted as `cancelled`.

mutable struct BuildCancel
    cancelled::Bool
    proc::Union{Nothing,Base.Process}
    lock::ReentrantLock
end
BuildCancel() = BuildCancel(false, nothing, ReentrantLock())

function request_cancel!(t::BuildCancel)
    @lock t.lock begin
        t.cancelled = true
        p = t.proc
        if p !== nothing && process_running(p)
            try
                kill(p, Base.SIGTERM)
            catch
            end
        end
    end
end

is_cancelled(t::BuildCancel) = @lock t.lock t.cancelled
_attach!(t::BuildCancel, p::Base.Process) = @lock t.lock (t.proc = p)
_detach!(t::BuildCancel)                  = @lock t.lock (t.proc = nothing)

export BuildCancel, request_cancel!, is_cancelled

# ── Subprocess runner streaming stdout+stderr to file + callback ──────

function _read_tail(path::AbstractString; limit::Int = MAX_OUTPUT_TEXT)
    isfile(path) || return ""
    data = read(path)
    if length(data) <= limit
        return String(data)
    end
    head = b"[... log truncated ...]\n"
    tail = data[end - (limit - length(head)) + 1 : end]
    return String(vcat(head, tail))
end

"""
    _stream_subprocess(cmd, env, log_path, on_line, token)

Run `cmd` with merged stdout/stderr, append each line to `log_path` *and*
invoke `on_line(line)`. Honors a `BuildCancel`: on cancel, sends SIGTERM,
waits up to 10 s, then SIGKILL. Returns the child's exit code (or a
nonzero sentinel on cancellation).
"""
function _stream_subprocess(cmd::Cmd, env::Dict{String,String},
                             log_path::AbstractString,
                             on_line::Function,
                             token::BuildCancel;
                             cwd::Union{Nothing,AbstractString}=nothing)
    mkpath(dirname(log_path))
    full_cmd = addenv(cmd, env)
    cwd === nothing || (full_cmd = setenv(full_cmd; dir = cwd))

    pipe = Pipe()
    open(log_path, "w") do logio
        write(logio, "\$ ")
        write(logio, string(cmd))
        write(logio, "\n")
        flush(logio)

        proc = run(pipeline(full_cmd; stdout = pipe, stderr = pipe); wait = false)
        _attach!(token, proc)
        close(pipe.in)   # close parent's copy of the write end

        # Watchdog: if cancellation arrives while we're blocked in readline
        # or waiting on the process, escalate SIGTERM to SIGKILL after 10s.
        watchdog = Threads.@spawn begin
            while process_running(proc)
                if is_cancelled(token)
                    sleep(10)
                    if process_running(proc)
                        try; kill(proc, Base.SIGKILL); catch; end
                    end
                    return
                end
                sleep(0.5)
            end
        end

        try
            while !eof(pipe)
                line = readline(pipe; keep = false)
                println(logio, line)
                flush(logio)
                try
                    on_line(line)
                catch e
                    @warn "on_line callback raised" exception=e
                end
            end
        finally
            wait(proc)
            _detach!(token)
            try; fetch(watchdog); catch; end
        end

        return proc.exitcode
    end
end

# ── Git plumbing ──────────────────────────────────────────────────────

function _git(cfg::Config, args::Vector{String}; cwd::AbstractString,
              log_path::AbstractString)
    env = copy(ENV)
    env["GIT_TERMINAL_PROMPT"] = "0"
    cmd = Cmd(["git"; args])
    mkpath(dirname(log_path))
    open(log_path, "w") do io
        write(io, "\$ ", string(cmd), "\n")
    end
    run(pipeline(setenv(cmd, env; dir = cwd);
                 stdout = log_path, stderr = log_path, append = true))
    return nothing
end

function _is_bare_repo(path::AbstractString)
    isdir(path) || return false
    env = copy(ENV); env["GIT_TERMINAL_PROMPT"] = "0"
    try
        out = read(setenv(`git -C $path rev-parse --is-bare-repository`, env), String)
        return strip(out) == "true"
    catch
        return false
    end
end

function _ensure_mirror!(b::Builder)
    cfg = b.cfg
    if isdir(cfg.mirror_dir) && !_is_bare_repo(cfg.mirror_dir)
        @warn "mirror dir is not a usable bare repo; recloning" path=cfg.mirror_dir
        rm(cfg.mirror_dir; recursive = true, force = true)
    end
    if !isdir(cfg.mirror_dir)
        mkpath(dirname(cfg.mirror_dir))
        _git(cfg,
            ["clone", "--mirror",
             "https://github.com/$(repo_fullname(cfg)).git",
             cfg.mirror_dir],
            cwd = dirname(cfg.mirror_dir),
            log_path = joinpath(cfg.log_dir, "mirror-init.log"),
        )
    else
        _git(cfg, ["fetch", "--prune", "origin"];
             cwd = cfg.mirror_dir,
             log_path = joinpath(cfg.log_dir, "mirror-fetch.log"))
    end
end

function _link_parity_dir!(cfg::Config, wt::AbstractString)
    mkpath(cfg.parity_dir)
    data = joinpath(wt, "data")
    mkpath(data)
    link = joinpath(data, "parity")
    if islink(link) || ispath(link)
        if isdir(link) && !islink(link)
            rm(link; recursive = true)
        else
            rm(link)
        end
    end
    symlink(cfg.parity_dir, link)
end

function _make_worktree!(b::Builder, sha::AbstractString)
    cfg = b.cfg
    wt = joinpath(cfg.workspace_dir, sha)
    isdir(wt) && rm(wt; recursive = true, force = true)
    mkpath(dirname(wt))
    _git(cfg, ["-C", cfg.mirror_dir, "worktree", "add", "--detach", wt, sha];
         cwd = cfg.mirror_dir,
         log_path = joinpath(cfg.log_dir, sha, "worktree.log"))
    _link_parity_dir!(cfg, wt)
    return wt
end

function _drop_worktree!(b::Builder, wt::AbstractString)
    cfg = b.cfg
    try
        _git(cfg, ["-C", cfg.mirror_dir, "worktree", "remove", "--force", wt];
             cwd = cfg.mirror_dir,
             log_path = joinpath(cfg.log_dir, basename(wt), "worktree-remove.log"))
    catch e
        @warn "worktree remove failed" wt exception=e
        isdir(wt) && rm(wt; recursive = true, force = true)
    end
end

# ── Env construction ─────────────────────────────────────────────────

function _env_for_family(cfg::Config, family::AbstractString, variant::AbstractString)
    env = copy(ENV)
    env["JULIA_NUM_THREADS"]  = get(ENV, "JULIA_NUM_THREADS", "4")
    env["HF_HUB_CACHE"]       = cfg.hf_cache
    env["JULIA_DEPOT_PATH"]   = cfg.julia_depot
    env["JIMM_TEST_FAMILIES"] = String(family)
    env["JIMM_TEST_VARIANTS"] = String(variant)
    if cfg.hf_token !== nothing
        env["HF_TOKEN"]                = cfg.hf_token
        env["HUGGING_FACE_HUB_TOKEN"]  = cfg.hf_token
    end
    return env
end

function _env_for_sidecar(cfg::Config)
    env = copy(ENV)
    env["UV_PROJECT_ENVIRONMENT"] = cfg.python_env
    env["HF_HUB_CACHE"]           = cfg.hf_cache
    if cfg.hf_token !== nothing
        env["HF_TOKEN"]               = cfg.hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = cfg.hf_token
    end
    return env
end

# ── Parity fixture dump ──────────────────────────────────────────────

function _ensure_fixtures!(b::Builder, job::Job, wt::AbstractString,
                            family::AbstractString, variant::AbstractString,
                            on_line::Function, token::BuildCancel)
    sidecar = get(_FAMILY_SIDECAR, family, nothing)
    sidecar === nothing && return

    if !isempty(variant)
        fixture = joinpath(b.cfg.parity_dir, "$(variant)_io.h5")
        if isfile(fixture)
            return
        end
        args = ["--variant", variant]
    else
        args = ["--all"]
    end

    log_path = joinpath(b.cfg.log_dir, job.head_sha, "$family-dump.log")
    cmd = Cmd(String["uv", "run", "--project", wt, "python", sidecar, args...])
    env = _env_for_sidecar(b.cfg)
    rc = _stream_subprocess(cmd, env, log_path, on_line, token; cwd = wt)
    rc == 0 || error("parity dump for $family/$(isempty(variant) ? "all" : variant) " *
                     "failed (rc=$rc); see $log_path")
    return
end

# ── Job execution ────────────────────────────────────────────────────

function _run_family!(b::Builder, job::Job, wt::AbstractString,
                       family::AbstractString, variant::AbstractString,
                       on_line::Function, token::BuildCancel)
    name = check_name(family, variant)
    log_path = joinpath(b.cfg.log_dir, job.head_sha, "$family.log")

    check = create_check_run(b.gh, repo_fullname(b.cfg), job.head_sha, name;
                              status = "in_progress")
    job.check_runs[family] = check.id
    on_line("==> [check-run] $(name) → in_progress (id=$(check.id))")

    rc = 1
    try
        _ensure_fixtures!(b, job, wt, family, variant, on_line, token)
        if is_cancelled(token)
            throw(InterruptException())
        end

        cmd = Cmd([
            b.cfg.julia_binary, "--project=.", "-e",
            "using Pkg; Pkg.instantiate(); Pkg.test()",
        ])
        env = _env_for_family(b.cfg, family, variant)
        rc = _stream_subprocess(cmd, env, log_path, on_line, token; cwd = wt)
    catch e
        if is_cancelled(token) || e isa InterruptException
            on_line("==> [check-run] $(name) cancelled")
            try
                complete_check_run(b.gh, repo_fullname(b.cfg), check.id;
                    conclusion = "cancelled",
                    output = Dict("title"   => "$family cancelled",
                                  "summary" => "Build was cancelled.",
                                  "text"    => _read_tail(log_path)))
            catch err
                @warn "completing check_run as cancelled failed" exception=err
            end
            rethrow()
        else
            @warn "subprocess failed" family exception=e
            try
                complete_check_run(b.gh, repo_fullname(b.cfg), check.id;
                    conclusion = "failure",
                    output = Dict("title"   => "$family errored",
                                  "summary" => "Service-level error: $(sprint(showerror, e))",
                                  "text"    => _read_tail(log_path)))
            catch err
                @warn "completing check_run as failure failed" exception=err
            end
            return
        end
    end

    conclusion = rc == 0 ? "success" : "failure"
    on_line("==> [check-run] $(name) → $(conclusion) (rc=$(rc))")
    complete_check_run(b.gh, repo_fullname(b.cfg), check.id;
        conclusion = conclusion,
        output = Dict("title"   => "$family $(rc == 0 ? "passed" : "failed")",
                      "summary" => "Exit code $(rc). Variant: `$(isempty(variant) ? "all" : variant)`. " *
                                   "Sweep: `$(job.full_sweep)`.",
                      "text"    => _read_tail(log_path)))
end

"""
    run_job(builder, job; on_line, token)

Run every family in `job` serially, streaming each line of the build's
combined stdout/stderr through `on_line(line)`. Cancellation is honored
via `token` (see `BuildCancel`).
"""
function run_job(builder::Builder, job::Job;
                  on_line::Function = identity,
                  token::BuildCancel = BuildCancel())
    on_line("==> starting $(job.label) sha=$(job.head_sha) " *
            "families=$(join(job.families, ",")) sweep=$(job.full_sweep)")
    _ensure_mirror!(builder)
    wt = _make_worktree!(builder, job.head_sha)
    try
        for family in job.families
            is_cancelled(token) && throw(InterruptException())
            variant = job.full_sweep ? "" :
                      get(REPRESENTATIVE_VARIANT, family, "")
            _run_family!(builder, job, wt, family, variant, on_line, token)
        end
    finally
        _drop_worktree!(builder, wt)
    end
    return nothing
end

end # module
