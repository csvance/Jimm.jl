---
name: kaimon-julia
description: Workflow guide for driving Julia projects through the Kaimon MCP server. Load this skill whenever working with Julia code in any capacity, including reading, writing, editing, or debugging .jl files, Project.toml/Manifest.toml, Julia REPL sessions, Pkg operations, Revise-based hot-reload workflows, or any Kaimon tool calls (start_session, ex, manage_repl, pkg_add, pkg_rm, check_eval, debug_exfiltrate, goto_definition, workspace_symbols). Kaimon is the preferred execution path for all Julia work in this repo.
---

# Kaimon + Julia REPL Workflow

A reference for working a Julia project through the Kaimon MCP server. Covers session lifecycle, output management, Revise behavior, and the rough edges that cost the most round trips when you hit them blind.

**Do not call Kaimon's `usage_instructions` MCP tool.** It exists and will return a usage guide that overlaps with — and in places contradicts — this skill (notably around the `ses` parameter name, the `Gate.serve()` launch pattern, and the `Available sessions` error semantics). This skill is the source of truth for working in this repo; reading `usage_instructions` will only burn tokens and introduce conflicting guidance. If something here looks wrong, ask the user rather than falling back to `usage_instructions`.

## 1. Session lifecycle

Prefer launching Julia yourself in a background terminal and letting it call `Gate.serve()` to attach to the running Kaimon TUI as a session. This puts precompile output in front of you, sidesteps the `start_session` tool-timeout dance, and gives you a real REPL you can attach to for diagnosis if Kaimon's MCP side gets confused.

```bash
# Launch once, in a background process (e.g. launch-process with wait=false):
julia -e 'using Revise; using Kaimon; using Pkg; Pkg.activate("/path/to/pkg"); Gate.serve(); wait(Condition())'
```

Substitute `/path/to/pkg` with the project directory you want active (`/workspace`, `/workspace/lib/Foo`, etc.). Loading `Revise` before `Kaimon` ensures it's tracking from the start; `Gate.serve()` opens the same MCP gate `start_session` would have spawned, so subsequent `ex`, `manage_repl`, `pkg_add`, etc. calls all work normally.

The trailing `wait(Condition())` is **not optional**. `Gate.serve()` returns as soon as the handshake completes; without a blocking call after it, `julia -e '...'` exits and takes the session with it. `julia -i -e '...'` is not a workaround in the default agent setup — interactive mode also exits when stdin is closed (e.g. piped from `/dev/null` by a background-process launcher). `wait(Condition())` parks the main task forever, which is what you want.

Once that process is up, call `start_session()` with **no arguments** to list connected sessions, find the one whose project matches and is `running (session: <key>)`, and use that key for everything else:

```
start_session()                                  # discover the session key (authoritative)
investigate_environment(ses=<key>)               # confirms version + Revise + deps
ex(e="...", ses=<key>)                           # all subsequent calls
```

- **The parameter is `ses`, not `session`.** Every Kaimon tool that targets a REPL takes an 8-character key as `ses=...`, even though the TUI and error messages refer to it as "the session" everywhere else. Passing `session=<key>` silently fails with `No session matched ''` because the kwarg is unrecognized and `ses` defaults to empty. Use `ses=` in every call.
- **`start_session()` is the only authoritative session list.** When you omit `ses`, some tools emit an `Available sessions: ...` error that includes stale or disconnected keys from earlier processes. Do not pick a key from that error message — call `start_session()` and use a key from its output.
- **A freshly spawned Julia can show `ready` until you claim it.** If `start_session()` lists every project as `ready` rather than `running (session: <key>)`, the side-launched Julia is alive but unattached. Call `start_session(project_path="/path/to/pkg")` once to attach; subsequent `start_session()` calls will then show `running (session: <key>)`. This looks identical to a dead Julia at first glance — check before restarting the background process.
- **One session, one key, every call.** The key must be passed as `ses=<key>` to every tool call that touches the REPL. There is no implicit "current session" when multiple are connected. Save the key the moment you get it.
- **`investigate_environment` first, always.** It reports the Julia version, `pwd`, active project, installed packages, dev packages, and whether Revise is active. Free information that prevents wrong assumptions about what's loaded.
- **Don't `Pkg.activate(...)` from inside `ex`.** The background launcher already activated the project. Re-activating breaks subsequent `pkg_add` calls.
- **Subproject sessions just work.** Point the background `Pkg.activate(...)` at any directory with a `Project.toml` — monorepo `lib/Foo/`, `test/`, `docs/`, etc. No need to touch `~/.config/kaimon/projects.json`, since the path-allowlist check `start_session` enforces is bypassed when Julia connects outward via `Gate.serve()`.
- **Pre-warm a tightly-pinned manifest.** If the target project's `Manifest.toml` pins versions older than what Kaimon was built against (leaf deps like `JSON`, `Preferences` are common culprits), the `using Kaimon` step will trigger a recompile that then chases missing transitive deps one at a time. Run `Pkg.update()` in the target project first, or vendor a manifest known to be compatible with Kaimon's build versions.
- **Falling back to `start_session(project_path=...)`.** Still works; useful when you can't or don't want to open a side terminal. Same caveats apply, plus: first-time startup for a complex environment (many deps, heavy compiled packages like `Plots`, `DifferentialEquations`, `Makie`, CUDA stacks) often runs longer than the default tool timeout. The precompile keeps running in the background after the timeout returns. *Do not* call `start_session(project_path=...)` again as a "retry" — that risks queuing a second competing startup. Call `start_session()` with no arguments to poll status; when the target shows `running (session: <key>)`, capture that key. The agent harness blocks standalone foreground `sleep` calls; if you want an explicit gap between status checks, either do useful work in between or run the sleep in a background process.

## 2. The `q` flag is a token-budget decision

Default is `q=true` (strip return value). Choose mechanically:

- `q=true` (default): assignments, imports, definitions, anything whose return value is "the assigned thing" or `nothing`. This is 80% of calls.
- `q=false`: you need the value to make a decision this turn (allocation count, equality check, type inspection).

The stdout rule is more aggressive than it looks: **anything written to stdout is stripped**, so `println("debug: x = $x")` vanishes. To surface a value, end the `ex` expression with the value itself and `q=false`:

```julia
ex(e="(length(result), typeof(result))", q=false)   # good
ex(e="println(result)", q=true)                      # invisible
```

`@info` survives the strip. It is the only reliable way to emit per-step progress from inside a long function and have the agent see it.

## 3. Large outputs: kill them at the source

Calling `ex` with `q=false` on a richly-printed value (a `BenchmarkTools.Trial`, a `DataFrame`, a large `Dict`) dumps the full pretty-print into the conversation and you pay for those tokens for the rest of the session.

Three layers of defense, cheapest first:

1. **End with `; nothing`** when you only care about the side effect of storing into a variable: `ex(e="results = run_all(); nothing")`. The trailing expression is `nothing`, so even `q=false` returns nothing.
2. **Write a small pretty-printer** and call it from a separate `ex`. The function does its own controlled `print(rpad(...))` and you shape the output exactly.
3. **`max_output=15000`** raises the truncation cap to 25 000 characters. Reach for it only when the output is intrinsically large and you actually need to read it.

`s=true` (silent mode, suppresses the `agent>` echo) exists but is rarely needed; the usage doc calls it out as a special case.

## 4. Revise: invisible until it isn't

Revise is loaded into the session before you get there. `Revise.revise()` is a no-op in this setup. The mental model:

- Edits to any `.jl` file `include`d from a tracked package module are picked up between `ex` calls automatically.
- The "tracked package module" is whatever you `using ...` from the project. Files included from that module are tracked transitively.
- **Driver scripts go outside `src/`.** A one-shot top-level script does not benefit from Revise tracking and adds overhead at `using` time.
- **Restart is cheap, but not always free.** `manage_repl(command="restart")` preserves the session key. You lose in-memory variables but regain a clean world. Restart when:
  - You added or removed a struct field.
  - You changed module-level code (an `include`, a top-level `const`, an `import`).
  - You hit a `MethodError` or world-age error that persists after edits that should have fixed it.
  - A long-lived background task crashed and left module-level state stale. The classic case is a server in `@async` (Oxygen, HTTP.jl) whose route registry, scheduler table, or connection pool is now out of sync with the source. The code reloads fine, but the in-memory registry no longer matches it.

For function-body edits, which is 95% of normal work, restart is not needed. Save, run `ex`, observe the new behavior.

## 5. Batch eagerly, but not blindly

`ex(e="x = 1; y = 2; z = f(x, y)")` is one round trip; three separate `ex` calls is three. Batch when the commands form one logical step and you do not need an intermediate value to decide what comes next.

Do not batch when:

- You need the value of the first call to decide the second (e.g., check `length(result)` before deciding whether to dump it).
- One command might fail and you want to isolate the error.
- The combined output would be large enough to truncate.

Rule of thumb: setup + warmup + smoke + check fit in one `ex`; benchmark runs and result dumps are separate calls.

## 6. `pkg_add` vs `Pkg.add`, and the stdlib trap

Use the `pkg_add` MCP tool, not `Pkg.add(...)` via `ex`. It is one tool call, modifies `Project.toml` atomically, and reports the resulting state cleanly. Same for `pkg_rm`.

**Sharp edge:** even Julia's standard library packages must be listed in `Project.toml` if the *package module* references them. The session REPL has stdlibs available, but a package being precompiled does not inherit the REPL's environment. The symptom looks like:

```
ArgumentError: Package MyPkg does not have LinearAlgebra in its dependencies
```

Fix: `pkg_add(packages=["LinearAlgebra", "Random", "Printf"])` and re-`using`. Scan the `using` statements in `src/` before the first `using <YourPackage>` and pre-add any stdlib they reference.

## 7. Cheap correctness gates before `@benchmark`

`@benchmark` is the slowest tool in the box. Define cheaper gates and run them between every meaningful edit:

- A `smoke_test()`-style function: runs every kernel once, asserts cross-implementation equality. Microseconds.
- An `allocation_gate()`-style function: runs each kernel after warmup, captures `@allocated`, raises if a kernel claiming zero allocations isn't. Microseconds.
- The actual benchmark: seconds to minutes; run when you are confident the code is right and you specifically want a timing number.

Design these gates to return small, structured values (a `NamedTuple`, a `Vector{Pair{Symbol,Int}}`) so they fit on one screen with `q=false`. They are meant to be eyeballed each round.

## 8. Less-used flags worth knowing

- **`mt=true`**: required for GLMakie / GLFW / OpenGL, which need thread 1. Irrelevant for pure-compute code. The moment you `using GLMakie`, every related `ex` needs `mt=true`. `ThreadAssertionError` from a plot library means you forgot it.
- **`check_eval(eval_id="...")`**: every `ex` returns an `eval_id`. If a call times out, or you want to keep working while a long run completes, poll the result later via `check_eval`. The right tool for multi-minute runs.
- **`debug_exfiltrate` / Infiltrator**: inspect state inside a function without modifying the call site. Useful when output is wrong and you want to see locals at the failing point. Often unnecessary if a correctness gate (Section 7) catches the bug at a boundary.
- **`goto_definition`, `workspace_symbols`**: code navigation backed by Julia's own LanguageServer indexing. Prefer these over `grep` when working in a large existing codebase. Note: `qdrant_search_code` is also exposed by Kaimon but requires a Qdrant server, which is not provisioned in this environment, so it will fail at call time.

## 9. The single biggest mistake to avoid

Running `ex` with `q=false` on a call that returns a large or richly-printed value. Once the tool result is in your context, you pay for those tokens for the rest of the conversation. The cure costs nothing: end the expression with `nothing`, store into a variable, and make a separate small call to inspect only what you need.