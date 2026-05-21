module Tui

using Dates
using Tachikoma
import Tachikoma: view, update!, should_quit, task_queue, set_wake!, has_pending_output

using ..ConfigMod
using ..GitHubAppMod
using ..PathFilter
using ..Jobs
using ..BuilderMod
using ..SkipMarker

export TuiModel, run_tui

# ── View modes ───────────────────────────────────────────────────────

@enum ViewMode VIEW_LIST VIEW_RUNNING VIEW_CONFIRM_CANCEL

mutable struct RunningJob
    job::Job
    cancel::BuildCancel
    started_at::DateTime
    current_family::String
end

# ── Model ────────────────────────────────────────────────────────────

@kwdef mutable struct TuiModel <: Model
    cfg::Config
    gh::GitHubApp
    builder::Builder

    jobs::Vector{Job}  = Job[]
    selected::Int      = 1
    list_offset::Int   = 0
    mode::ViewMode     = VIEW_LIST

    running::Union{Nothing,RunningJob} = nothing
    queue::Vector{Job} = Job[]            # back-to-back master queue

    # Live-output plumbing
    log_pane::ScrollPane = ScrollPane(String[];
        block = Block(title = " build output ",
                      border_style = tstyle(:border),
                      title_style  = tstyle(:title, bold = true)))

    # Background tasks (Tachikoma)
    tq::TaskQueue = TaskQueue()

    # Thread-safe log pipeline: background threads put! here; view() drains into log_pane.
    log_channel::Channel{String} = Channel{String}(Inf)
    _wake::Union{Nothing,Function} = nothing

    refreshing::Bool   = false
    status::String     = "press r to refresh, ↑/↓ to select, Enter to run"
    tick::Int          = 0
    quit::Bool         = false
end

should_quit(m::TuiModel)        = m.quit
task_queue(m::TuiModel)         = m.tq
set_wake!(m::TuiModel, f::Function) = (m._wake = f; nothing)
has_pending_output(m::TuiModel) = isready(m.log_channel)

# ── Helpers ──────────────────────────────────────────────────────────

function _short(sha::AbstractString, n::Int = 8)
    n = min(n, length(sha))
    return SubString(sha, 1, n)
end

function _row_text(j::Job)
    when = Dates.format(j.created_at, dateformat"yyyy-mm-dd HH:MM")
    kind = j.kind == Jobs.PR_JOB ? "PR    " : "master"
    fams = join(j.families, ",")
    title = if j.kind == Jobs.PR_JOB
        "#$(j.pr_number) $(j.pr_title === nothing ? "" : j.pr_title)"
    else
        "master @ $(_short(j.head_sha))"
    end
    return "$when  $kind  $(rpad(title, 50))  [$fams]"
end

function _push_log!(m::TuiModel, line::AbstractString)
    put!(m.log_channel, String(line))
    m._wake !== nothing && m._wake()
end

function _log_callback(m::TuiModel)
    return line -> _push_log!(m, line)
end

_current_master_index_or_nothing(m::TuiModel) =
    1 <= m.selected <= length(m.jobs) && m.jobs[m.selected].kind == Jobs.MASTER_JOB ?
        m.selected : nothing

# ── Background work: discovery (refresh) ─────────────────────────────

function _spawn_refresh!(m::TuiModel)
    m.refreshing && return
    m.refreshing = true
    m.status     = "refreshing…"
    spawn_task!(m.tq, :refresh) do
        return discover_jobs(m.cfg, m.gh)
    end
    return nothing
end

# ── Background work: running one job ─────────────────────────────────

function _spawn_run!(m::TuiModel, job::Job)
    cancel = BuildCancel()
    m.running = RunningJob(job, cancel, now(UTC), isempty(job.families) ? "" : first(job.families))
    m.mode    = VIEW_RUNNING
    # Clear the log pane between jobs so the user isn't confused by
    # leftover output from the previous build.
    empty!(m.log_pane.content::Vector{String})
    m.log_pane.block = Block(
        title       = " build output  $(job.label) ",
        border_style = tstyle(:border),
        title_style  = tstyle(:title, bold = true))

    on_line = _log_callback(m)
    spawn_task!(m.tq, :run) do
        try
            BuilderMod.run_job(m.builder, job; on_line = on_line, token = cancel)
            return :ok
        catch e
            if e isa InterruptException || BuilderMod.is_cancelled(cancel)
                return :cancelled
            end
            @error "run_job failed" exception=(e, catch_backtrace())
            return e
        end
    end
    return nothing
end

function _start_next_in_queue!(m::TuiModel)
    isempty(m.queue) && return false
    next_job = popfirst!(m.queue)
    _spawn_run!(m, next_job)
    return true
end

# ── Background work: skip a single job ───────────────────────────────

function _spawn_skip!(m::TuiModel, job::Job)
    spawn_task!(m.tq, :skip) do
        mark_skipped(m.gh, repo_fullname(m.cfg), job.head_sha, job.families;
                     source = :run)
        return job.head_sha
    end
    m.status = "skipping $(job.label)…"
    return nothing
end

# ── Key handling ─────────────────────────────────────────────────────

function update!(m::TuiModel, evt::KeyEvent)
    if m.mode == VIEW_CONFIRM_CANCEL
        return _on_key_confirm_cancel!(m, evt)
    elseif m.mode == VIEW_RUNNING
        return _on_key_running!(m, evt)
    else
        return _on_key_list!(m, evt)
    end
end

function _on_key_list!(m::TuiModel, evt::KeyEvent)
    if evt.key == :char
        c = evt.char
        c == 'q' && (m.quit = true; return)
        c == 'j' && (_move_selection!(m,  1); return)
        c == 'k' && (_move_selection!(m, -1); return)
        c == 'r' && (_spawn_refresh!(m); return)
        c == 's' && (_skip_selected!(m);   return)
        c == 'A' && (_run_all_master!(m); return)
        c == 'y' && (_run_selected!(m);   return)
    elseif evt.key == :up
        _move_selection!(m, -1)
    elseif evt.key == :down
        _move_selection!(m, 1)
    elseif evt.key == :enter || evt.key == :return
        _run_selected!(m)
    elseif evt.key == :escape
        m.quit = true
    end
end

function _on_key_running!(m::TuiModel, evt::KeyEvent)
    if evt.key == :char
        c = evt.char
        c == 'c' && (m.mode = VIEW_CONFIRM_CANCEL; return)
        c == 'C' && (_cancel_all!(m); return)
    elseif evt.key == :escape
        m.mode = VIEW_CONFIRM_CANCEL
        return
    end
    handle_key!(m.log_pane, evt)
end

function _on_key_confirm_cancel!(m::TuiModel, evt::KeyEvent)
    if evt.key == :char
        c = evt.char
        if c == 'y' || c == 'Y'
            _cancel_current!(m)
        elseif c == 'a' || c == 'A'
            _cancel_all!(m)
        elseif c == 'n' || c == 'N'
            m.mode = VIEW_RUNNING
        end
    elseif evt.key == :escape
        m.mode = VIEW_RUNNING
    end
end

function _move_selection!(m::TuiModel, delta::Int)
    isempty(m.jobs) && return
    m.selected = clamp(m.selected + delta, 1, length(m.jobs))
end

function _run_selected!(m::TuiModel)
    isempty(m.jobs) && return
    job = m.jobs[m.selected]
    # Remove from list — it's now in flight.
    deleteat!(m.jobs, m.selected)
    m.selected = clamp(m.selected, 1, max(length(m.jobs), 1))
    _spawn_run!(m, job)
end

function _run_all_master!(m::TuiModel)
    isempty(m.jobs) && return
    idx = _current_master_index_or_nothing(m)
    if idx === nothing
        m.status = "[A] only enqueues master commits (selection is a PR)"
        return
    end
    # Build queue oldest-first: anchor + every older master commit.
    anchor = m.jobs[idx]
    masters_older = [j for (i, j) in pairs(m.jobs)
                     if i > idx && j.kind == Jobs.MASTER_JOB]
    sort!(masters_older; by = j -> j.created_at)   # oldest first
    queue = vcat(masters_older, [anchor])

    # Drop them from the visible list.
    keep_idx = Set{Int}()
    for (i, j) in pairs(m.jobs)
        (j in queue) || push!(keep_idx, i)
    end
    m.jobs = m.jobs[sort(collect(keep_idx))]
    m.selected = clamp(m.selected, 1, max(length(m.jobs), 1))

    isempty(queue) && return
    first_job = popfirst!(queue)
    m.queue   = queue
    _spawn_run!(m, first_job)
end

function _skip_selected!(m::TuiModel)
    isempty(m.jobs) && return
    job = m.jobs[m.selected]
    deleteat!(m.jobs, m.selected)
    m.selected = clamp(m.selected, 1, max(length(m.jobs), 1))
    _spawn_skip!(m, job)
end

function _cancel_current!(m::TuiModel)
    rj = m.running
    rj === nothing && return
    request_cancel!(rj.cancel)
    m.status = "cancellation requested for $(rj.job.label)…"
    m.mode   = VIEW_RUNNING
end

function _cancel_all!(m::TuiModel)
    empty!(m.queue)
    _cancel_current!(m)
    m.status = "cancelling current + clearing queue"
end

# ── TaskEvent ingestion ──────────────────────────────────────────────

function update!(m::TuiModel, evt::TaskEvent)
    if evt.id == :refresh
        m.refreshing = false
        if evt.value isa Exception
            m.status = "refresh failed: $(sprint(showerror, evt.value))"
        else
            m.jobs = evt.value::Vector{Job}
            m.selected = clamp(m.selected, 1, max(length(m.jobs), 1))
            m.status   = "$(length(m.jobs)) job(s) pending"
        end
    elseif evt.id == :run
        # Job task finished — clean up `running`, drain the queue if any.
        rj = m.running
        m.running = nothing
        if rj !== nothing
            label = rj.job.label
            outcome = if evt.value isa Exception
                "errored: $(sprint(showerror, evt.value))"
            elseif evt.value === :cancelled
                "cancelled"
            else
                "completed"
            end
            m.status = "$(label) $(outcome)"
        end
        if !_start_next_in_queue!(m)
            m.mode = VIEW_LIST
        end
    elseif evt.id == :skip
        if evt.value isa Exception
            m.status = "skip failed: $(sprint(showerror, evt.value))"
        else
            m.status = "skipped $(_short(String(evt.value)))"
        end
    end
end

# Catch-all so other Tachikoma events (resize, mouse, etc.) don't error.
update!(m::TuiModel, evt::Event) = nothing

# ── View ─────────────────────────────────────────────────────────────

function view(m::TuiModel, f::Frame)
    m.tick += 1
    while isready(m.log_channel)
        push_line!(m.log_pane, take!(m.log_channel))
    end
    buf = f.buffer

    outer = Block(
        title        = " jimm-ci ",
        border_style = tstyle(:border),
        title_style  = tstyle(:title, bold = true),
    )
    inner = render(outer, f.area, buf)

    rows = split_layout(Layout(Vertical,
        [Fixed(1), Fill(), Fixed(1)]), inner)
    length(rows) < 3 && return
    header_area, body_area, status_area = rows[1], rows[2], rows[3]

    _render_header!(m, buf, header_area)

    if m.mode == VIEW_LIST
        _render_list!(m, buf, body_area)
    elseif m.mode == VIEW_RUNNING
        _render_running!(m, buf, body_area)
    elseif m.mode == VIEW_CONFIRM_CANCEL
        _render_running!(m, buf, body_area)
        _render_cancel_modal!(m, buf, body_area)
    end

    _render_status!(m, buf, status_area)
end

function _render_header!(m::TuiModel, buf, area)
    queue_n = length(m.queue)
    extra = queue_n > 0 ? "  queue: $queue_n" : ""
    spinner = if m.refreshing || m.running !== nothing
        si = mod1(m.tick ÷ 3, length(SPINNER_BRAILLE))
        string(SPINNER_BRAILLE[si], " ")
    else
        ""
    end
    title = "$(spinner)$(length(m.jobs)) pending$(extra)"
    set_string!(buf, area.x + 1, area.y, title, tstyle(:text); max_x = right(area))
end

function _render_list!(m::TuiModel, buf, area)
    if isempty(m.jobs)
        msg = "no jobs to run — press r to refresh"
        set_string!(buf, area.x + max(0, (area.width - length(msg)) ÷ 2),
                    area.y + area.height ÷ 2, msg, tstyle(:text_dim);
                    max_x = right(area))
        return
    end
    items = [ListItem(_row_text(j),
        j.kind == Jobs.PR_JOB ? tstyle(:primary) : tstyle(:secondary))
        for j in m.jobs]
    render(SelectableList(items;
        selected         = m.selected,
        offset           = m.list_offset,
        block            = Block(title        = " queue ",
                                  border_style = tstyle(:border),
                                  title_style  = tstyle(:title)),
        highlight_style  = tstyle(:accent, bold = true),
        tick             = m.tick,
    ), area, buf)
end

function _render_running!(m::TuiModel, buf, area)
    rj = m.running
    if rj === nothing
        # Job just finished but we haven't returned to LIST yet.
        render(m.log_pane, area, buf)
        return
    end
    rows = split_layout(Layout(Vertical, [Fixed(2), Fill()]), area)
    length(rows) < 2 && return
    info_area = rows[1]; pane_area = rows[2]

    elapsed = Dates.value(now(UTC) - rj.started_at) ÷ 1000
    info = "running $(rj.job.label)  elapsed $(elapsed)s  families=$(join(rj.job.families, ","))"
    set_string!(buf, info_area.x + 1, info_area.y, info,
                tstyle(:primary, bold = true); max_x = right(info_area))
    queue_n = length(m.queue)
    if queue_n > 0
        set_string!(buf, info_area.x + 1, info_area.y + 1,
                    "back-to-back queue: $queue_n more",
                    tstyle(:text_dim); max_x = right(info_area))
    end
    render(m.log_pane, pane_area, buf)
end

function _render_cancel_modal!(m::TuiModel, buf, area)
    w = min(60, area.width - 4)
    h = 7
    x = area.x + (area.width  - w) ÷ 2
    y = area.y + (area.height - h) ÷ 2
    # Background fill
    for cy in y:(y + h - 1), cx in x:(x + w - 1)
        in_bounds(buf, cx, cy) && set_char!(buf, cx, cy, ' ', tstyle(:text))
    end
    modal = Block(title = " cancel? ",
                  border_style = tstyle(:warning, bold = true),
                  title_style  = tstyle(:warning, bold = true))
    inner = render(modal, Rect(x, y, w, h), buf)
    set_string!(buf, inner.x + 2, inner.y + 1,
                "cancel the running build?",
                tstyle(:text); max_x = right(inner))
    set_string!(buf, inner.x + 2, inner.y + 3,
                "[y] cancel current   [a] cancel current + queue   [n] keep running",
                tstyle(:text_dim); max_x = right(inner))
end

function _render_status!(m::TuiModel, buf, area)
    hint = if m.mode == VIEW_LIST
        "[↑↓/jk] move  [Enter/y] run  [A] run all master  [s] skip  [r] refresh  [q] quit"
    elseif m.mode == VIEW_RUNNING
        "[c] cancel  [C] cancel all  [PgUp/PgDn] scroll"
    else
        "[y] yes  [a] cancel queue too  [n] no"
    end
    render(StatusBar(
        left  = [Span("  $(m.status)  ", tstyle(:text_dim))],
        right = [Span(hint * " ",        tstyle(:text_dim))],
    ), area, buf)
end

# ── Entry point ──────────────────────────────────────────────────────

"""
    run_tui(cfg, gh)

Launch the interactive TUI. Performs an initial discovery before the
first frame so the list is populated when the user sees it.
"""
function run_tui(cfg::Config, gh::GitHubApp)
    builder = Builder(cfg, gh)
    model = TuiModel(cfg = cfg, gh = gh, builder = builder)
    # Eager initial fetch — better UX than an empty list on first paint.
    try
        model.jobs   = discover_jobs(cfg, gh)
        model.status = "$(length(model.jobs)) job(s) pending"
    catch e
        model.status = "initial discovery failed: $(sprint(showerror, e))"
    end
    app(model; fps = 30)
    return nothing
end

end # module
