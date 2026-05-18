# Verifies that `hf_download` cooperates with Julia's task scheduler:
# a co-resident `@async` heartbeat task scheduled on the same OS thread
# as the download caller must keep ticking while the download is in
# flight. `Downloads.download` (stdlib) is supposed to wait on libuv
# events and yield between them, so the heartbeat should not stall.
#
# This test catches regressions where a future change to `hf_download`
# (e.g. swapping to a blocking libcurl call, or adding a sync
# `Base.fetch` over an FFI that doesn't yield) would starve other tasks
# on the caller's thread.

using Test
using Jimm

# Tiny file from one of the BiT repos used elsewhere in the suite.
# Bytes-wise it's ~570B; the dominant cost is the HTTPS round-trip,
# which gives the scheduler plenty of opportunities to interleave the
# heartbeat task.
const TEST_URL = "https://huggingface.co/timm/resnetv2_50x1_bit.goog_in21k/resolve/main/config.json"

@testset "hf_download cooperates with the scheduler" begin
    if get(ENV, "HF_OFFLINE", "") == "1"
        @info "skipping: HF_OFFLINE=1"
        return
    end

    # Force a real download every run; the cached-hit fast path in
    # hf_download returns immediately, which would make the test
    # meaningless.
    dest = joinpath(tempdir(),
                    "jimm_hf_download_nonblock_$(rand(UInt32)).json")
    isfile(dest) && rm(dest)

    counter = Threads.Atomic{Int}(0)
    stop = Threads.Atomic{Bool}(false)

    # @async (not Threads.@spawn) is intentional: it pins the heartbeat
    # to the same thread the test task lives on. If hf_download were
    # blocking that thread, the heartbeat would not tick. Threads.@spawn
    # would always tick on a multi-threaded Julia since the runtime
    # could migrate it to a free thread; @async is the strict test.
    heartbeat = @async begin
        while !stop[]
            Threads.atomic_add!(counter, 1)
            sleep(0.005)
        end
    end

    try
        sleep(0.05)                # warm up the heartbeat
        baseline = counter[]
        @test baseline > 0          # sanity: scheduler is alive

        hf_download(TEST_URL, dest)

        after = counter[]
        @test isfile(dest)
        # Headline assertion: heartbeat ticked at least once during the
        # download. In practice we observe many more ticks (HTTPS
        # round-trip is ~100-300ms, heartbeat is every 5ms), but we keep
        # the bound to 1 to avoid flakes on fast / cached connections.
        @info "hf_download cooperative-scheduling check: ticks during download = $(after - baseline)"
        @test after - baseline > 0
    finally
        stop[] = true
        wait(heartbeat)
        isfile(dest) && rm(dest)
    end
end
