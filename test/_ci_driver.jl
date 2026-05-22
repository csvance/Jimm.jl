# Multi-family test driver — runs every family selected by JIMM_TEST_FAMILIES
# inside a single Julia process so JIT/compile caches are reused across
# backbones that share layers.
#
# This is invoked from two paths:
#
#   1. `Pkg.test()` (local dev): `runtests.jl` is a one-line shim that
#      includes this file. `JIMM_TEST_FAMILIES` / `JIMM_TEST_VARIANTS`
#      gate which families run.
#
#   2. The CI builder (`ci/JimmCI/src/Builder.jl`): bypasses Pkg.test
#      entirely and invokes Julia with
#      `julia --project=. -e 'include("test/_ci_driver.jl")'`, then watches
#      stdout for the structured markers below to drive per-family GitHub
#      check_runs.
#
# Marker contract (kept stable; the CI builder greps for these literals):
#
#   ==> JIMM_FAMILY_BEGIN: family=<name>
#   ==> JIMM_FAMILY_END:   family=<name> rc=<0|1>
#
# `rc=0` iff every `@test` in the family passed and the test file ran to
# completion without throwing. Any uncaught error during `include` is
# reported as `rc=1` and the next family still runs.

using Test
using Jimm

isdefined(@__MODULE__, :family_enabled) || include("_filter.jl")

# Canonical run order. Matches PathFilter.ALL_FAMILIES so the CI's check_run
# list shows up in the same sequence regardless of how families were passed
# in via JIMM_TEST_FAMILIES.
const _DRIVER_ORDER = ("infra", "bit", "resnet", "convnext", "convnextv2")

function _scaffold_testset()
    @testset "Jimm scaffold" begin
        @test isdefined(Jimm, :Interop)
        @test isdefined(Jimm, :Layers)
        @test isdefined(Jimm, :Models)
        @test isdefined(Jimm.Interop, :read_parity)
        @test isdefined(Jimm.Interop, :apply_state_dict)
        @test isdefined(Jimm.Interop, :hf_download)
        @test isdefined(Jimm.Interop, :hf_hub_download)
        @test isdefined(Jimm.Interop, :hf_hub_cache_dir)
        @test isdefined(Jimm.Interop, :load_safetensors_state_dict)
        @test isdefined(Jimm.Interop, :adapt_input_conv)
        @test isdefined(Jimm.Layers, :std_conv)
        @test isdefined(Jimm.Layers, :layernorm2d)
        @test isdefined(Jimm.Layers, :grn_layer)
        @test isdefined(Jimm.Models, :bit_resnetv2)
        @test isdefined(Jimm.Models, :bit_resnetv2_mapping)
        @test isdefined(Jimm.Models, :load_bit_resnetv2_pretrained)
        @test isdefined(Jimm.Models, :BIT_VARIANTS)
        @test isdefined(Jimm.Models, :resnet)
        @test isdefined(Jimm.Models, :resnet_mapping)
        @test isdefined(Jimm.Models, :resnet_state_mapping)
        @test isdefined(Jimm.Models, :load_resnet_pretrained)
        @test isdefined(Jimm.Models, :RESNET_VARIANTS)
        @test isdefined(Jimm.Models, :convnextv2)
        @test isdefined(Jimm.Models, :convnextv2_mapping)
        @test isdefined(Jimm.Models, :load_convnextv2_pretrained)
        @test isdefined(Jimm.Models, :CONVNEXTV2_VARIANTS)
        @test isdefined(Jimm.Models, :convnext)
        @test isdefined(Jimm.Models, :convnext_mapping)
        @test isdefined(Jimm.Models, :load_convnext_pretrained)
        @test isdefined(Jimm.Models, :CONVNEXT_VARIANTS)
    end
end

function _dispatch_family(name::AbstractString)
    if name == "infra"
        _scaffold_testset()
        include(joinpath(@__DIR__, "test_hf_download.jl"))
        include(joinpath(@__DIR__, "test_hf_hub_download.jl"))
        include(joinpath(@__DIR__, "test_init.jl"))
    elseif name == "bit"
        include(joinpath(@__DIR__, "test_bit_resnet.jl"))
    elseif name == "resnet"
        include(joinpath(@__DIR__, "test_resnet.jl"))
    elseif name == "convnext"
        include(joinpath(@__DIR__, "test_convnext.jl"))
    elseif name == "convnextv2"
        include(joinpath(@__DIR__, "test_convnextv2.jl"))
    else
        error("unknown family in JIMM_TEST_FAMILIES: $(name)")
    end
end

# Run one family inside its own outer `@testset`. `@testset` rethrows
# `TestSetException` at end-of-block if any inner `@test` failed; anything
# else escaping `_dispatch_family` is a service-level error (e.g. uncaught
# exception during include). Both map to rc=1.
function _run_family(name::AbstractString)
    failed = false
    try
        @testset "$(name)" begin
            _dispatch_family(name)
        end
    catch e
        failed = true
        if !(e isa Test.TestSetException)
            println(stderr, "ERROR in family $(name): ", sprint(showerror, e))
            Base.show_backtrace(stderr, catch_backtrace())
            println(stderr)
        end
    end
    return failed ? 1 : 0
end

function _driver_main()
    requested = families_enabled()
    ordered   = [f for f in _DRIVER_ORDER if f in requested]
    overall   = 0
    for fam in ordered
        println(stdout, "==> JIMM_FAMILY_BEGIN: family=$(fam)"); flush(stdout)
        rc = _run_family(fam)
        rc == 0 || (overall = 1)
        println(stdout, "==> JIMM_FAMILY_END: family=$(fam) rc=$(rc)"); flush(stdout)
    end
    return overall
end

exit(_driver_main())