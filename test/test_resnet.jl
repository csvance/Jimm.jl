# Classic ResNet parity tests against timm reference outputs.

using Test
using Jimm
using Lux
using Random

isdefined(@__MODULE__, :variant_filter) || include("_filter.jl")

isdefined(@__MODULE__, :LOGITS_ATOL) || include("_parity_tol.jl")
const RESNET_VARIANTS_TO_TEST = (
    :resnet18_a1_in1k,
    :resnet34_a1_in1k,
    :resnet50_a1_in1k,
    :resnet101_a1_in1k,
    :resnet152_a1_in1k,
)

function resnet_fixture_path(variant::Symbol; in_chans::Int = 3)
    suffix = in_chans == 3 ? "" : "_in$(in_chans)c"
    base = get(ENV, "JIMM_PARITY_DIR", joinpath(@__DIR__, "..", "data", "parity"))
    joinpath(base, "$(variant)$(suffix)_io.h5")
end

function load_resnet_fixture(variant::Symbol; in_chans::Int = 3)
    path = resnet_fixture_path(variant; in_chans = in_chans)
    isfile(path) || return nothing
    return Jimm.Interop.read_parity(path)
end

resnet_hf_offline() = get(ENV, "HF_OFFLINE", "") == "1"

@testset "ResNet parity" begin
    for variant in variant_filter(RESNET_VARIANTS_TO_TEST)
        @testset "$(variant)" begin
            fixture = load_resnet_fixture(variant)
            if fixture === nothing
                @info "skipping $variant: fixture missing at $(resnet_fixture_path(variant))"
                continue
            end
            if resnet_hf_offline()
                @info "skipping $variant: HF_OFFLINE=1"
                continue
            end

            x = fixture.input
            expected_features = fixture.output["features"]
            expected_logits = fixture.output["logits"]

            @testset "forward_features" begin
                model = create_model(variant; in_chans = 3, num_classes = 0)
                ps, st = Lux.setup(Xoshiro(0), model)
                ps, st = load_pretrained(ps, st, variant)
                st = Lux.testmode(st)
                y, _ = model(x, ps, st)
                @test size(y) == size(expected_features)
                diff = maximum(abs.(y .- expected_features))
                ref_scale = max(maximum(abs.(expected_features)), eps(Float32))
                rel = diff / ref_scale
                @info "$(variant) features max-abs-diff = $diff, rel = $rel"
                @test rel < FEATURES_RTOL
            end

            @testset "forward (logits)" begin
                cfg = Jimm.RESNET_VARIANTS[variant]
                model = create_model(variant; in_chans = 3,
                               num_classes = cfg.default_num_classes)
                ps, st = Lux.setup(Xoshiro(0), model)
                ps, st = load_pretrained(ps, st, variant)
                st = Lux.testmode(st)
                y, _ = model(x, ps, st)
                @test size(y) == size(expected_logits)
                diff = maximum(abs.(y .- expected_logits))
                @info "$(variant) logits max-abs-diff = $diff"
                @test diff < LOGITS_ATOL
            end

            fixture_in1c = load_resnet_fixture(variant; in_chans = 1)
            if fixture_in1c === nothing
                @info "skipping $(variant) in_chans=1: fixture missing at " *
                      resnet_fixture_path(variant; in_chans = 1)
            else
                @testset "forward_features (in_chans=1)" begin
                    x1 = fixture_in1c.input
                    expected1 = fixture_in1c.output["features"]
                    model = create_model(variant; in_chans = 1, num_classes = 0)
                    ps, st = Lux.setup(Xoshiro(0), model)
                    ps, st = load_pretrained(ps, st, variant)
                    st = Lux.testmode(st)
                    y, _ = model(x1, ps, st)
                    @test size(y) == size(expected1)
                    diff = maximum(abs.(y .- expected1))
                    ref_scale = max(maximum(abs.(expected1)), eps(Float32))
                    rel = diff / ref_scale
                    @info "$(variant) features (in_chans=1) max-abs-diff = $diff, rel = $rel"
                    @test rel < FEATURES_RTOL
                end
            end
        end
    end
end
