# BiT ResNetV2 parity tests against timm reference outputs.
#
# Each test:
#   1. Reads a Python-dumped HDF5 fixture (data/parity/resnetv2_<key>_bit_io.h5)
#      containing the deterministic input and timm's forward_features /
#      forward outputs for that variant.
#   2. Downloads the variant's model.safetensors from HuggingFace (cached on
#      disk so reruns are fast and offline).
#   3. Builds the Jimm model, applies the weights, and asserts max-abs-diff
#      against the timm reference is under `LOGITS_ATOL`.
#
# Skipped if the fixture file is missing or HF_OFFLINE=1 is set in env (so
# CI without network or fixtures can still pass the rest of the suite).

using Test
using Jimm
using Lux
using Random

isdefined(@__MODULE__, :variant_filter) || include("_filter.jl")
isdefined(@__MODULE__, :LOGITS_ATOL) || include("_parity_tol.jl")

# Variant keys mirror the timm model name with the dot rewritten as an
# underscore; must match BIT_VARIANTS keys and the sidecar's
# BIT_VARIANTS_FULL keys.
const VARIANTS_TO_TEST = (
    :resnetv2_50x1_bit_goog_in21k,
    :resnetv2_50x3_bit_goog_in21k,
    :resnetv2_101x1_bit_goog_in21k,
    :resnetv2_101x3_bit_goog_in21k,
    :resnetv2_152x2_bit_goog_in21k,
    :resnetv2_152x4_bit_goog_in21k,
    :resnetv2_50x1_bit_goog_distilled_in1k,
    :resnetv2_50x1_bit_goog_in21k_ft_in1k,
    :resnetv2_50x3_bit_goog_in21k_ft_in1k,
    :resnetv2_101x1_bit_goog_in21k_ft_in1k,
    :resnetv2_101x3_bit_goog_in21k_ft_in1k,
    :resnetv2_152x2_bit_goog_in21k_ft_in1k,
    :resnetv2_152x4_bit_goog_in21k_ft_in1k,
    :resnetv2_152x2_bit_goog_teacher_in21k_ft_in1k,
    :resnetv2_152x2_bit_goog_teacher_in21k_ft_in1k_384,
)

function fixture_path(variant::Symbol; in_chans::Int = 3)
    suffix = in_chans == 3 ? "" : "_in$(in_chans)c"
    base = get(ENV, "JIMM_PARITY_DIR", joinpath(@__DIR__, "..", "data", "parity"))
    joinpath(base, "$(variant)$(suffix)_io.h5")
end

# The HDF5 fixture stores PyTorch-layout tensors; read_parity reverses
# axes to Lux-natural (W, H, C, N). The timm outputs for features land as
# (W/32, H/32, C, N) and logits as (K, N) after the axis reverse, which is
# exactly Jimm's output layout.
function load_fixture(variant::Symbol; in_chans::Int = 3)
    path = fixture_path(variant; in_chans = in_chans)
    isfile(path) || return nothing
    return Jimm.Interop.read_parity(path)
end

hf_offline() = get(ENV, "HF_OFFLINE", "") == "1"

@testset "BiT parity" begin
    for variant in variant_filter(VARIANTS_TO_TEST)
        @testset "$(variant)" begin
            fixture = load_fixture(variant)
            if fixture === nothing
                @info "skipping $variant: fixture missing at $(fixture_path(variant))"
                continue
            end
            if hf_offline()
                @info "skipping $variant: HF_OFFLINE=1"
                continue
            end

            x = fixture.input
            expected_features = fixture.output["features"]
            expected_logits   = fixture.output["logits"]
            # Input resolution is not asserted here: the model is fully
            # convolutional and the `_384` teacher variant uses its native
            # resolution. The shape checks on `y` against `expected_features`
            # / `expected_logits` below already cover size mismatch
            # end-to-end.

            # features mode: num_classes = 0
            @testset "forward_features" begin
                model = bit_resnetv2(variant; in_chans = 3, num_classes = 0)
                ps, st = Lux.setup(Xoshiro(0), model)
                ps = load_bit_resnetv2_pretrained(ps, variant; num_classes = 0)
                y, _ = model(x, ps, st)
                @test size(y) == size(expected_features)
                diff = maximum(abs.(y .- expected_features))
                ref_scale = max(maximum(abs.(expected_features)), eps(Float32))
                rel = diff / ref_scale
                @info "$(variant) features max-abs-diff = $diff, rel = $rel"
                @test rel < FEATURES_RTOL
            end

            # logits mode: num_classes = 21843
            @testset "forward (logits)" begin
                cfg = Jimm.BIT_VARIANTS[variant]
                model = bit_resnetv2(variant;
                                     in_chans = 3,
                                     num_classes = cfg.default_num_classes)
                ps, st = Lux.setup(Xoshiro(0), model)
                ps = load_bit_resnetv2_pretrained(ps, variant;
                                                  num_classes = cfg.default_num_classes)
                y, _ = model(x, ps, st)
                @test size(y) == size(expected_logits)
                diff = maximum(abs.(y .- expected_logits))
                @info "$(variant) logits max-abs-diff = $diff"
                @test diff < LOGITS_ATOL
            end

            # in_chans=1 parity: requires a separate fixture dumped from timm
            # with the model built using in_chans=1 (timm's adapt_input_conv
            # runs server-side at create_model time). The Julia side rebuilds
            # the same adaptation via Jimm's adapt_input_conv plumbed through
            # load_bit_resnetv2_pretrained(...; in_chans=1).
            fixture_in1c = load_fixture(variant; in_chans = 1)
            if fixture_in1c === nothing
                @info "skipping $(variant) in_chans=1: fixture missing at " *
                      fixture_path(variant; in_chans = 1)
            else
                @testset "forward_features (in_chans=1)" begin
                    x1 = fixture_in1c.input
                    expected1 = fixture_in1c.output["features"]
                    model = bit_resnetv2(variant; in_chans = 1, num_classes = 0)
                    ps, st = Lux.setup(Xoshiro(0), model)
                    ps = load_bit_resnetv2_pretrained(ps, variant;
                                                      num_classes = 0,
                                                      in_chans = 1)
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
