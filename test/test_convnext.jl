# ConvNeXt v1 parity tests against timm reference outputs.
#
# Each test:
#   1. Reads a Python-dumped HDF5 fixture
#      (data/parity/<variant>_io.h5) containing the deterministic input and
#      timm's forward_features (and logits, for variants with a trained head).
#   2. Downloads the variant's model.safetensors from HuggingFace (cached on
#      disk so reruns are fast and offline).
#   3. Builds the Jimm model, applies the weights, and asserts max-abs-diff
#      against the timm reference is under `LOGITS_ATOL`.
#
# Skipped if the fixture file is missing or HF_OFFLINE=1 is set (so CI without
# network or fixtures can still pass the rest of the suite).

using Test
using Jimm
using Lux
using Random

isdefined(@__MODULE__, :variant_filter) || include("_filter.jl")
isdefined(@__MODULE__, :LOGITS_ATOL) || include("_parity_tol.jl")

# Test every registered ConvNeXt v1 variant. Each entry is gated on its
# fixture file existing under data/parity/, so machines without the full
# set of dumps simply skip the missing variants. Deriving the list from
# CONVNEXT_VARIANTS keeps it in sync as new variants land in
# src/Models/ConvNeXt/Config.jl without a second edit here.
const VARIANTS_TO_TEST = Tuple(sort(collect(keys(Jimm.CONVNEXT_VARIANTS))))

function convnext_fixture_path(variant::Symbol; in_chans::Int = 3)
    suffix = in_chans == 3 ? "" : "_in$(in_chans)c"
    base = get(ENV, "JIMM_PARITY_DIR", joinpath(@__DIR__, "..", "data", "parity"))
    joinpath(base, "$(variant)$(suffix)_io.h5")
end

# The HDF5 fixture stores PyTorch-layout tensors; read_parity reverses axes
# to Lux-natural (W, H, C, N). The timm features for ConvNeXt land as
# (W/32, H/32, dims[end], N) after the axis reverse, which is exactly Jimm's
# output layout. Logits land as (K, N).
function convnext_load_fixture(variant::Symbol; in_chans::Int = 3)
    path = convnext_fixture_path(variant; in_chans = in_chans)
    isfile(path) || return nothing
    return Jimm.Interop.read_parity(path)
end

convnext_hf_offline() = get(ENV, "HF_OFFLINE", "") == "1"

@testset "ConvNeXt parity" begin
    for variant in variant_filter(VARIANTS_TO_TEST)
        @testset "$(variant)" begin
            fixture = convnext_load_fixture(variant)
            if fixture === nothing
                @info "skipping $variant: fixture missing at $(convnext_fixture_path(variant))"
                continue
            end
            if convnext_hf_offline()
                @info "skipping $variant: HF_OFFLINE=1"
                continue
            end

            cfg = Jimm.CONVNEXT_VARIANTS[variant]
            x = fixture.input
            expected_features = fixture.output["features"]
            # Input resolution is not asserted here: the model is fully
            # convolutional and accepts any size. The shape checks on `y`
            # against `expected_features` / `expected_logits` below cover
            # any size mismatch end-to-end.

            # features mode: num_classes = 0
            @testset "forward_features" begin
                model = create_model(variant; in_chans = 3, num_classes = 0)
                ps, st = Lux.setup(Xoshiro(0), model)
                st = Lux.testmode(st)
                ps, st = load_pretrained(ps, st, variant)
                y, _ = model(x, ps, st)
                @test size(y) == size(expected_features)
                diff = maximum(abs.(y .- expected_features))
                ref_scale = max(maximum(abs.(expected_features)), eps(Float32))
                rel = diff / ref_scale
                @info "$(variant) features max-abs-diff = $diff, rel = $rel"
                @test rel < FEATURES_RTOL
            end

            # logits mode: only run when the variant ships a trained head
            # (the `.fb_*` checkpoints; the DINOv3 encoders ship `num_classes=0`
            # and skip this block).
            if cfg.default_num_classes > 0 && haskey(fixture.output, "logits")
                expected_logits = fixture.output["logits"]
                @testset "forward (logits)" begin
                    model = create_model(variant;
                                      in_chans = 3,
                                      num_classes = cfg.default_num_classes)
                    ps, st = Lux.setup(Xoshiro(0), model)
                    st = Lux.testmode(st)
                    ps, st = load_pretrained(ps, st, variant)
                    y, _ = model(x, ps, st)
                    @test size(y) == size(expected_logits)
                    diff = maximum(abs.(y .- expected_logits))
                    @info "$(variant) logits max-abs-diff = $diff"
                    @test diff < LOGITS_ATOL
                end
            end

            # in_chans=1 parity: requires a separate fixture dumped from timm
            # with the model built using in_chans=1 (timm's adapt_input_conv
            # runs server-side at create_model time). The Julia side rebuilds
            # the same adaptation via Jimm's adapt_input_conv, triggered by
            # load_pretrained reading in_chans=1 from the model's stem weight.
            fixture_in1c = convnext_load_fixture(variant; in_chans = 1)
            if fixture_in1c === nothing
                @info "skipping $(variant) in_chans=1: fixture missing at " *
                      convnext_fixture_path(variant; in_chans = 1)
            else
                @testset "forward_features (in_chans=1)" begin
                    x1 = fixture_in1c.input
                    expected1 = fixture_in1c.output["features"]
                    model = create_model(variant; in_chans = 1, num_classes = 0)
                    ps, st = Lux.setup(Xoshiro(0), model)
                    st = Lux.testmode(st)
                    ps, st = load_pretrained(ps, st, variant)
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
