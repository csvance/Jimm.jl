# ConvNeXtV2 parity tests against timm reference outputs.
#
# Each test:
#   1. Reads a Python-dumped HDF5 fixture
#      (data/parity/<variant>_io.h5) containing the deterministic input and
#      timm's forward_features (and logits, for variants with a trained head).
#   2. Downloads the variant's model.safetensors from HuggingFace (cached on
#      disk so reruns are fast and offline).
#   3. Builds the Jimm model, applies the weights, and asserts max-abs-diff
#      against the timm reference is under `TOL`.
#
# Skipped if the fixture file is missing or HF_OFFLINE=1 is set (so CI without
# network or fixtures can still pass the rest of the suite).

using Test
using Jimm
using Lux
using Random

isdefined(@__MODULE__, :variant_filter) || include("_filter.jl")

const TOL = 1f-3

# Variant keys mirror the timm model name with the dot rewritten as an
# underscore; must match CONVNEXTV2_VARIANTS keys and the sidecar's
# CONVNEXTV2_VARIANTS_FULL keys. Each entry is gated on its fixture file
# existing under data/parity/, so machines without the full set of dumps
# simply skip the missing variants.
const VARIANTS_TO_TEST = (
    # atto
    :convnextv2_atto_fcmae,
    :convnextv2_atto_fcmae_ft_in1k,
    # femto
    :convnextv2_femto_fcmae,
    :convnextv2_femto_fcmae_ft_in1k,
    # pico
    :convnextv2_pico_fcmae,
    :convnextv2_pico_fcmae_ft_in1k,
    # nano
    :convnextv2_nano_fcmae,
    :convnextv2_nano_fcmae_ft_in1k,
    :convnextv2_nano_fcmae_ft_in22k_in1k,
    :convnextv2_nano_fcmae_ft_in22k_in1k_384,
    # tiny
    :convnextv2_tiny_fcmae,
    :convnextv2_tiny_fcmae_ft_in1k,
    :convnextv2_tiny_fcmae_ft_in22k_in1k,
    :convnextv2_tiny_fcmae_ft_in22k_in1k_384,
    # base
    :convnextv2_base_fcmae,
    :convnextv2_base_fcmae_ft_in1k,
    :convnextv2_base_fcmae_ft_in22k_in1k,
    :convnextv2_base_fcmae_ft_in22k_in1k_384,
    # large
    :convnextv2_large_fcmae,
    :convnextv2_large_fcmae_ft_in1k,
    :convnextv2_large_fcmae_ft_in22k_in1k,
    :convnextv2_large_fcmae_ft_in22k_in1k_384,
    # huge
    :convnextv2_huge_fcmae,
    :convnextv2_huge_fcmae_ft_in1k,
    :convnextv2_huge_fcmae_ft_in22k_in1k_384,
    :convnextv2_huge_fcmae_ft_in22k_in1k_512,
)

function convnextv2_fixture_path(variant::Symbol; in_chans::Int = 3)
    suffix = in_chans == 3 ? "" : "_in$(in_chans)c"
    joinpath(@__DIR__, "..", "data", "parity", "$(variant)$(suffix)_io.h5")
end

# The HDF5 fixture stores PyTorch-layout tensors; read_parity reverses axes
# to Lux-natural (W, H, C, N). The timm features for ConvNeXtV2 land as
# (W/32, H/32, dims[end], N) after the axis reverse, which is exactly Jimm's
# output layout. Logits land as (K, N).
function convnextv2_load_fixture(variant::Symbol; in_chans::Int = 3)
    path = convnextv2_fixture_path(variant; in_chans = in_chans)
    isfile(path) || return nothing
    return Jimm.Interop.read_parity(path)
end

convnextv2_hf_offline() = get(ENV, "HF_OFFLINE", "") == "1"

@testset "ConvNeXtV2 parity" begin
    for variant in variant_filter(VARIANTS_TO_TEST)
        @testset "$(variant)" begin
            fixture = convnextv2_load_fixture(variant)
            if fixture === nothing
                @info "skipping $variant: fixture missing at $(convnextv2_fixture_path(variant))"
                continue
            end
            if convnextv2_hf_offline()
                @info "skipping $variant: HF_OFFLINE=1"
                continue
            end

            cfg = Jimm.CONVNEXTV2_VARIANTS[variant]
            x = fixture.input
            expected_features = fixture.output["features"]
            # Input resolution is not asserted here: the model is fully
            # convolutional and accepts any size, and the `_384` / `_512`
            # checkpoints' fixtures use their native resolution. The shape
            # checks on `y` against `expected_features` / `expected_logits`
            # below already cover any size mismatch end-to-end.

            # features mode: num_classes = 0
            @testset "forward_features" begin
                model = convnextv2(variant; in_chans = 3, num_classes = 0)
                ps, st = Lux.setup(Xoshiro(0), model)
                st = Lux.testmode(st)
                ps = load_convnextv2_pretrained(ps, variant; num_classes = 0)
                y, _ = model(x, ps, st)
                @test size(y) == size(expected_features)
                diff = maximum(abs.(y .- expected_features))
                @info "$(variant) features max-abs-diff = $diff"
                @test diff < TOL
            end

            # logits mode: only run when the variant ships a trained head.
            if cfg.default_num_classes > 0 && haskey(fixture.output, "logits")
                expected_logits = fixture.output["logits"]
                @testset "forward (logits)" begin
                    model = convnextv2(variant;
                                       in_chans = 3,
                                       num_classes = cfg.default_num_classes)
                    ps, st = Lux.setup(Xoshiro(0), model)
                    st = Lux.testmode(st)
                    ps = load_convnextv2_pretrained(ps, variant;
                                                    num_classes = cfg.default_num_classes)
                    y, _ = model(x, ps, st)
                    @test size(y) == size(expected_logits)
                    diff = maximum(abs.(y .- expected_logits))
                    @info "$(variant) logits max-abs-diff = $diff"
                    @test diff < TOL
                end
            end

            # in_chans=1 parity: requires a separate fixture dumped from timm
            # with the model built using in_chans=1 (timm's adapt_input_conv
            # runs server-side at create_model time). The Julia side rebuilds
            # the same adaptation via Jimm's adapt_input_conv plumbed through
            # load_convnextv2_pretrained(...; in_chans=1).
            fixture_in1c = convnextv2_load_fixture(variant; in_chans = 1)
            if fixture_in1c === nothing
                @info "skipping $(variant) in_chans=1: fixture missing at " *
                      convnextv2_fixture_path(variant; in_chans = 1)
            else
                @testset "forward_features (in_chans=1)" begin
                    x1 = fixture_in1c.input
                    expected1 = fixture_in1c.output["features"]
                    model = convnextv2(variant; in_chans = 1, num_classes = 0)
                    ps, st = Lux.setup(Xoshiro(0), model)
                    st = Lux.testmode(st)
                    ps = load_convnextv2_pretrained(ps, variant;
                                                    num_classes = 0,
                                                    in_chans = 1)
                    y, _ = model(x1, ps, st)
                    @test size(y) == size(expected1)
                    diff = maximum(abs.(y .- expected1))
                    @info "$(variant) features (in_chans=1) max-abs-diff = $diff"
                    @test diff < TOL
                end
            end
        end
    end
end
