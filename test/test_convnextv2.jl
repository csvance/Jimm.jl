# ConvNeXtV2 parity tests against timm reference outputs.
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
isdefined(@__MODULE__, :run_variant_parity) || include("_parity_helpers.jl")

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

@testset "ConvNeXtV2 parity" begin
    for variant in variant_filter(VARIANTS_TO_TEST)
        @testset "$(variant)" begin
            fixture = load_parity_fixture(variant)
            if fixture === nothing
                @info "skipping $variant: fixture missing at $(parity_fixture_path(variant))"
                continue
            end
            if hf_offline()
                @info "skipping $variant: HF_OFFLINE=1"
                continue
            end
            run_variant_parity(variant, fixture)
        end
    end
end
