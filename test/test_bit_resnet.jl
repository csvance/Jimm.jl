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
isdefined(@__MODULE__, :run_variant_parity) || include("_parity_helpers.jl")

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

@testset "BiT parity" begin
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
