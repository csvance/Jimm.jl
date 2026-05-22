# Classic ResNet parity tests against timm reference outputs.

using Test
using Jimm
using Lux
using Random

isdefined(@__MODULE__, :variant_filter) || include("_filter.jl")
isdefined(@__MODULE__, :run_variant_parity) || include("_parity_helpers.jl")

const RESNET_VARIANTS_TO_TEST = (
    :resnet18_a1_in1k,
    :resnet34_a1_in1k,
    :resnet50_a1_in1k,
    :resnet101_a1_in1k,
    :resnet152_a1_in1k,
)

@testset "ResNet parity" begin
    for variant in variant_filter(RESNET_VARIANTS_TO_TEST)
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
