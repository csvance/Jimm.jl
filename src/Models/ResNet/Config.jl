# Variant catalog for classic timm ResNet.

"""
    ResNetVariant

Architectural config for a classic timm ResNet variant.
"""
struct ResNetVariant
    name::Symbol
    block::Symbol
    layers::NTuple{4, Int}
    planes::NTuple{4, Int}
    num_features::Int
    hf_repo::String
    default_num_classes::Int
    default_input_size::Int
end

"""
    RESNET_VARIANTS :: Dict{Symbol, ResNetVariant}

Lookup table for classic ResNet variants currently ported from timm.
Keys are the timm model names with dots rewritten as underscores.
"""
const RESNET_VARIANTS = Dict{Symbol, ResNetVariant}(
    :resnet18_a1_in1k => ResNetVariant(
        :resnet18_a1_in1k, :basic, (2, 2, 2, 2), (64, 128, 256, 512), 512,
        "timm/resnet18.a1_in1k", 1000, 224),
    :resnet34_a1_in1k => ResNetVariant(
        :resnet34_a1_in1k, :basic, (3, 4, 6, 3), (64, 128, 256, 512), 512,
        "timm/resnet34.a1_in1k", 1000, 224),
    :resnet50_a1_in1k => ResNetVariant(
        :resnet50_a1_in1k, :bottleneck, (3, 4, 6, 3), (64, 128, 256, 512), 2048,
        "timm/resnet50.a1_in1k", 1000, 224),
    :resnet101_a1_in1k => ResNetVariant(
        :resnet101_a1_in1k, :bottleneck, (3, 4, 23, 3), (64, 128, 256, 512), 2048,
        "timm/resnet101.a1_in1k", 1000, 224),
    :resnet152_a1_in1k => ResNetVariant(
        :resnet152_a1_in1k, :bottleneck, (3, 8, 36, 3), (64, 128, 256, 512), 2048,
        "timm/resnet152.a1_in1k", 1000, 224),
)
