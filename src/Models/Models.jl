module Models

using Lux
using NNlib
using ..Layers
using ..Interop: apply_state_dict, axis_reverse, hf_hub_download,
                  hf_hub_cache_dir, load_safetensors_state_dict,
                  as_channel4d, adapt_input_conv

# Shared ConvNeXt v1/v2 building blocks must be included before either
# family's Model.jl, since both reference `_CN_INIT`, `convnext_stage`, the
# mapping-entry builders, etc.
include("ConvNeXtCommon/Common.jl")

include("ResNetV2/Model.jl")
include("ResNet/Model.jl")
include("ConvNeXtV2/Model.jl")
include("ConvNeXt/Model.jl")

export bit_resnetv2, bit_resnetv2_mapping, load_bit_resnetv2_pretrained,
       BiTVariant, BIT_VARIANTS,
       resnet, resnet_mapping, resnet_state_mapping, load_resnet_pretrained,
       ResNetVariant, RESNET_VARIANTS,
       convnextv2, convnextv2_mapping, load_convnextv2_pretrained,
       ConvNeXtV2Variant, CONVNEXTV2_VARIANTS,
       convnext, convnext_mapping, load_convnext_pretrained,
       ConvNeXtVariant, CONVNEXT_VARIANTS

end # module Models
