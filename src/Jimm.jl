module Jimm

include("Interop/Interop.jl")
include("Layers/Layers.jl")
include("Models/Models.jl")

using .Interop
using .Layers
using .Models

# Interop
export read_parity, apply_state_dict, axis_reverse, pyperm, as_channel4d,
       adapt_input_conv
export hf_download, hf_hub_download, hf_hub_cache_dir, default_cache_dir
export load_safetensors_state_dict

# Layers
export std_conv, layernorm2d, grn_layer,
       kaiming_normal_fan_out, normal_init

# Models
export BiTVariant, BIT_VARIANTS,
       ResNetVariant, RESNET_VARIANTS,
       ConvNeXtV2Variant, CONVNEXTV2_VARIANTS,
       ConvNeXtVariant, CONVNEXT_VARIANTS,
       create_model, create_pretrained, default_num_classes

end # module Jimm
