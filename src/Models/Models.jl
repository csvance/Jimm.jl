module Models

using Lux
using NNlib
using ..Layers
using ..Interop: apply_state_dict, axis_reverse, hf_hub_download,
                  hf_hub_cache_dir, load_safetensors_state_dict,
                  as_channel4d, adapt_input_conv

# Navigate `ps` (or `st`) to the subtree addressed by `prefix`. Used by
# every `load_<family>_pretrained` to introspect the model's `in_chans`
# and `num_classes` directly from the parameter shapes, so the caller
# doesn't have to repeat what they already told `create_model`.
function _navigate(tree, prefix::Tuple{Vararg{Symbol}})
    sub = tree
    for p in prefix
        sub = getproperty(sub, p)
    end
    return sub
end

# Shared ConvNeXt v1/v2 building blocks must be included before either
# family's Model.jl, since both reference `_CN_INIT`, `convnext_stage`, the
# mapping-entry builders, etc.
include("ConvNeXtCommon/Common.jl")

include("ResNetV2/Model.jl")
include("ResNet/Model.jl")
include("ConvNeXtV2/Model.jl")
include("ConvNeXt/Model.jl")

"""
    load_pretrained(ps, st, variant; kwargs...) -> (ps, st)

Family-agnostic pretrained-weight loader. Dispatches on `variant` to the
matching `load_<family>_pretrained` based on which `<FAMILY>_VARIANTS`
dict owns the symbol. All four family loaders share the
`(ps, st, variant; kwargs...) -> (ps, st)` signature, so calls flow
through unchanged. `kwargs` are forwarded as-is; supported keys are
`num_classes`, `in_chans`, `revision`, `cache_dir`, `prefix`.
"""
function load_pretrained(ps, st, variant::Symbol; kwargs...)
    if haskey(BIT_VARIANTS, variant)
        return load_bit_resnetv2_pretrained(ps, st, variant; kwargs...)
    elseif haskey(RESNET_VARIANTS, variant)
        return load_resnet_pretrained(ps, st, variant; kwargs...)
    elseif haskey(CONVNEXT_VARIANTS, variant)
        return load_convnext_pretrained(ps, st, variant; kwargs...)
    elseif haskey(CONVNEXTV2_VARIANTS, variant)
        return load_convnextv2_pretrained(ps, st, variant; kwargs...)
    else
        error("Unknown variant: $variant. Not found in any of " *
              "BIT_VARIANTS, RESNET_VARIANTS, CONVNEXT_VARIANTS, " *
              "CONVNEXTV2_VARIANTS.")
    end
end

"""
    create_model(variant; kwargs...) -> model

Family-agnostic model constructor, mirroring `timm.create_model`.
Dispatches on `variant` to the matching family constructor and returns
the bare `@compact` model — no parameters, no state.

The canonical three-step pattern for using a pretrained variant
standalone is:

```julia
model = create_model(variant; in_chans = 3, num_classes = 1000)
ps, st = Lux.setup(rng, model)
ps, st = load_pretrained(ps, st, variant)
```

For composition into a larger model, embed `create_model` inside an
outer `@compact` block, run `Lux.setup` once on the composed tree,
then load the pretrained slot with `prefix`:

```julia
outer = @compact(backbone = create_model(:resnet50_a1_in1k; num_classes = 0),
                 head = Dense(2048 => num_outputs)) do x
    head(backbone(x))
end
ps, st = Lux.setup(rng, outer)
ps, st = load_pretrained(ps, st, :resnet50_a1_in1k; prefix = (:backbone,))
```

`kwargs` are forwarded to the family constructor (`in_chans`,
`num_classes`).
"""
function create_model(variant::Symbol; kwargs...)
    if haskey(BIT_VARIANTS, variant)
        return bit_resnetv2(variant; kwargs...)
    elseif haskey(RESNET_VARIANTS, variant)
        return resnet(variant; kwargs...)
    elseif haskey(CONVNEXT_VARIANTS, variant)
        return convnext(variant; kwargs...)
    elseif haskey(CONVNEXTV2_VARIANTS, variant)
        return convnextv2(variant; kwargs...)
    else
        error("Unknown variant: $variant. Not found in any of " *
              "BIT_VARIANTS, RESNET_VARIANTS, CONVNEXT_VARIANTS, " *
              "CONVNEXTV2_VARIANTS.")
    end
end

export bit_resnetv2, bit_resnetv2_mapping, load_bit_resnetv2_pretrained,
       BiTVariant, BIT_VARIANTS,
       resnet, resnet_mapping, resnet_state_mapping, load_resnet_pretrained,
       ResNetVariant, RESNET_VARIANTS,
       convnextv2, convnextv2_mapping, load_convnextv2_pretrained,
       ConvNeXtV2Variant, CONVNEXTV2_VARIANTS,
       convnext, convnext_mapping, load_convnext_pretrained,
       ConvNeXtVariant, CONVNEXT_VARIANTS,
       create_model, load_pretrained

end # module Models
