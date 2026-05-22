```@meta
CurrentModule = Jimm
```

# Models

## Family-agnostic interface

Every variant is built and loaded through the symbol-dispatched
`create_model` / `load_pretrained` pair. The canonical pattern is the
three-step idiom:

```julia
model = create_model(variant; in_chans, num_classes)
ps, st = Lux.setup(rng, model)
ps, st = load_pretrained(ps, st, variant)
```

`create_model` returns a bare `@compact` model with no parameters or
state. `load_pretrained` reads `in_chans` and `num_classes` directly
from the model's stem and head shapes, so you only specify them once
at the constructor. Keeping the constructor pure means it composes
inside a larger `@compact` block without redundant `Lux.setup` work —
see [Getting Started](../getting_started.md#composing-into-a-larger-model)
for the nested pattern with `prefix`.

```@docs
create_model
load_pretrained
```

## Per-family interface

Each family also exposes a direct constructor and loader, sharing the
same `(ps, st, variant; kwargs...) -> (ps, st)` shape:

```julia
model = <family>(variant; in_chans, num_classes)
ps, st = Lux.setup(rng, model)
ps, st = load_<family>_pretrained(ps, st, variant)
```

`<family>` is one of `resnet`, `bit_resnetv2`, `convnext`,
`convnextv2`. `variant` is a Julia symbol matching a key in the
corresponding `<FAMILY>_VARIANTS` dictionary.

## ResNet

```@docs
resnet
load_resnet_pretrained
resnet_mapping
resnet_state_mapping
ResNetVariant
RESNET_VARIANTS
```

## BiT ResNetV2

```@docs
bit_resnetv2
load_bit_resnetv2_pretrained
bit_resnetv2_mapping
BiTVariant
BIT_VARIANTS
```

## ConvNeXt

```@docs
convnext
load_convnext_pretrained
convnext_mapping
ConvNeXtVariant
CONVNEXT_VARIANTS
```

## ConvNeXt V2

```@docs
convnextv2
load_convnextv2_pretrained
convnextv2_mapping
ConvNeXtV2Variant
CONVNEXTV2_VARIANTS
```
