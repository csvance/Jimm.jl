```@meta
CurrentModule = Jimm
```

# Models

Every model family follows the same three-call user interface:

```julia
model = <family>(variant; in_chans, num_classes)
ps, st = Lux.setup(rng, model)
ps = load_<family>_pretrained(ps, variant; num_classes, in_chans)
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
