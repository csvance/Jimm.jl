```@meta
CurrentModule = Jimm
```

# API Reference

Jimm.jl exposes three groups of public symbols, all re-exported from
the top-level `Jimm` module.

| Group | Purpose | Page |
|---|---|---|
| Models | The family-agnostic `create_model` / `create_pretrained` pair, variant tables, and per-variant config structs. | [Models](models.md) |
| Layers | Building blocks shared across families: weight-standardized conv, channel-axis LayerNorm, Global Response Norm, and timm-equivalent initializers. | [Layers](layers.md) |
| Interop | PyTorch and HuggingFace plumbing: HDF5 parity fixtures, `state_dict` application, axis transforms, HuggingFace Hub cache, SafeTensors loading. | [Interop](interop.md) |

The top-level `Jimm` module re-exports every public symbol from
these three groups, so end users normally just call
`using Jimm`. The submodule-qualified names (`Jimm.Interop.read_parity`,
`Jimm.Layers.std_conv`) remain available for cases where the
shorter top-level name would collide with something else in the
caller's namespace.

## Module map

```
Jimm
├── Jimm.Models    # family constructors, loaders, variant tables
├── Jimm.Layers    # std_conv, layernorm2d, grn_layer, init recipes
└── Jimm.Interop   # parity, HF Hub, SafeTensors
```

Internal block constructors (`resnet_basic_block`,
`convnextv2_block`, the stage scaffolding under `ConvNeXtCommon`,
and the per-family `resnet` / `bit_resnetv2` / `convnext` /
`convnextv2` constructors themselves) are deliberately unexported.
They are implementation details of `create_model` and
`create_pretrained`; rely on those two as the public construction
surface.
