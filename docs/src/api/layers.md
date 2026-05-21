```@meta
CurrentModule = Jimm
```

# Layers

Reusable building blocks shared across model families. These exist
to match specific `timm` constructs in semantics, layout, and
default parameters. They are not intended as a general-purpose
layer library.

## Building blocks

```@docs
std_conv
layernorm2d
grn_layer
```

## Initializers

```@docs
kaiming_normal_fan_out
normal_init
```
