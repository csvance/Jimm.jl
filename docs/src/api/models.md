```@meta
CurrentModule = Jimm
```

# Models

## Family-agnostic interface

`create_pretrained` is the symbol-dispatched entry point for loading
released weights. It returns the model and a closure that loads the
HuggingFace checkpoint into a `(ps, st)` pair you produce with
`Lux.setup`:

```julia
model, load = create_pretrained(variant)
ps, st = Lux.setup(rng, model)
ps, st = load(ps, st)
```

The closure captures `variant`, `in_chans`, `num_classes`, and the HF
/ `prefix` kwargs at construction time, so the loader body no longer
needs to introspect `ps` to recover what you already told it.
`create_model` is the random-init counterpart: it returns the bare
`@compact` model with no weights loaded. See
[Getting Started](../getting_started.md#composing-into-a-larger-model)
for the nested pattern with `prefix`.

```@docs
create_pretrained
create_model
default_num_classes
```

## Per-family namespaces

Each family exports its variant config struct and the
`<FAMILY>_VARIANTS` registry dict. The remaining family internals
(per-family constructors, weight mappings, state mappings) live in
`Jimm.Models.*` for callers who need to escape the
`create_pretrained` / `create_model` front door.

## ResNet

```@docs
ResNetVariant
RESNET_VARIANTS
```

### Registered variants

```@eval
using Markdown, Jimm
rows = sort(collect(Jimm.RESNET_VARIANTS); by = p -> String(first(p)))
io = IOBuffer()
println(io, "| Variant | HF repo | num_classes | num_features | input size |")
println(io, "|:---|:---|---:|---:|---:|")
for (k, v) in rows
    println(io, "| `:$(k)` | [`$(v.hf_repo)`](https://huggingface.co/$(v.hf_repo)) | $(v.default_num_classes) | $(v.num_features) | $(v.default_input_size) |")
end
Markdown.parse(String(take!(io)))
```

## BiT ResNetV2

```@docs
BiTVariant
BIT_VARIANTS
```

### Registered variants

```@eval
using Markdown, Jimm
rows = sort(collect(Jimm.BIT_VARIANTS); by = p -> String(first(p)))
io = IOBuffer()
println(io, "| Variant | HF repo | num_classes | num_features | input size |")
println(io, "|:---|:---|---:|---:|---:|")
for (k, v) in rows
    println(io, "| `:$(k)` | [`$(v.hf_repo)`](https://huggingface.co/$(v.hf_repo)) | $(v.default_num_classes) | $(v.num_features) | $(v.default_input_size) |")
end
Markdown.parse(String(take!(io)))
```

## ConvNeXt

```@docs
ConvNeXtVariant
CONVNEXT_VARIANTS
```

### Registered variants

```@eval
using Markdown, Jimm
rows = sort(collect(Jimm.CONVNEXT_VARIANTS); by = p -> String(first(p)))
io = IOBuffer()
println(io, "| Variant | HF repo | num_classes | num_features | input size |")
println(io, "|:---|:---|---:|---:|---:|")
for (k, v) in rows
    println(io, "| `:$(k)` | [`$(v.hf_repo)`](https://huggingface.co/$(v.hf_repo)) | $(v.default_num_classes) | $(v.dims[end]) | $(v.default_input_size) |")
end
Markdown.parse(String(take!(io)))
```

## ConvNeXt V2

```@docs
ConvNeXtV2Variant
CONVNEXTV2_VARIANTS
```

### Registered variants

```@eval
using Markdown, Jimm
rows = sort(collect(Jimm.CONVNEXTV2_VARIANTS); by = p -> String(first(p)))
io = IOBuffer()
println(io, "| Variant | HF repo | num_classes | num_features | input size |")
println(io, "|:---|:---|---:|---:|---:|")
for (k, v) in rows
    println(io, "| `:$(k)` | [`$(v.hf_repo)`](https://huggingface.co/$(v.hf_repo)) | $(v.default_num_classes) | $(v.dims[end]) | $(v.default_input_size) |")
end
Markdown.parse(String(take!(io)))
```
