```@meta
CurrentModule = Jimm
```

# Interop

PyTorch and HuggingFace plumbing. These helpers cover the bridge
between PyTorch's logical NCHW layout and Lux's WHCN layout, HDF5
parity fixtures consumed by the test suite, and the HuggingFace
Hub cache that production weight loading goes through.

## Parity fixtures and state-dict application

```@docs
read_parity
apply_state_dict
```

## Axis transforms

These are the per-key transform functions used in the
`<family>_mapping` tables when an entry's HDF5-natural layout is
not the layout Lux expects.

```@docs
axis_reverse
pyperm
as_channel4d
adapt_input_conv
```

## HuggingFace Hub

```@docs
hf_hub_download
hf_download
hf_hub_cache_dir
default_cache_dir
```

## SafeTensors

```@docs
load_safetensors_state_dict
```
