```@meta
CurrentModule = Jimm
```

# Interop

PyTorch and HuggingFace plumbing: applying a PyTorch `state_dict` to a
Lux `(ps, st)` pair, resolving and caching weights through the
HuggingFace Hub, and loading `.safetensors` blobs.

## State-dict application

```@docs
apply_state_dict
```

## HuggingFace Hub

```@docs
hf_hub_download
hf_download
hf_hub_cache_dir
```

## SafeTensors

```@docs
load_safetensors_state_dict
```