```@meta
CurrentModule = Jimm
```

# Getting Started

This page walks through Jimm.jl end-to-end: installing the package,
loading a pretrained ResNet50 and predicting an ImageNet class,
switching to feature-extractor mode, working with non-RGB inputs, and
the HuggingFace cache layout.

## Installation

Jimm targets Julia 1.12 or newer.

```julia
using Pkg
Pkg.add(url = "https://github.com/csvance/Jimm.jl")
```

For local hacking, clone the repo and `Pkg.develop` the path:

```julia
using Pkg
Pkg.develop(path = "/path/to/Jimm.jl")
```

The Python sidecar (`pyproject.toml` with `timm`, `torch`, `h5py`,
`safetensors`, `huggingface-hub`) is only needed to regenerate parity
fixtures for new backbones. Inference and pretrained-weight loading
work from Julia alone. See [Testing](testing.md) for the contributor
setup.

## Loading ResNet50 and predicting an ImageNet class

```julia
using Jimm, Lux, Random

# `create_model` / `load_pretrained` are family-agnostic; the symbol
# selects the family. The per-family entry points (`resnet`,
# `load_resnet_pretrained`, etc.) work identically.
model = create_model(:resnet50_a1_in1k; num_classes = 1000)
ps, st = Lux.setup(Xoshiro(0), model)
ps, st = load_pretrained(ps, st, :resnet50_a1_in1k)

x = randn(Float32, 224, 224, 3, 1)
logits, _ = model(x, ps, st)              # (1000, 1)

top5 = partialsortperm(vec(logits), 1:5; rev = true)
```

A few things worth noting in the snippet:

- Variant keys are Julia symbols. `:resnet50_a1_in1k` is the `timm`
  model name `resnet50.a1_in1k` with the dot rewritten as an
  underscore. The full name with the dot is at
  `RESNET_VARIANTS[:resnet50_a1_in1k].hf_repo`.
- `Lux.setup` produces a `ps` (parameters) and `st` (state)
  NamedTuple. `load_pretrained` (and the per-family
  `load_<family>_pretrained`) returns a new `(ps, st)` with the
  HuggingFace weights merged in. Stateless families (BiT, ConvNeXt,
  ConvNeXtV2) return `st` unchanged; ResNet merges BatchNorm running
  statistics into `st`.
- The three-step pattern (`create_model` → `Lux.setup` → `load_pretrained`)
  is the canonical one. It composes cleanly when the backbone is
  nested inside a larger model — see [Composing into a larger
  model](#composing-into-a-larger-model) below.
- The input is shaped `(W, H, C, N)`, Lux's convention. PyTorch's
  `(N, C, H, W)` is read-reversed at load time so most weights land
  in the layout Lux expects directly.
- `x` here is random noise. For a real prediction, replace it with a
  preprocessed image: resize to 224x224, scale to `Float32` in
  `[0, 1]`, then normalize with the ImageNet mean and std.

To map the top-5 indices to class names, pair the variant with any
ImageNet class label list (Jimm ships none of its own; the timm
`imagenet_classes.txt` works directly because the class index ordering
is unchanged).

## Feature extractor mode

Pass `num_classes = 0` to both the constructor and the loader to get
the post-stage feature map back instead of logits. This matches
`timm.create_model(..., num_classes=0).forward_features(x)`:

```julia
model = create_model(:resnet50_a1_in1k; num_classes = 0)
ps, st = Lux.setup(Xoshiro(0), model)
ps, st = load_pretrained(ps, st, :resnet50_a1_in1k)

x = randn(Float32, 224, 224, 3, 1)
features, _ = model(x, ps, st)            # (7, 7, 2048, 1)
```

The output is shaped `(W/32, H/32, num_features, N)` for every
registered backbone. For ResNet50 that is `(7, 7, 2048, 1)` on a
224x224 input; for `convnextv2_atto_fcmae` it is `(7, 7, 320, 1)`.

Use this mode when you want to attach Jimm's pretrained encoder to
your own downstream head (regression, segmentation, neural ODE, etc.)
without carrying around the 1000-class classifier that the
`load_*_pretrained` call would otherwise initialize.

## Single-channel and other non-RGB inputs

Pass `in_chans` to `create_model`. `load_pretrained` reads the stem
shape from `ps` and adapts the released 3-channel weight via
[`adapt_input_conv`](@ref) automatically — matching timm's
`adapt_input_conv` semantics: sum across input channels for
`in_chans = 1`, tile and rescale by `3 / in_chans` for other counts.

```julia
model = create_model(:convnextv2_atto_fcmae; in_chans = 1, num_classes = 0)
ps, st = Lux.setup(Xoshiro(0), model)
ps, st = load_pretrained(ps, st, :convnextv2_atto_fcmae)

x = randn(Float32, 224, 224, 1, 1)
features, _ = model(x, ps, st)            # (7, 7, 320, 1)
```

This is the right path for grayscale medical or scientific imagery,
where collapsing the RGB stem is preferable to repeating the
single-channel input three times.

## Composing into a larger model

`create_model` returns a bare `@compact` model with no parameters, so
you can drop it into an outer `@compact` block and let one
`Lux.setup` cover the whole composed tree. `load_pretrained` then
fills only the backbone slot via `prefix`, leaving sibling layers
(your custom head, additional towers, etc.) at their random init:

```julia
outer = @compact(
    backbone = create_model(:resnet50_a1_in1k; num_classes = 0),
    head     = Dense(2048 => num_outputs),
) do x
    head(backbone(x))
end

ps, st = Lux.setup(Xoshiro(0), outer)
ps, st = load_pretrained(ps, st, :resnet50_a1_in1k; prefix = (:backbone,))
```

The mapping function prefixes every leaf path with `prefix...`, so
`apply_state_dict` overwrites just the `ps.backbone.*` and `st.backbone.*`
subtrees. Anything you added (`:head`, sibling slots, deeper nestings)
keeps its `Lux.setup` initialization. Deeper nestings chain symbols:
`prefix = (:encoder, :backbone)`.

## Switching families

The `create_model` / `load_pretrained` dispatch above already covers
every family by symbol — no separate calls per family needed. The
per-family entry points exist for callers who already know the family
or who want to reach for family-specific kwargs explicitly:

```julia
model = <family>(variant; in_chans, num_classes)
ps, st = Lux.setup(rng, model)
ps, st = load_<family>_pretrained(ps, st, variant)
```

The constructors and loaders for each family are:

- `resnet` + [`load_resnet_pretrained`](@ref). 5 variants in
  [`RESNET_VARIANTS`](@ref).
- `bit_resnetv2` + [`load_bit_resnetv2_pretrained`](@ref). 15 variants
  in [`BIT_VARIANTS`](@ref).
- `convnext` + [`load_convnext_pretrained`](@ref). 23 variants in
  [`CONVNEXT_VARIANTS`](@ref) (19 Apache 2.0 Facebook AI checkpoints
  plus 4 Meta DINOv3 encoders).
- `convnextv2` + [`load_convnextv2_pretrained`](@ref). 26 variants in
  [`CONVNEXTV2_VARIANTS`](@ref).

See the [API Reference](api/models.md) for the full signature of each
constructor and loader.

## Pretrained weights and the HuggingFace cache

`load_<family>_pretrained` resolves `model.safetensors` against the
standard HuggingFace Hub cache layout
(`HF_HUB_CACHE` → `$HF_HOME/hub` → `~/.cache/huggingface/hub`), so the
same blob is shared with `timm` and `huggingface_hub`: whichever tool
downloads first, the other sees a cache hit. Subsequent calls
short-circuit on the cached snapshot symlink.

For gated repos, set `HUGGING_FACE_HUB_TOKEN` in the environment
before calling the loader. The token is forwarded as a bearer header
on the resolve and download requests.

For lower-level access, [`hf_hub_download`](@ref) returns the
resolved snapshot path directly and mirrors the cache semantics of
`huggingface_hub.hf_hub_download`:

```julia
path = Jimm.Interop.hf_hub_download("timm/resnet50.a1_in1k",
                                    "model.safetensors";
                                    revision = "main")
state_dict = Jimm.Interop.load_safetensors_state_dict(path)
```

This is the escape hatch for cases where you want to inspect or
transform the raw PyTorch state dict before applying it.
