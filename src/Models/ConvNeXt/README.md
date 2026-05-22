# ConvNeXt (v1)

Lux port of timm's `convnext.py` (the original 2022 ConvNeXt). Covers
two release lineages: Meta's four DINOv3 pretrained encoders (no
classifier head, DINOv3 License) and the 19 Facebook AI checkpoints
from the original Liu et al. paper (T/S/B/L/XL crossed with `.fb_in1k`,
`.fb_in22k`, `.fb_in22k_ft_in1k`, `.fb_in22k_ft_in1k_384`; XLarge has
no `.fb_in1k`). Other timm `convnext_*` lineages (`.in12k_*`, `.clip_*`)
work with the same constructor and mapping code, only the variant table
needs entries.

## DINOv3 encoders (Meta, 2025)

| variant key                              | HF repo                                      | num features | num classes | input size |
|------------------------------------------|----------------------------------------------|--------------|-------------|------------|
| `:convnext_tiny_dinov3_lvd1689m`         | `timm/convnext_tiny.dinov3_lvd1689m`         | 768          | 0           | 224        |
| `:convnext_small_dinov3_lvd1689m`        | `timm/convnext_small.dinov3_lvd1689m`        | 768          | 0           | 224        |
| `:convnext_base_dinov3_lvd1689m`         | `timm/convnext_base.dinov3_lvd1689m`         | 1024         | 0           | 224        |
| `:convnext_large_dinov3_lvd1689m`        | `timm/convnext_large.dinov3_lvd1689m`        | 1536         | 0           | 224        |

All four DINOv3 variants ship `num_classes = 0` (bare encoder, no usable
classification head). Build with `num_classes = 0` to get features. The
network is fully convolutional and accepts any input size; 224 is the
training resolution for these checkpoints.

## Facebook AI checkpoints from the original ConvNeXt paper

| variant key                                    | HF repo                                            | num features | num classes | input size |
|------------------------------------------------|----------------------------------------------------|--------------|-------------|------------|
| `:convnext_tiny_fb_in1k`                       | `timm/convnext_tiny.fb_in1k`                       | 768          | 1000        | 224        |
| `:convnext_tiny_fb_in22k`                      | `timm/convnext_tiny.fb_in22k`                      | 768          | 21841       | 224        |
| `:convnext_tiny_fb_in22k_ft_in1k`              | `timm/convnext_tiny.fb_in22k_ft_in1k`              | 768          | 1000        | 224        |
| `:convnext_tiny_fb_in22k_ft_in1k_384`          | `timm/convnext_tiny.fb_in22k_ft_in1k_384`          | 768          | 1000        | 384        |
| `:convnext_small_fb_in1k`                      | `timm/convnext_small.fb_in1k`                      | 768          | 1000        | 224        |
| `:convnext_small_fb_in22k`                     | `timm/convnext_small.fb_in22k`                     | 768          | 21841       | 224        |
| `:convnext_small_fb_in22k_ft_in1k`             | `timm/convnext_small.fb_in22k_ft_in1k`             | 768          | 1000        | 224        |
| `:convnext_small_fb_in22k_ft_in1k_384`         | `timm/convnext_small.fb_in22k_ft_in1k_384`         | 768          | 1000        | 384        |
| `:convnext_base_fb_in1k`                       | `timm/convnext_base.fb_in1k`                       | 1024         | 1000        | 224        |
| `:convnext_base_fb_in22k`                      | `timm/convnext_base.fb_in22k`                      | 1024         | 21841       | 224        |
| `:convnext_base_fb_in22k_ft_in1k`              | `timm/convnext_base.fb_in22k_ft_in1k`              | 1024         | 1000        | 224        |
| `:convnext_base_fb_in22k_ft_in1k_384`          | `timm/convnext_base.fb_in22k_ft_in1k_384`          | 1024         | 1000        | 384        |
| `:convnext_large_fb_in1k`                      | `timm/convnext_large.fb_in1k`                      | 1536         | 1000        | 224        |
| `:convnext_large_fb_in22k`                     | `timm/convnext_large.fb_in22k`                     | 1536         | 21841       | 224        |
| `:convnext_large_fb_in22k_ft_in1k`             | `timm/convnext_large.fb_in22k_ft_in1k`             | 1536         | 1000        | 224        |
| `:convnext_large_fb_in22k_ft_in1k_384`         | `timm/convnext_large.fb_in22k_ft_in1k_384`         | 1536         | 1000        | 384        |
| `:convnext_xlarge_fb_in22k`                    | `timm/convnext_xlarge.fb_in22k`                    | 2048         | 21841       | 224        |
| `:convnext_xlarge_fb_in22k_ft_in1k`            | `timm/convnext_xlarge.fb_in22k_ft_in1k`            | 2048         | 1000        | 224        |
| `:convnext_xlarge_fb_in22k_ft_in1k_384`        | `timm/convnext_xlarge.fb_in22k_ft_in1k_384`        | 2048         | 1000        | 384        |

The `.fb_*` variants all ship a trained classifier head. Build with
`num_classes = 1000` for the IN1K and IN22K-finetuned-IN1K weights, or
`num_classes = 21841` for the raw IN22K weights, and the loader will
populate `head.norm.*` and `head.fc.*` from the safetensors blob. Pass
`num_classes = 0` to drop the head and use the network as a feature
encoder regardless of which checkpoint you load.

## Architectural notes

ConvNeXt v1 and v2 share the same stem, downsample, classifier head, and
stage scaffolding; those pieces live in `../ConvNeXtCommon/Common.jl` and
are reused verbatim. The two families differ only in the block body:

| | v1 (`convnext`) | v2 (`convnextv2`) |
|---|---|---|
| MLP path | Linear → GELU → Linear | 1x1 Conv → GELU → GRN → 1x1 Conv |
| Per-channel scaling | LayerScale `gamma` (init `1e-6`) | none |
| Storage | `nn.Linear` 2D weights | `nn.Conv2d` 4D 1x1 weights |

The Lux implementation builds the v1 block with 1x1 `Conv` layers (the v2
path), reshaping the released 2D Linear weights to 4D at load time. The
two paths are mathematically identical (a 1x1 conv on `(N, C, H, W)` is a
Linear on the channel axis at each spatial location), so this lets v1 and
v2 share the rest of the scaffolding without an explicit permute in the
forward pass.

## Quickstart

```julia
using Jimm, Lux, Random

# Feature extraction with a DINOv3 encoder: returns (W/32, H/32, num_features, N).
model, load = create_pretrained(:convnext_tiny_dinov3_lvd1689m; num_classes = 0)
ps, st = Lux.setup(Xoshiro(0), model)
st = Lux.testmode(st)
ps, st = load(ps, st)
x = randn(Float32, 224, 224, 3, 1)
features, _ = model(x, ps, st)            # (7, 7, 768, 1)

# Classification with an FB-paper checkpoint: returns (num_classes, N).
model, load = create_pretrained(:convnext_tiny_fb_in22k_ft_in1k)
ps, st = Lux.setup(Xoshiro(0), model)
st = Lux.testmode(st)
ps, st = load(ps, st)
logits, _ = model(x, ps, st)              # (1000, 1)
```

`create_pretrained` captures `in_chans` and `num_classes` in the
returned closure, so they're specified once and the loader doesn't
need to introspect `ps`. Use `create_model(variant; ...)` for a
random-init build without weights.

## Transfer learning with a custom classifier

ConvNeXt's classification head is `head_norm` (LayerNorm whose dim
depends only on the feature width) plus `head_fc` (Dense whose dim
depends on `num_classes`). When you build with a non-matching
`num_classes`, the loader still loads the backbone *and* `head_norm`
from the pretrained checkpoint, and emits a `@warn` letting you know
`head_fc` was left at its `Lux.setup` random init:

```julia
model, load = create_pretrained(:convnext_tiny_fb_in22k_ft_in1k; num_classes = 42)
ps, st = Lux.setup(Xoshiro(0), model)
ps, st = load(ps, st)
# ┌ Warning: variant convnext_tiny_fb_in22k_ft_in1k ships 1000-class
# │ pretrained weights, but the model has a 42-class head. Loading the
# │ backbone (and head_norm) only; the classifier is left at its
# │ Lux.setup random initialization for you to train.
```

The DINOv3 variants have `default_num_classes = 0` and ship no
classifier weights, so any model built with `num_classes > 0` on top
of a DINOv3 encoder will get the same warning and a randomly-initialized
`head_fc`.

## License

The `.fb_*` Facebook AI weights are released under the Apache 2.0
license, matching timm's own license and the Julia code in this package.
The DINOv3 weights are not.

> [!WARNING]
> The DINOv3 pretrained weights are released by Meta under the
> **DINOv3 License**
> ([dinov3-license](https://ai.meta.com/resources/models-and-libraries/dinov3-license/)).
> Read the license before using the weights for any downstream task; the
> license imposes obligations on outputs derived from the weights that
> differ from a standard permissive open-source license.
>
> This license restriction applies only to the *weights*. The Julia code
> in this package is Apache 2.0.
