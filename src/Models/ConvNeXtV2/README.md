# ConvNeXtV2

FCMAE (masked autoencoder) pretrained encoders across the full timm size
ladder, with optional ImageNet-1K (and, for the larger sizes, ImageNet-22k
→ ImageNet-1K) fine-tuned classification heads.

| variant key                                       | HF repo                                            | num features | num classes | input size |
|---------------------------------------------------|----------------------------------------------------|--------------|-------------|------------|
| `:convnextv2_atto_fcmae`                          | `timm/convnextv2_atto.fcmae`                       | 320          | 0           | 224        |
| `:convnextv2_atto_fcmae_ft_in1k`                  | `timm/convnextv2_atto.fcmae_ft_in1k`               | 320          | 1000        | 224        |
| `:convnextv2_femto_fcmae`                         | `timm/convnextv2_femto.fcmae`                      | 384          | 0           | 224        |
| `:convnextv2_femto_fcmae_ft_in1k`                 | `timm/convnextv2_femto.fcmae_ft_in1k`              | 384          | 1000        | 224        |
| `:convnextv2_pico_fcmae`                          | `timm/convnextv2_pico.fcmae`                       | 512          | 0           | 224        |
| `:convnextv2_pico_fcmae_ft_in1k`                  | `timm/convnextv2_pico.fcmae_ft_in1k`               | 512          | 1000        | 224        |
| `:convnextv2_nano_fcmae`                          | `timm/convnextv2_nano.fcmae`                       | 640          | 0           | 224        |
| `:convnextv2_nano_fcmae_ft_in1k`                  | `timm/convnextv2_nano.fcmae_ft_in1k`               | 640          | 1000        | 224        |
| `:convnextv2_nano_fcmae_ft_in22k_in1k`            | `timm/convnextv2_nano.fcmae_ft_in22k_in1k`         | 640          | 1000        | 224        |
| `:convnextv2_nano_fcmae_ft_in22k_in1k_384`        | `timm/convnextv2_nano.fcmae_ft_in22k_in1k_384`     | 640          | 1000        | 384        |
| `:convnextv2_tiny_fcmae`                          | `timm/convnextv2_tiny.fcmae`                       | 768          | 0           | 224        |
| `:convnextv2_tiny_fcmae_ft_in1k`                  | `timm/convnextv2_tiny.fcmae_ft_in1k`               | 768          | 1000        | 224        |
| `:convnextv2_tiny_fcmae_ft_in22k_in1k`            | `timm/convnextv2_tiny.fcmae_ft_in22k_in1k`         | 768          | 1000        | 224        |
| `:convnextv2_tiny_fcmae_ft_in22k_in1k_384`        | `timm/convnextv2_tiny.fcmae_ft_in22k_in1k_384`     | 768          | 1000        | 384        |
| `:convnextv2_base_fcmae`                          | `timm/convnextv2_base.fcmae`                       | 1024         | 0           | 224        |
| `:convnextv2_base_fcmae_ft_in1k`                  | `timm/convnextv2_base.fcmae_ft_in1k`               | 1024         | 1000        | 224        |
| `:convnextv2_base_fcmae_ft_in22k_in1k`            | `timm/convnextv2_base.fcmae_ft_in22k_in1k`         | 1024         | 1000        | 224        |
| `:convnextv2_base_fcmae_ft_in22k_in1k_384`        | `timm/convnextv2_base.fcmae_ft_in22k_in1k_384`     | 1024         | 1000        | 384        |
| `:convnextv2_large_fcmae`                         | `timm/convnextv2_large.fcmae`                      | 1536         | 0           | 224        |
| `:convnextv2_large_fcmae_ft_in1k`                 | `timm/convnextv2_large.fcmae_ft_in1k`              | 1536         | 1000        | 224        |
| `:convnextv2_large_fcmae_ft_in22k_in1k`           | `timm/convnextv2_large.fcmae_ft_in22k_in1k`        | 1536         | 1000        | 224        |
| `:convnextv2_large_fcmae_ft_in22k_in1k_384`       | `timm/convnextv2_large.fcmae_ft_in22k_in1k_384`    | 1536         | 1000        | 384        |
| `:convnextv2_huge_fcmae`                          | `timm/convnextv2_huge.fcmae`                       | 2816         | 0           | 224        |
| `:convnextv2_huge_fcmae_ft_in1k`                  | `timm/convnextv2_huge.fcmae_ft_in1k`               | 2816         | 1000        | 224        |
| `:convnextv2_huge_fcmae_ft_in22k_in1k_384`        | `timm/convnextv2_huge.fcmae_ft_in22k_in1k_384`     | 2816         | 1000        | 384        |
| `:convnextv2_huge_fcmae_ft_in22k_in1k_512`        | `timm/convnextv2_huge.fcmae_ft_in22k_in1k_512`     | 2816         | 1000        | 512        |

`fcmae` variants are bare encoders: the released checkpoint has no
classification head, so build them with `num_classes = 0`. The
`fcmae_ft_in1k` and `fcmae_ft_in22k_in1k` variants ship a 1000-class head;
build with `num_classes = 1000` for logits, or `num_classes = 0` for
features only. The `_384` and `_512` suffixes denote the native training
resolution of the released weights; the network itself is fully
convolutional and accepts any input size, but the listed resolution is
where the weights were tuned. `convnextv2_small` is not included because
timm only registers it as `.untrained` (no pretrained weights).

## Quickstart

```julia
using Jimm, Lux, Random

# Classifier head: returns (num_classes, N).
model, load = create_pretrained(:convnextv2_atto_fcmae_ft_in1k)
ps, st = Lux.setup(Xoshiro(0), model)
ps, st = load(ps, st)
x = randn(Float32, 224, 224, 3, 1)
logits, _ = model(x, ps, st)              # (1000, 1)
```

`create_pretrained` captures `in_chans` and `num_classes` in the
returned closure, so they're specified once and the loader doesn't
need to introspect `ps`. Use `create_model(variant; ...)` for a
random-init build without weights.

## Transfer learning with a custom classifier

ConvNeXtV2's classification head is `head_norm` (LayerNorm whose dim
depends only on the feature width) plus `head_fc` (Dense whose dim
depends on `num_classes`). When you build with a non-matching
`num_classes`, the loader still loads the backbone *and* `head_norm`
from the pretrained checkpoint, and emits a `@warn` letting you know
`head_fc` was left at its `Lux.setup` random init:

```julia
model, load = create_pretrained(:convnextv2_atto_fcmae_ft_in1k; num_classes = 42)
ps, st = Lux.setup(Xoshiro(0), model)
ps, st = load(ps, st)
# ┌ Warning: variant convnextv2_atto_fcmae_ft_in1k ships 1000-class
# │ pretrained weights, but the model has a 42-class head. Loading the
# │ backbone (and head_norm) only; the classifier is left at its
# │ Lux.setup random initialization for you to train.
```

The `fcmae` (non-`ft_in1k`) variants have `default_num_classes = 0` and
ship no classifier weights, so any model built with `num_classes > 0`
on top of an `fcmae` encoder will get the same warning and a
randomly-initialized `head_fc`.

## License

> [!WARNING]
> The pretrained weights are released by Meta under the
> **Creative Commons Attribution-NonCommercial 4.0** license
> ([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)).
> **Commercial use of these weights is not permitted.** This applies
> to every row in the table above and is independent of Jimm.jl's
> own Apache 2.0 code license.

See the upstream
[ConvNeXt-V2 release](https://github.com/facebookresearch/ConvNeXt-V2)
for the full text. If commercial use matters, the BiT family (Apache 2.0)
or other future families with permissive weights are the alternatives.
