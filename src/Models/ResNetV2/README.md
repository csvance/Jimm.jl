# BiT (Big Transfer) ResNetV2

Six backbone sizes (50x1, 50x3, 101x1, 101x3, 152x2, 152x4) crossed with
the pretrained tag flavors timm registers, weights pulled from the
matching `timm/` HuggingFace repos.

| variant key                                              | HF repo                                                       | num features | num classes | input size |
|----------------------------------------------------------|---------------------------------------------------------------|--------------|-------------|------------|
| `:resnetv2_50x1_bit_goog_in21k`                          | `timm/resnetv2_50x1_bit.goog_in21k`                           | 2048         | 21843       | 224        |
| `:resnetv2_50x3_bit_goog_in21k`                          | `timm/resnetv2_50x3_bit.goog_in21k`                           | 6144         | 21843       | 224        |
| `:resnetv2_101x1_bit_goog_in21k`                         | `timm/resnetv2_101x1_bit.goog_in21k`                          | 2048         | 21843       | 224        |
| `:resnetv2_101x3_bit_goog_in21k`                         | `timm/resnetv2_101x3_bit.goog_in21k`                          | 6144         | 21843       | 224        |
| `:resnetv2_152x2_bit_goog_in21k`                         | `timm/resnetv2_152x2_bit.goog_in21k`                          | 4096         | 21843       | 224        |
| `:resnetv2_152x4_bit_goog_in21k`                         | `timm/resnetv2_152x4_bit.goog_in21k`                          | 8192         | 21843       | 224        |
| `:resnetv2_50x1_bit_goog_distilled_in1k`                 | `timm/resnetv2_50x1_bit.goog_distilled_in1k`                  | 2048         | 1000        | 224        |
| `:resnetv2_50x1_bit_goog_in21k_ft_in1k`                  | `timm/resnetv2_50x1_bit.goog_in21k_ft_in1k`                   | 2048         | 1000        | 224        |
| `:resnetv2_50x3_bit_goog_in21k_ft_in1k`                  | `timm/resnetv2_50x3_bit.goog_in21k_ft_in1k`                   | 6144         | 1000        | 224        |
| `:resnetv2_101x1_bit_goog_in21k_ft_in1k`                 | `timm/resnetv2_101x1_bit.goog_in21k_ft_in1k`                  | 2048         | 1000        | 224        |
| `:resnetv2_101x3_bit_goog_in21k_ft_in1k`                 | `timm/resnetv2_101x3_bit.goog_in21k_ft_in1k`                  | 6144         | 1000        | 224        |
| `:resnetv2_152x2_bit_goog_in21k_ft_in1k`                 | `timm/resnetv2_152x2_bit.goog_in21k_ft_in1k`                  | 4096         | 1000        | 224        |
| `:resnetv2_152x4_bit_goog_in21k_ft_in1k`                 | `timm/resnetv2_152x4_bit.goog_in21k_ft_in1k`                  | 8192         | 1000        | 224        |
| `:resnetv2_152x2_bit_goog_teacher_in21k_ft_in1k`         | `timm/resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k`          | 4096         | 1000        | 224        |
| `:resnetv2_152x2_bit_goog_teacher_in21k_ft_in1k_384`     | `timm/resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k_384`      | 4096         | 1000        | 384        |

`goog_in21k` variants ship a 21843-class head pretrained on ImageNet-21k;
build with `num_classes = 21843` for logits or `num_classes = 0` for
features only. `goog_in21k_ft_in1k`, `goog_distilled_in1k`, and the
`goog_teacher_*` flavors ship a 1000-class head fine-tuned (or distilled)
on ImageNet-1k; build with `num_classes = 1000` or `num_classes = 0`. The
`_384` suffix denotes the native training resolution of the released
weights; the network itself is fully convolutional and accepts any input
size, but the listed resolution is where the weights were tuned.

## Quickstart

```julia
using Jimm, Lux, Random

# Backbone features (num_classes = 0): returns (W/32, H/32, num_features, N).
model = create_model(:resnetv2_50x1_bit_goog_in21k; num_classes = 0)
ps, st = Lux.setup(Xoshiro(0), model)
ps, st = load_pretrained(ps, st, :resnetv2_50x1_bit_goog_in21k)
x = randn(Float32, 224, 224, 3, 1)
features, _ = model(x, ps, st)            # (7, 7, 2048, 1)
```

`load_pretrained` reads `in_chans` and the classifier presence/shape
directly from `ps`, so the constructor is the single source of truth.
The per-family `bit_resnetv2` + `load_bit_resnetv2_pretrained` pair is
also exported and works identically.

## Transfer learning with a custom classifier

To fine-tune on a downstream task with a different class count, just
build the model with your target `num_classes`. `load_pretrained`
populates the backbone and emits a `@warn` letting you know the
classifier was left at its `Lux.setup` random initialization:

```julia
model = create_model(:resnetv2_50x1_bit_goog_in21k_ft_in1k; num_classes = 42)
ps, st = Lux.setup(Xoshiro(0), model)
ps, st = load_pretrained(ps, st, :resnetv2_50x1_bit_goog_in21k_ft_in1k)
# ┌ Warning: variant resnetv2_50x1_bit_goog_in21k_ft_in1k ships 1000-class
# │ pretrained weights, but the model has a 42-class head. Loading the
# │ backbone only; the classifier head is left at its Lux.setup random
# │ initialization for you to train.
```

## Advanced: manual mapping with `load_classifier`

If you have a `.safetensors` blob already in memory (e.g., from a fork
or a non-standard repo layout), [`bit_resnetv2_mapping`](@ref) accepts
the same flag the loader uses internally:

```julia
sd = Jimm.Interop.load_safetensors_state_dict(local_path)
mapping = bit_resnetv2_mapping(sd, :resnetv2_50x1_bit_goog_in21k;
                                load_classifier = true,   # include head.fc.*
                                in_chans = 3)
ps = Jimm.Interop.apply_state_dict(ps, sd, mapping)
```

Set `load_classifier = false` to skip the `head.fc.*` keys.

## License

Pretrained weights are released by Google under the **Apache License,
Version 2.0**, matching the upstream
[Big Transfer release](https://github.com/google-research/big_transfer)
and `timm`'s own license. Commercial use is permitted. Attribution
chain is preserved in the repo's [`NOTICE`](../../../NOTICE).
