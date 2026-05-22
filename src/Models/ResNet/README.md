# ResNet

Classic ResNet ports matching timm.

| Variant key | timm / HuggingFace repo | Block | Features | Classes | Input |
|-------------|--------------------------|-------|----------|---------|-------|
| `:resnet18_a1_in1k` | `timm/resnet18.a1_in1k` | BasicBlock | 512 | 1000 | 224 |
| `:resnet34_a1_in1k` | `timm/resnet34.a1_in1k` | BasicBlock | 512 | 1000 | 224 |
| `:resnet50_a1_in1k` | `timm/resnet50.a1_in1k` | Bottleneck | 2048 | 1000 | 224 |
| `:resnet101_a1_in1k` | `timm/resnet101.a1_in1k` | Bottleneck | 2048 | 1000 | 224 |
| `:resnet152_a1_in1k` | `timm/resnet152.a1_in1k` | Bottleneck | 2048 | 1000 | 224 |

## Quickstart

```julia
using Jimm, Lux, Random

# Backbone features (num_classes = 0): returns (W/32, H/32, num_features, N).
model = create_model(:resnet18_a1_in1k; in_chans = 3, num_classes = 0)
ps, st = Lux.setup(Xoshiro(0), model)
ps, st = load_pretrained(ps, st, :resnet18_a1_in1k)
st = Lux.testmode(st)

x = randn(Float32, 224, 224, 3, 1)
features, _ = model(x, ps, st)  # (7, 7, 512, 1)
```

`load_pretrained` reads `in_chans` and the classifier presence/shape
directly from `ps`, so the constructor is the single source of truth.
The per-family `resnet` + `load_resnet_pretrained` pair is also exported
and works identically.

## Transfer learning with a custom classifier

To fine-tune on a downstream task with a different class count, just
build the model with your target `num_classes`. `load_pretrained`
populates the backbone (including the BatchNorm running statistics)
and emits a `@warn` letting you know the classifier was left at its
`Lux.setup` random initialization for you to train:

```julia
model = create_model(:resnet50_a1_in1k; num_classes = 42)
ps, st = Lux.setup(Xoshiro(0), model)
ps, st = load_pretrained(ps, st, :resnet50_a1_in1k)
# ┌ Warning: variant resnet50_a1_in1k ships 1000-class pretrained weights,
# │ but the model has a 42-class head. Loading the backbone only;
# │ the classifier head is left at its Lux.setup random initialization for
# │ you to train.
```

## Advanced: manual mapping with `load_classifier`

If you have a `.safetensors` blob already in memory (e.g., from a
fork or a non-standard repo layout) and want to apply it without the
HF download path, [`resnet_mapping`](@ref) and
[`resnet_state_mapping`](@ref) accept the same routing flags the loader
uses internally:

```julia
sd = Jimm.Interop.load_safetensors_state_dict(local_path)
param_mapping = resnet_mapping(sd, :resnet18_a1_in1k;
                                load_classifier = true,   # include fc.*
                                in_chans = 3)
state_mapping = resnet_state_mapping(sd, :resnet18_a1_in1k)
ps = Jimm.Interop.apply_state_dict(ps, sd, param_mapping)
st = Jimm.Models.apply_resnet_state_dict(st, sd, state_mapping)
```

Set `load_classifier = false` to skip the `fc.*` keys (use when your
model has `num_classes = 0` or a custom head dim).
