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
model, load = create_pretrained(:resnet18_a1_in1k; in_chans = 3, num_classes = 0)
ps, st = Lux.setup(Xoshiro(0), model)
ps, st = load(ps, st)
st = Lux.testmode(st)

x = randn(Float32, 224, 224, 3, 1)
features, _ = model(x, ps, st)  # (7, 7, 512, 1)
```

`create_pretrained` captures `in_chans` and `num_classes` in the
returned closure, so they're specified once and the loader doesn't
need to introspect `ps`. Use `create_model(variant; ...)` for a
random-init build without weights.

## Transfer learning with a custom classifier

To fine-tune on a downstream task with a different class count, just
build the model with your target `num_classes`. The closure populates
the backbone (including the BatchNorm running statistics) and emits a
`@warn` letting you know the classifier was left at its `Lux.setup`
random initialization for you to train:

```julia
model, load = create_pretrained(:resnet50_a1_in1k; num_classes = 42)
ps, st = Lux.setup(Xoshiro(0), model)
ps, st = load(ps, st)
# ┌ Warning: variant resnet50_a1_in1k ships 1000-class pretrained weights,
# │ but the model has a 42-class head. Loading the backbone only;
# │ the classifier head is left at its Lux.setup random initialization for
# │ you to train.
```
