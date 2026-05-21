```@meta
CurrentModule = Jimm
```

# Jimm.jl

Julia ports of [`timm`](https://github.com/huggingface/pytorch-image-models)
(PyTorch Image Models, by Ross Wightman) backbones for
[Lux.jl](https://lux.csail.mit.edu/), with pretrained weights loaded
directly from HuggingFace Hub in `.safetensors` format. The name is an
homage to the project we port from.

Jimm is a strict Lux.jl port of `timm`: same architectures, same
hyperparameters, same weight initialization, same `state_dict` key
layout. The goal is that any HuggingFace `timm/<variant>` checkpoint
loads into the corresponding Jimm model without manual rewiring, and
that the forward pass matches `timm` to within float32 round-off.
**Compatibility with `timm` is the project's #1 priority**; if the two
diverge, `timm` is the reference.

## Status

Most of Jimm was written by AI agents driving the porting workflow
encoded in `.claude/skills/timm-to-lux/`, with human review at each
phase and the parity tests as the correctness backstop. The code is
already being used in real projects, so the registered backbones work
for forward inference with the released weights. That said: **expect
bugs and rough edges**, especially around anything the parity tests do
not exercise (custom training loops, mixed-precision paths, exotic
input shapes). File issues and PRs.

## Available backbones

| Family            | Constructor    | Weights | Weight license   |
|-------------------|----------------|---------|------------------|
| BiT ResNetV2      | `bit_resnetv2` | 15      | Apache 2.0       |
| ResNet            | `resnet`       | 5       | Apache 2.0       |
| ConvNeXt          | `convnext`     | 19      | Apache 2.0       |
| ConvNeXt (DINOv3) | `convnext`     | 4       | DINOv3 License   |
| ConvNeXt V2       | `convnextv2`   | 26      | CC BY-NC 4.0     |

Weight licenses are set by the upstream releases (Google for BiT,
Facebook AI for the original ConvNeXt `.fb_*` checkpoints, Meta for
the ConvNeXt DINOv3 encoders and ConvNeXtV2) and are separate from
Jimm.jl's own Apache 2.0 code license. **ConvNeXtV2 weights are CC
BY-NC 4.0, which forbids commercial use**; the **ConvNeXt DINOv3
weights carry Meta's DINOv3 License**, which imposes obligations on
derived outputs (read the [license text](https://ai.meta.com/resources/models-and-libraries/dinov3-license/)
before deploying). Pick BiT or ConvNeXt (`.fb_*`) when commercial
deployment matters.

Variant keys are the `timm` model name with the dot rewritten as an
underscore (so the key remains a single Julia identifier). The full
`timm` name with the dot lives at `<FAMILY>_VARIANTS[key].hf_repo`.

## At a glance

```julia
using Jimm, Lux, Random

# ResNet50 with the trained 1000-class ImageNet head.
model = resnet(:resnet50_a1_in1k; num_classes = 1000)
ps, st = Lux.setup(Xoshiro(0), model)
ps = load_resnet_pretrained(ps, :resnet50_a1_in1k; num_classes = 1000)

x = randn(Float32, 224, 224, 3, 1)
logits, _ = model(x, ps, st)          # (1000, 1)
top1 = argmax(vec(logits))            # ImageNet class index
```

See [Getting Started](getting_started.md) for the full walkthrough,
including feature-extractor mode (`num_classes = 0`) and single-channel
inputs.

## Where to go next

- [Getting Started](getting_started.md): end-to-end prediction
  example, switching families, grayscale inputs, HuggingFace cache.
- [Porting Backbones](porting.md): contributor guide for adding a new
  `timm` backbone, with the parity-driven workflow.
- [Testing](testing.md): how the parity test suite is structured, the
  env-var filters, and how to dump a fixture for a new variant.
- [API Reference](api/index.md): every exported function and type.
