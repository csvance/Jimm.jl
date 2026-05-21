```@meta
CurrentModule = Jimm
```

# Jimm.jl

Julia ports of [`timm`](https://github.com/huggingface/pytorch-image-models)
(PyTorch Image Models, by Ross Wightman) backbones for
[Lux.jl](https://lux.csail.mit.edu/), with pretrained weights loaded
directly from HuggingFace Hub in `.safetensors` format. The name is an
homage to the project we port from.

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

| Family                      | Constructor    | Weights | Weight License                     | Commercial Use |
|-----------------------------|----------------|---------|------------------------------------|----------------|
| [ResNet][resnet]            | `resnet`       | 5       | [Apache 2.0][license-apache2]      | ✅              |
| [BiT ResNetV2][bit]         | `bit_resnetv2` | 15      | [Apache 2.0][license-apache2]      | ✅              |
| [ConvNeXt][convnextv1]      | `convnext`     | 19      | [Apache 2.0][license-apache2]      | ✅              |
| [ConvNeXt (DINOv3)][dinov3] | `convnext`     | 4       | [DINOv3 License][license-dinov3]   | ⚠️             |
| [ConvNeXt V2][convnextv2]   | `convnextv2`   | 26      | [CC BY-NC 4.0][license-convnextv2] | ❌              |

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


[timm]: https://github.com/huggingface/pytorch-image-models

[lux]: https://lux.csail.mit.edu/

[docs]: https://csvance.github.io/Jimm.jl/

[docs-getting-started]: https://csvance.github.io/Jimm.jl/getting_started/

[docs-porting]: https://csvance.github.io/Jimm.jl/porting/

[docs-testing]: https://csvance.github.io/Jimm.jl/testing/

[docs-api]: https://csvance.github.io/Jimm.jl/api/

[medicalmetrics]: https://medicalmetrics.com/

[license-apache2]: https://www.apache.org/licenses/LICENSE-2.0

[license-dinov3]: https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md

[license-convnextv2]: https://github.com/facebookresearch/ConvNeXt-V2/blob/main/LICENSE

[bit]: https://arxiv.org/abs/1912.11370

[convnextv1]: https://arxiv.org/abs/2201.03545

[dinov3]: https://arxiv.org/abs/2508.10104

[convnextv2]: https://arxiv.org/abs/2301.00808

[resnet]: https://arxiv.org/abs/1512.03385