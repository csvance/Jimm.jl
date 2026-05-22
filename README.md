<p align="center">
    <img width="300px" src="docs/src/assets/logo.svg"/>
</p>
<div align="center">

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://csvance.github.io/Jimm.jl/dev/)
[![CI](https://img.shields.io/github/checks-status/csvance/Jimm.jl/master?label=CI)](https://github.com/csvance/Jimm.jl/commits/master)

</div>

Julia ports of [`timm`][timm] (PyTorch Image Models, by Ross Wightman) backbones
for [Lux.jl][lux], with pretrained weights loaded directly from HuggingFace Hub
in `.safetensors` format. The name is an homage to the project we port from.

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

## Basic usage

```julia
using Jimm, Lux, Random

# ResNet50 with the trained 1000-class ImageNet head.
# `create_pretrained` is family-agnostic; the symbol selects the family.
# It returns the model and a closure that loads the released weights
# into `(ps, st)` once you've run `Lux.setup`.
model, load = create_pretrained(:resnet50_a1_in1k)
ps, st = Lux.setup(Xoshiro(0), model)
ps, st = load(ps, st)

x = randn(Float32, 224, 224, 3, 1)
logits, _ = model(x, ps, st)              # (1000, 1)
top1 = argmax(vec(logits))                # ImageNet class index
```

`create_model(variant; ...)` (without weight loading) is also exported
for from-scratch training. For the full walkthrough, including
feature-extractor mode (`num_classes = 0`), single-channel inputs
(`in_chans = 1`), and the HuggingFace cache layout, see the
[Getting Started][docs-getting-started] docs page.

## License and attribution

Jimm.jl is licensed under the Apache License, Version 2.0 (see
[`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)). The license matches upstream
`timm`. The Julia code in this repository is original, but layer naming,
hyperparameters, padding ordering, and `state_dict` key layout are
deliberately taken from `timm` so pretrained weights load directly.

## Acknowledgements

Thanks to Ross Wightman for `timm`, to [HuggingFace][huggingface] for
hosting the `.safetensors` weights that Jimm.jl loads at runtime, to
the Julia ML ecosystem maintainers whose work makes a port like this
plausible, and to [Medical Metrics Inc.][medicalmetrics] for allowing
me to work on and open-source the project.

[huggingface]: https://huggingface.co/

[timm]: https://github.com/huggingface/pytorch-image-models

[lux]: https://lux.csail.mit.edu/

[docs]: https://csvance.github.io/Jimm.jl/

[docs-getting-started]: https://csvance.github.io/Jimm.jl/dev/getting_started/

[medicalmetrics]: https://medicalmetrics.com/

[license-apache2]: https://www.apache.org/licenses/LICENSE-2.0

[license-dinov3]: https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md

[license-convnextv2]: https://github.com/facebookresearch/ConvNeXt-V2/blob/main/LICENSE

[bit]: https://arxiv.org/abs/1912.11370

[convnextv1]: https://arxiv.org/abs/2201.03545

[dinov3]: https://arxiv.org/abs/2508.10104

[convnextv2]: https://arxiv.org/abs/2301.00808

[resnet]: https://arxiv.org/abs/1512.03385