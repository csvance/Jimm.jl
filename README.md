<p align="center">
    <img width="300px" src="data/jimm-jl-logo.svg"/>
</p>
<div align="center">

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://csvance.github.io/Jimm.jl/dev/)

</div>

Julia ports of [`timm`][timm] (PyTorch Image Models, by Ross Wightman) backbones
for [Lux.jl][lux], with pretrained weights loaded directly from HuggingFace Hub
in `.safetensors` format. The name is an homage to the project we port from.

## Available backbones

| Family            | Constructor    | Weights | Weight License             | Commercial Use |
|-------------------|----------------|---------|----------------------------|----------------|
| ResNet            | `resnet`       | 5       | [Apache 2.0][apache2]      | ✅              |
| BiT ResNetV2      | `bit_resnetv2` | 15      | [Apache 2.0][apache2]      | ✅              |
| ConvNeXt          | `convnext`     | 19      | [Apache 2.0][apache2]      | ✅              |
| ConvNeXt (DINOv3) | `convnext`     | 4       | [DINOv3 License][dinov3]   | ⚠️             |
| ConvNeXt V2       | `convnextv2`   | 26      | [CC BY-NC 4.0][convnextv2] | ❌              |

## Basic usage

```julia
using Jimm, Lux, Random

# ResNet50 with the trained 1000-class ImageNet head.
model = resnet(:resnet50_a1_in1k; num_classes = 1000)
ps, st = Lux.setup(Xoshiro(0), model)
ps = load_resnet_pretrained(ps, :resnet50_a1_in1k; num_classes = 1000)

x = randn(Float32, 224, 224, 3, 1)
logits, _ = model(x, ps, st)              # (1000, 1)
top1 = argmax(vec(logits))                # ImageNet class index
```

For the full walkthrough, including feature-extractor mode
(`num_classes = 0`), single-channel inputs (`in_chans = 1`), and the
HuggingFace cache layout, see the [Getting Started][docs-getting-started] docs page.

## Documentation

The full documentation is at [csvance.github.io/Jimm.jl][docs]:

- [Getting Started][docs-getting-started]: end-to-end usage examples.
- [Porting Backbones][docs-porting]: contributor guide for adding a new `timm` backbone.
- [Testing][docs-testing]: parity test suite layout, env-var filters, and fixture dumping.
- [API Reference][docs-api]: every exported function and type.

## License and attribution

Jimm.jl is licensed under the Apache License, Version 2.0 (see
[`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)). The license matches upstream
`timm`. The Julia code in this repository is original, but layer naming,
hyperparameters, padding ordering, and `state_dict` key layout are
deliberately taken from `timm` so pretrained weights load directly. Where
`timm` itself credits an earlier upstream (e.g. Google's Big Transfer
release for BiT ResNet), that attribution chain is preserved in
[`NOTICE`](NOTICE).

## Acknowledgements

Thanks to Ross Wightman for `timm`, to the Julia ML ecosystem
maintainers whose work makes a port like this plausible, and to my
employer [Medical Metrics Inc.][medicalmetrics].

[timm]: https://github.com/huggingface/pytorch-image-models
[lux]: https://lux.csail.mit.edu/
[docs]: https://csvance.github.io/Jimm.jl/
[docs-getting-started]: https://csvance.github.io/Jimm.jl/getting_started/
[docs-porting]: https://csvance.github.io/Jimm.jl/porting/
[docs-testing]: https://csvance.github.io/Jimm.jl/testing/
[docs-api]: https://csvance.github.io/Jimm.jl/api/
[medicalmetrics]: https://medicalmetrics.com/
[apache2]: https://www.apache.org/licenses/LICENSE-2.0
[dinov3]: https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md
[convnextv2]: https://github.com/facebookresearch/ConvNeXt-V2/blob/main/LICENSE