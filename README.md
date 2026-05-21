# Jimm.jl

[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://csvance.github.io/Jimm.jl/)

Julia ports of [`timm`](https://github.com/huggingface/pytorch-image-models)
(PyTorch Image Models, by Ross Wightman) backbones for
[Lux.jl](https://lux.csail.mit.edu/), with pretrained weights loaded
directly from HuggingFace Hub in `.safetensors` format. The name is an
homage to the project we port from.

Jimm is a strict Lux.jl port of `timm`: same architectures, same
hyperparameters, same weight initialization, same `state_dict` key layout.
The goal is that any HuggingFace `timm/<variant>` checkpoint loads into the
corresponding Jimm model without manual rewiring, and that the forward pass
matches `timm` to within float32 round-off. **Compatibility with `timm` is
the project's #1 priority**; if the two diverge, `timm` is the reference.

## Available backbones

| family            | constructor    | num weights | weight license     |
|-------------------|----------------|-------------|--------------------|
| BiT ResNetV2      | `bit_resnetv2` | 15          | Apache 2.0         |
| ResNet            | `resnet`       | 5           | Apache 2.0         |
| ConvNeXt          | `convnext`     | 19          | Apache 2.0         |
| ConvNeXt (DINOv3) | `convnext`     | 4           | DINOv3 License     |
| ConvNeXt V2       | `convnextv2`   | 26          | CC BY-NC 4.0       |

The weight licenses above are set by the upstream releases (Google for
BiT, Facebook AI for the original ConvNeXt `.fb_*` checkpoints, Meta for
the ConvNeXt DINOv3 encoders and ConvNeXtV2) and are separate from
Jimm.jl's own Apache 2.0 code license. The plain `ConvNeXt` row holds
the 19 Facebook AI checkpoints from the original 2022 ConvNeXt paper
(T/S/B/L/XL crossed with IN1K, IN22K, and IN22K-finetuned-IN1K, at 224
and 384 resolutions), all Apache 2.0. **ConvNeXtV2 weights are CC BY-NC
4.0, which forbids commercial use**; the **ConvNeXt DINOv3 weights carry
Meta's DINOv3 License**, which imposes obligations on derived outputs
(read the
[license text](https://ai.meta.com/resources/models-and-libraries/dinov3-license/)
before deploying). Pick BiT or ConvNeXt (`.fb_*`) when commercial
deployment matters.

Variant keys are the `timm` model name with the dot rewritten as an
underscore (so the key remains a single Julia identifier). The full `timm`
name with the dot lives at `<FAMILY>_VARIANTS[key].hf_repo`.

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
HuggingFace cache layout, see the
[Getting Started](https://csvance.github.io/Jimm.jl/getting_started/)
docs page.

## Documentation

The full documentation is at
[csvance.github.io/Jimm.jl](https://csvance.github.io/Jimm.jl/):

- [Getting Started](https://csvance.github.io/Jimm.jl/getting_started/):
  end-to-end usage examples.
- [Porting Backbones](https://csvance.github.io/Jimm.jl/porting/):
  contributor guide for adding a new `timm` backbone.
- [Testing](https://csvance.github.io/Jimm.jl/testing/): parity test
  suite layout, env-var filters, and fixture dumping.
- [API Reference](https://csvance.github.io/Jimm.jl/api/): every
  exported function and type.

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
employer [Medical Metrics Inc.](https://medicalmetrics.com/).
