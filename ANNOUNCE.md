# [ANN] Jimm.jl: Lux ports of timm image backbones, with HuggingFace pretrained weights

I'm happy to share a first public look at **Jimm.jl**, a Julia package
that ports image-classification backbones from Ross Wightman's
[`timm`](https://github.com/huggingface/pytorch-image-models) (PyTorch
Image Models) to [Lux.jl](https://lux.csail.mit.edu/). Pretrained
weights load directly from the HuggingFace Hub in `.safetensors` format,
sharing the same on-disk cache that `timm` and `huggingface_hub` use,
so the same blob is reused whichever tool downloads first.

**Repo:** https://github.com/medicalmetrics/Jimm.jl
*(Apache 2.0; weight licenses vary per checkpoint, see below.)*

## What's in it today

A small but useful set of modern CNN backbones for research and
practitioner use, all with pretrained weights:

| family            | constructor    | variants | weight license                    |
|-------------------|----------------|----------|-----------------------------------|
| BiT ResNetV2      | `bit_resnetv2` | 15       | Apache 2.0                        |
| ConvNeXt          | `convnext`     | 19       | Apache 2.0 (Liu et al. 2022 FAIR) |
| ConvNeXt (DINOv3) | `convnext`     | 4        | Meta DINOv3 License               |
| ConvNeXt V2       | `convnextv2`   | 26       | CC BY-NC 4.0                      |

That covers the original ConvNeXt paper variants (Tiny/Small/Base/Large/XLarge
at 224 and 384, IN1K and IN22K), the four Meta DINOv3 encoders, the FCMAE
ConvNeXt V2 checkpoints, and Google's BiT ResNet-V2 with weight standardization
and GroupNorm. A typical session:

```julia
using Jimm, Lux, Random

# Backbone features only: returns (W/32, H/32, num_features, N).
model = convnext(:convnext_tiny_fb_in22k_ft_in1k;
                 in_chans = 3, num_classes = 0)
ps, st = Lux.setup(Xoshiro(0), model)
st = Lux.testmode(st)
ps = load_convnext_pretrained(ps, :convnext_tiny_fb_in22k_ft_in1k;
                              num_classes = 0)
x = randn(Float32, 224, 224, 3, 1)
features, _ = model(x, ps, st)            # (7, 7, 768, 1)
```

Pass `num_classes = 1000` instead to get the classification head, or
`in_chans = 1` for grayscale (the loader runs the same `adapt_input_conv`
adaptation `timm` does).

## What Jimm is, and what it isn't

Jimm is a **strict port** of `timm`: same architectures, same hyperparameters,
same weight initialization, same `state_dict` key layout. The goal is that
any `timm/<variant>` checkpoint on HuggingFace loads into the corresponding
Jimm model without manual rewiring, and that the forward pass matches `timm`
to within float32 round-off. When Jimm and `timm` disagree, `timm` is the
reference.

Jimm is **not** a Julia-native reimagining of image backbones, a
general computer-vision toolkit, or a training framework. It ships only
the layers and primitives `timm` itself provides, no datasets, no
augmentation pipelines, and no detection or segmentation heads beyond
what `timm` exposes on a backbone. Anything that would cause numerical
divergence from `timm` is out of scope.

To be clear up front: **we are not at 1:1 parity with the full `timm` catalog
today, and we likely won't be in the future.** `timm` has hundreds of
architectures and thousands of pretrained checkpoints; Jimm tracks the subset
its contributors actually use. Backbones land via PR.

## Parity testing as the correctness gate

The whole project rests on one quality bar: a Lux forward pass with the
released weights loaded must match `timm`'s forward pass on the same input
to within `1e-3` max-abs-diff (existing variants land well inside this:
BiT ResNetV2-50 around `1.5e-4` for features and `2e-5` for logits).

To answer the question that comes up a lot: **yes, parity tests load
real `.safetensors` weights from HuggingFace, not a dummy state dict.**
The Python sidecar that produces parity fixtures (under `test/parity/`)
builds the `timm` model with `pretrained=True`, runs `forward_features`
(and `forward` for variants with a trained head), and writes only the
input tensor and reference outputs to a small HDF5 file. The Julia parity
test reads that fixture, downloads the exact same `model.safetensors` blob
from HuggingFace via the shared HF Hub cache, applies it to the Lux model
through the family's `load_*_pretrained` loader, runs the forward pass,
and asserts max-abs-diff against the HDF5 reference. If the safetensors
loader misroutes a single weight tensor, or applies the wrong axis
permutation, the forward output diverges and the test fails. So one
test, end-to-end, covers both the architecture port and the weight
loader.

CI is still a work in progress. The full sweep downloads pretrained
weights and runs full forward passes for every registered variant, so
it is expensive, and we are still figuring out how to run it well on
the runners we have. Two environment variables (`JIMM_TEST_VARIANTS`,
`JIMM_TEST_FAMILIES`) let contributors scope a run to a single backbone
or variant on a constrained machine, and `scripts/test_variant.sh`
wraps the fixture-dump-then-test flow for the common case.

## Porting a new backbone

The mechanical work of porting is captured as a Claude Code skill in
the repository (`.claude/skills/timm-to-lux/`). If you have Claude Code
and tokens to spare, the practical path is: open the repo, ask Claude
to port `timm/<your_model>`, and follow the skill. It encodes the
seven-phase port-and-verify workflow we have used for every backbone in
the table above:

1. Capture `timm` parity fixtures via the Python sidecar.
2. Scaffold the Lux model under `src/Models/<Family>/`.
3. Implement layers with `@compact`.
4. Wire the HuggingFace `.safetensors` loader.
5. Verify forward parity end-to-end.
6. Bisect divergence with per-stage fixtures when needed.
7. Verify random-init parity (same RNG seed, same `_init_weights` recipe).

In practice the skill produces a working PR for a new backbone in one
Claude Code session for most `timm` architectures, including the
fiddly bits (cross-correlation vs convolution, GroupNorm defaults, weight
standardization, NCHW vs WHCN). It is not magic: complex architectures
still need a human in the loop, especially when divergence shows up
mid-network. But it does mean the marginal cost of adding a new backbone
is roughly "one model's worth of inference tokens" rather than "one
afternoon of careful reimplementation."

**Contributions are welcome and encouraged**, with or without the skill.
The README's "Contributing a new backbone" section walks through the
acceptance criteria (pretrained parity, random-init parity, state-dict
round-trip, variant-table entry), and there is no shortage of `timm`
families on the wish list. ViTs (DINOv2/DINOv3, SigLIP, EVA), EfficientNet,
RegNet, and the various Swin/MaxViT lineages are all open targets.

## Installation

```julia
] add https://github.com/medicalmetrics/Jimm.jl
```

(Not yet registered in `General`; intend to register once the architecture
and loader API has stabilized a bit more and we have at least one more
non-ConvNeXt family that exercises the cross-family abstractions.)

Try it on a backbone you would have otherwise reached for PyTorch for, and
let us know how it goes. Bug reports, PRs for new variants, and PRs for
entirely new backbones are all welcome.

Thanks to Ross Wightman for `timm` itself and to Avik Pal and the Lux.jl
maintainers for a Julia framework that made the port plausible in the
first place.
