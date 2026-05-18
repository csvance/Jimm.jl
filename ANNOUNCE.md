# [ANN] Jimm.jl: Lux ports of timm image backbones, with HuggingFace pretrained weights

I'm happy to share a first public look at **Jimm.jl**, a Julia package
that ports image-classification backbones from Ross Wightman's
[`timm`](https://github.com/huggingface/pytorch-image-models) (PyTorch
Image Models) to [Lux.jl](https://lux.csail.mit.edu/). Pretrained weights
load directly from the HuggingFace Hub in `.safetensors` format, sharing
the same on-disk cache `timm` and `huggingface_hub` use.

**Repo:** https://github.com/csvance/Jimm.jl
*(see the README for install, quickstart, the full variant table, and
the porting workflow; this post is just the elevator pitch.)*

## Why this exists

The motivation was concrete: I needed Julia's SciML ecosystem together
with modern vision backbones, and Python doesn't have a peer for SciML.
The original stack was vision encoders feeding `torchdiffeq` in
PyTorch, which works but leaves a lot of the differential-equation,
sensitivity-analysis, and probabilistic-programming tooling that Julia
is genuinely best in class at on the table. Moving the diff-eq side to
Julia meant the vision side had to come too. Jimm started as a one-off
port of a single backbone for that internal use case and snowballed
from there. If your work also lives at that intersection of pretrained
vision encoders and the rest of the SciML stack, the hope is that Jimm
makes Julia a complete option for that workload.

## What you get today

A small but useful set of modern CNN backbones for research and
practitioner use, with pretrained weights: **BiT ResNetV2** (15 variants,
Apache 2.0), **ConvNeXt v1** (19 Facebook AI variants from the original
2022 paper, Apache 2.0, plus 4 DINOv3 encoders under Meta's DINOv3
License), and **ConvNeXt V2** (26 FCMAE variants, CC BY-NC 4.0). 64
checkpoints in total. Pretrained weights load by passing the variant key
to a one-line `load_<family>_pretrained` call. ViT, EfficientNet, Swin,
and the rest of the `timm` catalog are open targets for contribution.

## What Jimm is, and isn't

It is a strict port: same architectures, same hyperparameters, same
weight init, same `state_dict` key layout, so any `timm/<variant>`
checkpoint on HuggingFace loads without manual rewiring and the forward
pass matches `timm` to within float32 round-off. It is **not** a
Julia-native reimagining, a general CV toolkit, or a training framework,
and **it is not at 1:1 parity with the full `timm` catalog, nor is it
likely to ever be.** `timm` has hundreds of architectures and thousands of
checkpoints; Jimm tracks the subset its contributors actually use.
Backbones land via PR.

## Correctness gate

Every registered variant has a parity test that downloads the real
`.safetensors` from HuggingFace, loads them through the Lux model, runs
the forward pass, and asserts max-abs-diff against `timm`'s output on
the same input is under `1e-3` (most variants land closer to `1e-4`).
That single test covers both the architecture port and the weight loader:
if the safetensors loader misroutes a tensor or applies the wrong axis
permutation, the forward output diverges and the test fails. CI is still
a work in progress (the full sweep is expensive and we are figuring out
how to run it well with the resources we have), but contributors can scope
runs to a single variant via `JIMM_TEST_VARIANTS`.

## How this code was produced (and a caveat)

Most of Jimm was written by AI agents driving the porting workflow
encoded in `.claude/skills/timm-to-lux/`, with human review at each
phase and the parity tests as the correctness backstop. The code is
already being used in real projects, so it works, but **expect bugs and
rough edges**, especially around features that the parity tests do not
exercise (anything past forward inference with the released weights).
File issues; we will fix them.

## Porting new backbones with Claude Code

If you have Claude Code and tokens to spare, the practical path to a new
backbone is: open the repo, ask Claude to port `timm/<your_model>`, and
follow the skill at `.claude/skills/timm-to-lux/`. It produces a working
PR for most `timm` architectures in a single session, handling the
fiddly bits (cross-correlation vs convolution, GroupNorm defaults,
weight standardization, NCHW vs WHCN). Complex architectures still need
a human in the loop when divergence shows up mid-network, but the
marginal cost of adding a new backbone is roughly "one model's worth of
inference tokens" rather than "one afternoon of careful
reimplementation."

**Contributions are welcome and encouraged**, with or without the skill.
See the README's "Contributing a new backbone" section for the
acceptance criteria. Bug reports, PRs for new variants of registered
families, and PRs for entirely new families are all in scope.

Thanks to Ross Wightman for `timm` and to the Julia ML ecosystem
maintainers that made the port plausible in the first place.
