# Jimm.jl

Julia ports of [`timm`](https://github.com/huggingface/pytorch-image-models)
(PyTorch Image Models, by Ross Wightman) backbones for
[Lux.jl](https://lux.csail.mit.edu/), with pretrained weights loaded
directly from HuggingFace Hub in `.safetensors` format. The name is an
homage to the project we port from.

## Motivation

Jimm exists because we needed Julia's SciML ecosystem together with
modern pretrained vision backbones, and Python does not have a peer
for SciML. The original stack was vision encoders feeding `torchdiffeq`
in PyTorch, which works but leaves much to be desired. Moving the
DiffEQ side to Julia meant the vision side had to come too. Jimm
started as a one-off port of a single backbone for that internal use
case and snowballed. If your work also sits at the intersection of
pretrained vision encoders and the rest of the SciML stack, Jimm aims
to make Julia a more complete option for that workload.

## Status and caveats

Most of Jimm was written by AI agents driving the porting workflow
encoded in `.claude/skills/timm-to-lux/`, with human review at each
phase and the parity tests (see [acceptance
criteria](#acceptance-criteria) below) as the correctness backstop. The
code is already being used in real projects, so the registered
backbones work for forward inference with the released weights. That
said: **expect bugs and rough edges**, especially around anything the
parity tests do not exercise (custom training loops, mixed-precision
paths, exotic input shapes). File issues and PRs; we will fix them.

The package is also not at 1:1 parity with the full `timm` catalog and
is not likely to ever be. `timm` ships hundreds of architectures and
thousands of pretrained checkpoints; Jimm tracks only the subset its
contributors actually use. New backbones land via PR (see
[Contributing a new backbone](#contributing-a-new-backbone)).

## What Jimm is

Jimm is a strict Lux.jl port of `timm`: same architectures, same
hyperparameters, same weight initialization, same `state_dict` key layout.
The goal is that any HuggingFace `timm/<variant>` checkpoint loads into the
corresponding Jimm model without manual rewiring, and that the forward pass
matches `timm` to within float32 round-off. **Compatibility with `timm` is
the project's #1 priority**; if the two diverge, `timm` is the reference.

## What Jimm is not

Jimm is not a redesign or a Julia-native reimagining of image backbones.
It does not introduce new naming or "improved" defaults, and the layers it
ships (`src/Layers/`) are only those `timm` itself provides, ported to
match. It is not a general computer-vision toolkit: no datasets, no
training loops, no augmentation pipelines, no detection or segmentation
heads beyond what `timm` itself exposes on a backbone. Anything that would
cause a Jimm model to diverge numerically from its `timm` counterpart is
out of scope.

## Available backbones

| family            | constructor    | num weights | weight license     | docs                                                                 |
|-------------------|----------------|-------------|--------------------|----------------------------------------------------------------------|
| BiT ResNetV2      | `bit_resnetv2` | 15          | Apache 2.0         | [src/Models/ResNetV2/README.md](src/Models/ResNetV2/README.md)       |
| ConvNeXt          | `convnext`     | 19          | Apache 2.0         | [src/Models/ConvNeXt/README.md](src/Models/ConvNeXt/README.md)       |
| ConvNeXt (DINOv3) | `convnext`     | 4           | DINOv3 License ⚠️  | [src/Models/ConvNeXt/README.md](src/Models/ConvNeXt/README.md)       |
| ConvNeXt V2       | `convnextv2`   | 26          | CC BY-NC 4.0 ⚠️    | [src/Models/ConvNeXtV2/README.md](src/Models/ConvNeXtV2/README.md)   |

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

## Quickstart

```julia
using Jimm, Lux, Random

# Backbone features only: returns (W/32, H/32, num_features, N), matching
# timm.create_model(..., num_classes=0).forward_features(x).
model = bit_resnetv2(:resnetv2_50x1_bit_goog_in21k;
                     in_chans = 3, num_classes = 0)
ps, st = Lux.setup(Xoshiro(0), model)
ps = load_bit_resnetv2_pretrained(ps, :resnetv2_50x1_bit_goog_in21k;
                                   num_classes = 0)
x = randn(Float32, 224, 224, 3, 1)
features, _ = model(x, ps, st)            # (7, 7, 2048, 1)

# Classifier head: returns (num_classes, N), matching timm.forward(x).
model = convnextv2(:convnextv2_atto_fcmae_ft_in1k;
                   in_chans = 3, num_classes = 1000)
ps, st = Lux.setup(Xoshiro(0), model)
ps = load_convnextv2_pretrained(ps, :convnextv2_atto_fcmae_ft_in1k;
                                num_classes = 1000)
logits, _ = model(x, ps, st)              # (1000, 1)
```

Jimm uses Lux's `(W, H, C, N)` array layout throughout; PyTorch's
`(N, C, H, W)` is read-reversed at load time so most weights land in
the layout Lux expects directly. A small per-tensor transform table in
each family's mapping function handles the residual cases (Dense
weights, LayerNorm scale/bias).

### Non-RGB inputs

To use a backbone on grayscale or other non-3-channel inputs, pass the
same `in_chans` to both the constructor and the loader. The loader
adapts the released 3-channel stem weight via `Jimm.adapt_input_conv`,
matching timm's `adapt_input_conv` semantics (sum across input channels
for `in_chans = 1`; tile and rescale by `3 / in_chans` for other
counts):

```julia
model = convnextv2(:convnextv2_atto_fcmae; in_chans = 1, num_classes = 0)
ps, st = Lux.setup(Xoshiro(0), model)
ps = load_convnextv2_pretrained(ps, :convnextv2_atto_fcmae;
                                num_classes = 0, in_chans = 1)
x = randn(Float32, 224, 224, 1, 1)
features, _ = model(x, ps, st)            # (7, 7, 320, 1)
```

## Pretrained weights and the HuggingFace cache

`load_<family>_pretrained` resolves `model.safetensors` against the
standard HuggingFace Hub cache layout (`HF_HUB_CACHE` → `$HF_HOME/hub`
→ `~/.cache/huggingface/hub`), so the same blob is shared with `timm`
and `huggingface_hub`: whichever tool downloads first, the other sees
a cache hit. Subsequent calls short-circuit on the cached snapshot
symlink.

For lower-level access, `Jimm.Interop.hf_hub_download(repo_id, filename;
revision = "main")` returns the resolved snapshot path directly and
mirrors the cache semantics of `huggingface_hub.hf_hub_download`.

## Installation

**Julia (required):**

```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

**Python (only needed to regenerate parity fixtures for new backbones):**

```
uv sync
```

`pyproject.toml` declares the sidecar environment (Python 3.12+) with
`timm`, `torch`, `h5py`, `safetensors`, and `huggingface-hub`.

### Running a subset of the tests

Each parity test downloads pretrained weights and runs a full forward
through the network, so the whole sweep is expensive on a constrained
machine. Two environment variables narrow what runs without editing
files:

- `JIMM_TEST_VARIANTS`: comma-separated variant keys (e.g.
  `convnextv2_atto_fcmae`) to keep within each family's parity sweep.
  Setting this alone also restricts the active families to whichever ones
  contain the listed variants and drops the infra family, so a single
  variant key is enough to scope a run.
- `JIMM_TEST_FAMILIES`: comma-separated list of test families. Recognized
  values are `infra` (scaffold, HuggingFace download, init recipes), `bit`,
  `convnext`, and `convnextv2`. When set, this is authoritative and
  overrides the family inference described above. Unset and
  `JIMM_TEST_VARIANTS` also unset means every family runs.

```
# Just the ConvNeXtV2 atto fcmae forward and in_chans=1 parity tests,
# nothing else (no infra, no BiT):
JIMM_TEST_VARIANTS=convnextv2_atto_fcmae \
    julia --project -e 'using Pkg; Pkg.test()'

# To include the infra checks alongside a single backbone variant:
JIMM_TEST_FAMILIES=infra,convnextv2 \
JIMM_TEST_VARIANTS=convnextv2_atto_fcmae \
    julia --project -e 'using Pkg; Pkg.test()'
```

Parity tests also skip when their HDF5 fixture is missing under
`data/parity/`, so a contributor can dump one variant's fixture (and
optionally its `_in1c` companion) and run just that one without
touching the test code.

### One-shot per-variant test

For the common case (one variant, dump-if-missing-then-test),
`scripts/test_variant.sh` wraps the fixture-dump and Julia invocation:

```
# Resolve family, dump fixture if absent, run only this variant's testset:
scripts/test_variant.sh convnextv2_atto_fcmae

# Same, but for the in_chans=1 parity test (dumps the _in1c fixture):
scripts/test_variant.sh convnextv2_atto_fcmae --in-chans 1

# Force a fresh fixture dump even if one already exists:
scripts/test_variant.sh convnextv2_atto_fcmae --force
```

The script resolves the family from the variant prefix, calls the
appropriate Python sidecar under `test/parity/` via `uv run` if the
HDF5 fixture is missing, then runs the Julia test suite with
`JIMM_TEST_VARIANTS=<variant>` set. Requires both `uv` (for the Python
sidecar) and `julia` on `PATH`.

## Contributing a new backbone

PRs are welcome, especially for backbones not yet in the table above.
The bar is parity with `timm`; mechanics below.

### Acceptance criteria

A new backbone is mergeable when all four hold:

1. **Pretrained parity.** Forward output of the Lux model with weights
   loaded via `load_<family>_pretrained` matches `timm`'s forward output
   on the same input to within the test-suite `TOL` (currently `1f-3`
   max-abs-diff) for features (`num_classes = 0`) and logits
   (`num_classes > 0`). Existing models land well inside this: BiT
   ResNetV2-50 features around `1.5e-4`, its logits around `2e-5`;
   ConvNeXtV2 atto comparable.
2. **Random-init parity.** With the Lux model initialized using the same
   `_init_weights` recipe `timm` uses for the family, and `timm`
   initialized with the same RNG seed, forward outputs match to within
   the same `TOL`. See `_CN2_INIT` at `src/Models/ConvNeXtV2/Model.jl`
   (`truncated_normal(mean=0f0, std=0.02f0)` mirroring timm's
   `trunc_normal_(std=0.02)`) for a worked example.
3. **State-dict round-trip.** The mapping function consumes every PyTorch
   `state_dict` key. Mappings raise on a missing key so silent random-init
   leaks cannot pass parity by coincidence; see the assertion block at
   the end of `convnextv2_mapping` in `src/Models/ConvNeXtV2/Model.jl`.
4. **Variant table entry.** New variants are registered in
   `<FAMILY>_VARIANTS` (`src/Models/<Family>/Config.jl`) with their HF repo
   id and default class count, and listed in the table above.

### Why the bar is `~1e-3`, not `1e-5`

Float32 round-off accumulates through the depth of a network. Each conv,
norm, and weight-standardization stage reorders sums in ways that differ
from PyTorch's BLAS / cuDNN kernels even when the math is semantically
identical, so a 50-layer ResNet against the timm reference reliably lands
near `1e-4` absolute diff on `O(1)` activations. The shallower head (one
pool + one Dense) sits closer to `1e-5` because almost no new
accumulation happens after the backbone.

`TOL = 1f-3` is the realistic test gate: tight enough to catch the
silent-divergence failure modes that actually matter (cross-correlation
vs convolution, population vs sample variance, GELU approximation
mismatch, axis permutations, missing norm `epsilon`), and loose enough
not to fail on legitimate float32 reordering. If your port reports a
max-abs-diff in the `1e-2` range or higher, that is a real bug, not
round-off; bisect it with the per-stage parity hooks described in
`.claude/skills/timm-to-lux/SKILL.md`.

### Workflow with Claude Code

The full port workflow is encoded as a Claude Code skill that loads
automatically when you run Claude Code inside this repo:

- `.claude/skills/timm-to-lux/SKILL.md` — the seven-phase port-and-verify
  workflow: capture `timm` parity fixtures via the Python sidecar in
  `test/parity/`, scaffold the Lux model under `src/Models/<Family>/`,
  implement layers with `@compact`, wire the HuggingFace `.safetensors`
  loader, and verify parity end-to-end then bisect with per-stage
  fixtures on divergence.
- `.claude/skills/kaimon-julia/SKILL.md` — the underlying Julia REPL
  workflow (Revise, Kaimon MCP) the porting skill depends on.

If you have Claude Code, the practical path is: open this repo, ask
Claude to port `timm/<your_model>`, and follow the skill. The
implementations of BiT ResNetV2 and ConvNeXtV2 are the worked examples
the skill points at.

### Checklist for the PR

- [ ] Python parity sidecar checked in under `test/parity/`.
- [ ] Julia parity test under `test/test_<family>.jl` runs and passes.
- [ ] `load_<family>_pretrained` works against the live HuggingFace
      `model.safetensors`.
- [ ] Variant(s) registered in `<FAMILY>_VARIANTS`.
- [ ] README variant table updated.

## Project layout

- `src/Layers/` — reusable building blocks (`std_conv`, `layernorm2d`,
  `grn_layer`) and timm-equivalent initializers (`kaiming_normal_fan_out`,
  `normal_init`) shared across model families.
- `src/Models/` — per-family model files. Each family directory contains
  a constructor, a `Config.jl` variant table, a PyTorch `state_dict`
  mapping, and a `load_<family>_pretrained` loader.
- `src/Interop/` — PyTorch/HuggingFace interop: HDF5 parity fixtures
  (`read_parity`, `apply_state_dict`), HuggingFace `.safetensors`
  downloads (`hf_hub_download`, `load_safetensors_state_dict`).
- `test/parity/` — Python sidecars that dump `timm` parity fixtures
  (HDF5) for the Julia test suite to consume.
- `.claude/skills/` — Claude Code skills that encode the port workflow.

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
