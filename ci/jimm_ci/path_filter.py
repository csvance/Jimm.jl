"""Map changed file paths to the set of Jimm test families they affect.

Replaces the dorny/paths-filter action with a small, auditable function. The
rules mirror the layout of src/Models/* and test/test_*.jl in the Jimm.jl
repository.
"""
from __future__ import annotations

from collections.abc import Iterable

# Paths that affect every family. Any match here promotes the result to all
# families, regardless of which other files changed.
_SHARED_PREFIXES: tuple[str, ...] = (
    "src/Layers/",
    "src/Interop/",
    "ci/",
)

_SHARED_EXACT: frozenset[str] = frozenset({
    "src/Jimm.jl",
    "Project.toml",
    "Manifest.toml",
    "test/runtests.jl",
    "test/_filter.jl",
})

_FAMILY_PREFIXES: dict[str, tuple[str, ...]] = {
    "bit":        ("src/Models/ResNetV2/",),
    "convnext":   ("src/Models/ConvNeXt/",),
    "convnextv2": ("src/Models/ConvNeXtV2/",),
    "infra":      (),
}

_FAMILY_EXACT: dict[str, frozenset[str]] = {
    "bit":        frozenset({"test/test_bit_resnet.jl"}),
    "convnext":   frozenset({"test/test_convnext.jl"}),
    "convnextv2": frozenset({"test/test_convnextv2.jl"}),
    "infra":      frozenset({
        "test/test_hf_download.jl",
        "test/test_hf_hub_download.jl",
        "test/test_init.jl",
    }),
}

ALL_FAMILIES: tuple[str, ...] = ("infra", "bit", "convnext", "convnextv2")

# Variant key to use when a family is included in a PR-scope run. Empty string
# means "no variant filter", i.e. run the family's full test set; used for
# infra and for master/nightly sweeps.
REPRESENTATIVE_VARIANT: dict[str, str] = {
    "infra":      "",
    "bit":        "resnetv2_50x1_bit_goog_in21k_ft_in1k",
    "convnext":   "convnext_tiny_fb_in1k",
    "convnextv2": "convnextv2_atto_fcmae_ft_in1k",
}


def _is_shared(path: str) -> bool:
    if path in _SHARED_EXACT:
        return True
    return any(path.startswith(p) for p in _SHARED_PREFIXES)


def _family_for(path: str) -> str | None:
    for fam, prefixes in _FAMILY_PREFIXES.items():
        if any(path.startswith(p) for p in prefixes):
            return fam
        if path in _FAMILY_EXACT[fam]:
            return fam
    return None


def families_for_paths(paths: Iterable[str]) -> tuple[str, ...]:
    """Return families whose tests should run, in canonical order.

    An empty result means CI should be skipped entirely (no recognized files
    changed).
    """
    paths = list(paths)
    if any(_is_shared(p) for p in paths):
        return ALL_FAMILIES
    touched: set[str] = set()
    for p in paths:
        fam = _family_for(p)
        if fam is not None:
            touched.add(fam)
    return tuple(f for f in ALL_FAMILIES if f in touched)
