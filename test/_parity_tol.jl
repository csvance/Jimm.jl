# Shared parity tolerance constants for all family test files.
# Include with: isdefined(@__MODULE__, :LOGITS_ATOL) || include("_parity_tol.jl")

const LOGITS_ATOL = 1f-3
# Features parity uses a relative bar: max-abs diff divided by max-abs of the
# timm reference. Deep backbones accumulate FP32 rounding through many stages,
# inflating raw pre-norm feature diffs even when downstream logits stay tight,
# so an absolute ceiling there gives false negatives.
const FEATURES_RTOL = 1f-4