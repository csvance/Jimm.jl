# Pkg.test() entry point. The real work lives in `_ci_driver.jl`, which is
# shared with the self-hosted CI builder.
include("_ci_driver.jl")
