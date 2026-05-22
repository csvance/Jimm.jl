# Shared scaffolding for the per-family parity test files
# (test_convnextv2.jl, test_convnext.jl, test_resnet.jl, test_bit_resnet.jl).
#
# Each family file is a thin shell: a variant list, an outer @testset for the
# family name, and a loop that loads the fixture, skips when missing or
# HF_OFFLINE=1, and then delegates the actual forward/load/assert work to
# run_variant_parity below.
#
# Include with:
#   isdefined(@__MODULE__, :run_variant_parity) || include("_parity_helpers.jl")

isdefined(@__MODULE__, :LOGITS_ATOL) || include("_parity_tol.jl")

parity_fixture_path(variant::Symbol; in_chans::Int = 3) = begin
    suffix = in_chans == 3 ? "" : "_in$(in_chans)c"
    base = get(ENV, "JIMM_PARITY_DIR", joinpath(@__DIR__, "..", "data", "parity"))
    joinpath(base, "$(variant)$(suffix)_io.h5")
end

# HDF5 fixtures store PyTorch-layout tensors; read_parity reverses axes to
# Lux-natural (W, H, C, N) for features and (K, N) for logits, which matches
# every family's Luximm output layout.
function load_parity_fixture(variant::Symbol; in_chans::Int = 3)
    path = parity_fixture_path(variant; in_chans = in_chans)
    isfile(path) || return nothing
    return Luximm.Interop.read_parity(path)
end

hf_offline() = get(ENV, "HF_OFFLINE", "") == "1"

# Runs the three parity sub-tests for one variant: forward_features,
# forward (logits) when the fixture ships them, and forward_features at
# in_chans=1 when a 1-channel fixture is available. `fixture` is the
# 3-channel fixture already loaded by the caller.
#
# The logits sub-test keys on `haskey(fixture.output, "logits")` alone:
# the timm dumper only writes that key for variants with a trained head,
# which matches the `default_num_classes(variant) > 0` rule the per-family
# files used previously.
function run_variant_parity(variant::Symbol, fixture)
    x = fixture.input
    expected_features = fixture.output["features"]

    @testset "forward_features" begin
        model, load = create_pretrained(variant; num_classes = 0)
        ps, st = Lux.setup(Xoshiro(0), model)
        st = Lux.testmode(st)
        ps, st = load(ps, st)
        y, _ = model(x, ps, st)
        @test size(y) == size(expected_features)
        diff = maximum(abs.(y .- expected_features))
        ref_scale = max(maximum(abs.(expected_features)), eps(Float32))
        rel = diff / ref_scale
        @info "$(variant) features max-abs-diff = $diff, rel = $rel"
        @test rel < FEATURES_RTOL
    end

    if haskey(fixture.output, "logits")
        expected_logits = fixture.output["logits"]
        @testset "forward (logits)" begin
            model, load = create_pretrained(variant)
            ps, st = Lux.setup(Xoshiro(0), model)
            st = Lux.testmode(st)
            ps, st = load(ps, st)
            y, _ = model(x, ps, st)
            @test size(y) == size(expected_logits)
            diff = maximum(abs.(y .- expected_logits))
            @info "$(variant) logits max-abs-diff = $diff"
            @test diff < LOGITS_ATOL
        end
    end

    fixture_in1c = load_parity_fixture(variant; in_chans = 1)
    if fixture_in1c === nothing
        @info "skipping $(variant) in_chans=1: fixture missing at " *
              parity_fixture_path(variant; in_chans = 1)
    else
        @testset "forward_features (in_chans=1)" begin
            x1 = fixture_in1c.input
            expected1 = fixture_in1c.output["features"]
            model, load = create_pretrained(variant; in_chans = 1, num_classes = 0)
            ps, st = Lux.setup(Xoshiro(0), model)
            st = Lux.testmode(st)
            ps, st = load(ps, st)
            y, _ = model(x1, ps, st)
            @test size(y) == size(expected1)
            diff = maximum(abs.(y .- expected1))
            ref_scale = max(maximum(abs.(expected1)), eps(Float32))
            rel = diff / ref_scale
            @info "$(variant) features (in_chans=1) max-abs-diff = $diff, rel = $rel"
            @test rel < FEATURES_RTOL
        end
    end
end
