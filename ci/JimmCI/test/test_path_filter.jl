using Test
using JimmCI.PathFilter

@testset "families_for_paths" begin
    @testset "no recognized files" begin
        @test isempty(families_for_paths(["docs/foo.md"]))
        @test isempty(families_for_paths(String[]))
    end

    @testset "single-family routing" begin
        @test families_for_paths(["src/Models/ResNet/layers.jl"]) == ["resnet"]
        @test families_for_paths(["src/Models/ResNetV2/bit.jl"]) == ["bit"]
        @test families_for_paths(["src/Models/ConvNeXt/x.jl"]) == ["convnext"]
        @test families_for_paths(["src/Models/ConvNeXtV2/x.jl"]) == ["convnextv2"]
    end

    @testset "test-file exact matches" begin
        @test families_for_paths(["test/test_resnet.jl"]) == ["resnet"]
        @test families_for_paths(["test/test_bit_resnet.jl"]) == ["bit"]
        @test families_for_paths(["test/test_convnext.jl"]) == ["convnext"]
        @test families_for_paths(["test/test_convnextv2.jl"]) == ["convnextv2"]
        @test families_for_paths(["test/test_init.jl"]) == ["infra"]
    end

    @testset "shared prefixes promote to all families" begin
        @test families_for_paths(["src/Layers/conv.jl"]) == collect(ALL_FAMILIES)
        @test families_for_paths(["src/Interop/foo.jl"]) == collect(ALL_FAMILIES)
        @test families_for_paths(["src/Models/ConvNeXtCommon/x.jl"]) ==
              collect(ALL_FAMILIES)
        @test families_for_paths(["ci/whatever.jl"]) == collect(ALL_FAMILIES)
    end

    @testset "shared exact files promote to all families" begin
        @test families_for_paths(["Project.toml"]) == collect(ALL_FAMILIES)
        @test families_for_paths(["Manifest.toml"]) == collect(ALL_FAMILIES)
        @test families_for_paths(["src/Jimm.jl"]) == collect(ALL_FAMILIES)
        @test families_for_paths(["test/runtests.jl"]) == collect(ALL_FAMILIES)
        @test families_for_paths(["test/_filter.jl"]) == collect(ALL_FAMILIES)
    end

    @testset "multi-family in canonical order" begin
        @test families_for_paths(["src/Models/ConvNeXt/x.jl", "src/Models/ResNet/y.jl"]) ==
              ["resnet", "convnext"]
    end

    @testset "shared wins over per-family" begin
        @test families_for_paths(["src/Models/ResNet/foo.jl", "src/Layers/conv.jl"]) ==
              collect(ALL_FAMILIES)
    end

    @testset "REPRESENTATIVE_VARIANT covers every family" begin
        for f in ALL_FAMILIES
            @test haskey(REPRESENTATIVE_VARIANT, f)
        end
    end
end
