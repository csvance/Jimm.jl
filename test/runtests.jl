using Test
using Jimm

include("_filter.jl")

if family_enabled("infra")
    @testset "Jimm scaffold" begin
        @test isdefined(Jimm, :Interop)
        @test isdefined(Jimm, :Layers)
        @test isdefined(Jimm, :Models)
        @test isdefined(Jimm.Interop, :read_parity)
        @test isdefined(Jimm.Interop, :apply_state_dict)
        @test isdefined(Jimm.Interop, :hf_download)
        @test isdefined(Jimm.Interop, :hf_hub_download)
        @test isdefined(Jimm.Interop, :hf_hub_cache_dir)
        @test isdefined(Jimm.Interop, :load_safetensors_state_dict)
        @test isdefined(Jimm.Interop, :adapt_input_conv)
        @test isdefined(Jimm.Layers, :std_conv)
        @test isdefined(Jimm.Layers, :layernorm2d)
        @test isdefined(Jimm.Layers, :grn_layer)
        @test isdefined(Jimm.Models, :bit_resnetv2)
        @test isdefined(Jimm.Models, :bit_resnetv2_mapping)
        @test isdefined(Jimm.Models, :load_bit_resnetv2_pretrained)
        @test isdefined(Jimm.Models, :BIT_VARIANTS)
        @test isdefined(Jimm.Models, :convnextv2)
        @test isdefined(Jimm.Models, :convnextv2_mapping)
        @test isdefined(Jimm.Models, :load_convnextv2_pretrained)
        @test isdefined(Jimm.Models, :CONVNEXTV2_VARIANTS)
        @test isdefined(Jimm.Models, :convnext)
        @test isdefined(Jimm.Models, :convnext_mapping)
        @test isdefined(Jimm.Models, :load_convnext_pretrained)
        @test isdefined(Jimm.Models, :CONVNEXT_VARIANTS)
    end

    include("test_hf_download.jl")
    include("test_hf_hub_download.jl")
    include("test_init.jl")
end

if family_enabled("bit")
    include("test_bit_resnet.jl")
end

if family_enabled("convnextv2")
    include("test_convnextv2.jl")
end

if family_enabled("convnext")
    include("test_convnext.jl")
end
