# Statistical sanity checks that `Lux.setup` on `bit_resnetv2` and
# `convnextv2` produces weight distributions matching timm's
# `_init_weights` recipes. We can't match PyTorch's RNG bit-for-bit, so
# tolerances are generous; the goal is to catch a regression to
# `glorot_uniform` or to an unintentionally non-zero bias init.

using Test
using Jimm
using Lux
using Random
using Statistics

# Helper: empirical std should sit close to the analytic value for an
# array large enough that the sample std converges. 12% works comfortably
# for the smallest layer we touch (40 elements in a GRN scale-bias is too
# small to use this; we test those exactly below).
function _approx_std(actual::Real, target::Real; rtol::Real = 0.12)
    return isapprox(actual, target; rtol = rtol)
end

@testset "timm random-init recipes" begin
    @testset "BiT ResNetV2 r50x1" begin
        model = bit_resnetv2(:resnetv2_50x1_bit_goog_in21k;
                              in_chans = 3, num_classes = 21843)
        ps, _ = Lux.setup(Xoshiro(0), model)

        # Stem: 7x7, in=3, out=64. Kaiming fan_out → std = sqrt(2/(64*49)).
        w_stem = ps.stem_conv.conv.weight
        @test size(w_stem) == (7, 7, 3, 64)
        @test _approx_std(std(w_stem), sqrt(2 / (64 * 7 * 7)))

        # A bottleneck conv (3x3, mid channels) should also be Kaiming fan_out.
        w_b3 = ps.stage2.layer_1.conv2.conv.weight
        kw, kh, _, oc = size(w_b3)
        @test _approx_std(std(w_b3), sqrt(2 / (oc * kw * kh)))

        # Head: timm's Normal(0, 0.01) on `head.fc.*`, zero bias.
        w_head = ps.head_fc.weight
        @test _approx_std(std(w_head), 0.01)
        @test all(ps.head_fc.bias .== 0)

        # GroupNorm scales/biases: 1 / 0 across the network.
        @test all(ps.final_norm.gn.scale .== 1)
        @test all(ps.final_norm.gn.bias .== 0)
        @test all(ps.stage3.layer_2.norm1.gn.scale .== 1)
        @test all(ps.stage3.layer_2.norm1.gn.bias .== 0)
    end

    @testset "ConvNeXtV2 atto (ft_in1k)" begin
        model = convnextv2(:convnextv2_atto_fcmae_ft_in1k;
                            in_chans = 3, num_classes = 1000)
        ps, _ = Lux.setup(Xoshiro(0), model)

        # Stem 4x4 conv: trunc_normal(std=0.02), zero bias.
        w_stem = ps.stem_conv.weight
        @test size(w_stem) == (4, 4, 3, 40)
        @test _approx_std(std(w_stem), 0.02)
        @test all(ps.stem_conv.bias .== 0)

        # Stem LN2d: ones / zeros.
        @test all(ps.stem_norm.scale .== 1)
        @test all(ps.stem_norm.bias .== 0)

        # Stage 4, block 1 (post-downsample): depthwise + 1x1 fc1/fc2.
        blk = ps.stage4.blocks.layer_1
        @test _approx_std(std(blk.conv_dw.weight), 0.02)
        @test all(blk.conv_dw.bias .== 0)
        @test _approx_std(std(blk.fc1.weight), 0.02)
        @test all(blk.fc1.bias .== 0)
        @test _approx_std(std(blk.fc2.weight), 0.02)
        @test all(blk.fc2.bias .== 0)

        # GRN gamma/beta init to zero; block LN2d to ones/zeros.
        @test all(blk.grn.scale .== 0)
        @test all(blk.grn.bias .== 0)
        @test all(blk.norm.scale .== 1)
        @test all(blk.norm.bias .== 0)

        # Downsample (stages 1-3 only): LN2d ones/zeros + Conv2x2 trunc_normal.
        ds = ps.stage2.downsample
        @test all(ds.norm.scale .== 1)
        @test all(ds.norm.bias .== 0)
        @test _approx_std(std(ds.conv.weight), 0.02)
        @test all(ds.conv.bias .== 0)

        # Head Dense: trunc_normal(std=0.02), zero bias. Head LN2d ones/zeros.
        @test _approx_std(std(ps.head_fc.weight), 0.02)
        @test all(ps.head_fc.bias .== 0)
        @test all(ps.head_norm.scale .== 1)
        @test all(ps.head_norm.bias .== 0)
    end
end
