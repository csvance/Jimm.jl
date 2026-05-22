# BiT ResNetV2 backbone, Lux port of timm's `resnetv2_<variant>_bit.goog_in21k`
# family. A single `bit_resnetv2(variant; ...)` constructor table-drives all
# variants from `BIT_VARIANTS`; `std_conv` lives in `Jimm.Layers` since other
# timm families (NFNet, etc.) reuse the same block.
#
# Architecture (identical across variants; only `stem_chs`, `stage_chs`,
# `depths` change with the width factor):
#   - Stem: StdConv2d(in=in_chans, out=stem_chs, k=7, s=2, p=3, no bias);
#     explicit zero-pad-1 then MaxPool(k=3, s=2, pad=0).
#   - Four pre-activation ResNetV2 stages with depths from the variant,
#     widths from the variant, first-block strides (1, 2, 2, 2).
#   - GroupNorm(groups=32, eps=1e-5, affine=true) + ReLU as activation.
#   - Weight standardization on every conv (population variance, eps=1e-8).
#   - Final GN+ReLU before features.
#   - Optional classification head (timm's `ClassifierHead(use_conv=True)`):
#     global average pool → 1×1 Conv((1,1), num_features => num_classes) →
#     flatten to (num_classes, N). Attached when `num_classes > 0`.
#
# `num_classes=0` returns the post-`final_norm` feature map (timm's
# `forward_features` output). `num_classes>0` returns logits (timm's
# `forward` output).

include("Config.jl")

# timm's `_init_weights` recipe for ResNetV2/BiT: every Conv2d (including
# StdConv2d) gets Kaiming-normal fan_out + ReLU gain; the classifier head
# Conv2d (timm names it `head.fc.*`) gets Normal(0, 0.01). Biases zero,
# affine norm params at ones/zeros (already Lux defaults). `zero_init_last`
# is `False` in `ResNetV2.__init__` by default, so we don't zero the last
# block's conv3 weight.
const _BIT_CONV_INIT = kaiming_normal_fan_out
const _BIT_HEAD_INIT = normal_init(; std = 0.01f0)

# Shared backbone forward, called from both branches of `bit_resnetv2`.
# Any future change to the backbone path (activation, padding, pool, etc.)
# lives here so the feature-extractor and classifier branches can't desync.
function _bit_resnetv2_features(x, stem_conv, stage1, stage2, stage3, stage4, final_norm)
    x = stem_conv(x)
    # timm BiT pads with **zeros** before the maxpool (ConstantPad2d(value=0)),
    # then pools with no internal padding. NNlib's maxpool(pad=1) uses
    # -Inf padding, which yields different output near negative-valued
    # boundaries. Pad explicitly with zeros to match timm.
    x = NNlib.pad_zeros(x, (1, 1, 1, 1, 0, 0, 0, 0))
    x = NNlib.maxpool(x, (3, 3); stride = 2, pad = 0)
    x = stage1(x)
    x = stage2(x)
    x = stage3(x)
    x = stage4(x)
    return final_norm(x)
end

"""
    bit_resnetv2(variant; in_chans=3, num_classes=0) -> @compact block

Build a BiT ResNetV2 backbone. `variant` is a key from [`BIT_VARIANTS`]
(e.g. `:resnetv2_50x1_bit_goog_in21k`).

When `num_classes == 0`, the forward pass returns the post-`final_norm`
feature map shaped `(W/32, H/32, num_features, N)`, matching
`timm.create_model(..., num_classes=0).forward_features(x)`.

When `num_classes > 0`, a `ClassifierHead`-style head is attached
(global avg pool → 1×1 conv → flatten) and the forward pass returns
logits shaped `(num_classes, N)`, matching
`timm.create_model(..., num_classes=num_classes).forward(x)`.
"""
function bit_resnetv2(variant::Symbol; in_chans::Int = 3, num_classes::Int = 0)
    cfg = get(BIT_VARIANTS, variant) do
        error(
            "Unknown BiT variant: $variant. Known variants: " *
            "$(sort(collect(keys(BIT_VARIANTS))))",
        )
    end
    depths = cfg.layers
    widths = cfg.stage_chs
    strides = (1, 2, 2, 2)
    stem_chs = cfg.stem_chs

    if num_classes == 0
        @compact(
            stem_conv = std_conv(
                7,
                7,
                in_chans,
                stem_chs;
                stride = 2,
                pad = 3,
                init_weight = _BIT_CONV_INIT,
            ),
            stage1 = resnet_stage(stem_chs, widths[1], depths[1], strides[1]),
            stage2 = resnet_stage(widths[1], widths[2], depths[2], strides[2]),
            stage3 = resnet_stage(widths[2], widths[3], depths[3], strides[3]),
            stage4 = resnet_stage(widths[3], widths[4], depths[4], strides[4]),
            final_norm = gn_act(widths[4]),
        ) do x
            @return _bit_resnetv2_features(
                x,
                stem_conv,
                stage1,
                stage2,
                stage3,
                stage4,
                final_norm,
            )
        end
    else
        nc = num_classes
        @compact(
            stem_conv = std_conv(
                7,
                7,
                in_chans,
                stem_chs;
                stride = 2,
                pad = 3,
                init_weight = _BIT_CONV_INIT,
            ),
            stage1 = resnet_stage(stem_chs, widths[1], depths[1], strides[1]),
            stage2 = resnet_stage(widths[1], widths[2], depths[2], strides[2]),
            stage3 = resnet_stage(widths[2], widths[3], depths[3], strides[3]),
            stage4 = resnet_stage(widths[3], widths[4], depths[4], strides[4]),
            final_norm = gn_act(widths[4]),
            head_fc = Conv(
                (1, 1),
                widths[4] => nc;
                use_bias = true,
                cross_correlation = true,
                init_weight = _BIT_HEAD_INIT,
                init_bias = zeros32,
            ),
        ) do x
            x = _bit_resnetv2_features(
                x,
                stem_conv,
                stage1,
                stage2,
                stage3,
                stage4,
                final_norm,
            )
            # timm's ClassifierHead(use_conv=True): global avg pool → 1×1 conv → flatten.
            # NNlib has no adaptive pool; compute the kernel from the actual spatial
            # extent so the path stays input-size-agnostic.
            x = NNlib.meanpool(x, size(x)[1:2]; pad = 0)   # (1, 1, num_features, N)
            x = head_fc(x)                                 # (1, 1, nc, N)
            @return reshape(x, (nc, size(x, 4)))           # (nc, N)
        end
    end
end

# -- gn_act: GroupNorm + ReLU ---------------------------------------------

"""
    gn_act(C; groups=32, eps=1f-5) -> @compact block

GroupNorm(groups, C; affine=true) followed by ReLU.
"""
function gn_act(C::Int; groups::Int = 32, eps::Float32 = 1.0f-5)
    @compact(gn = GroupNorm(C, groups; affine = true, epsilon = eps),) do x
        @return NNlib.relu.(gn(x))
    end
end

# -- preact_bottleneck ----------------------------------------------------

function preact_bottleneck(in_ch::Int, out_ch::Int, stride::Int; downsample::Bool)
    mid = out_ch ÷ 4

    if downsample
        @compact(
            norm1 = gn_act(in_ch),
            conv1 = std_conv(1, 1, in_ch, mid; init_weight = _BIT_CONV_INIT),
            norm2 = gn_act(mid),
            conv2 = std_conv(
                3,
                3,
                mid,
                mid;
                stride = stride,
                pad = 1,
                init_weight = _BIT_CONV_INIT,
            ),
            norm3 = gn_act(mid),
            conv3 = std_conv(1, 1, mid, out_ch; init_weight = _BIT_CONV_INIT),
            ds_conv = std_conv(
                1,
                1,
                in_ch,
                out_ch;
                stride = stride,
                init_weight = _BIT_CONV_INIT,
            ),
        ) do x
            x_pre = norm1(x)
            s = ds_conv(x_pre)
            y = conv1(x_pre)
            y = conv2(norm2(y))
            y = conv3(norm3(y))
            @return y .+ s
        end
    else
        @compact(
            norm1 = gn_act(in_ch),
            conv1 = std_conv(1, 1, in_ch, mid; init_weight = _BIT_CONV_INIT),
            norm2 = gn_act(mid),
            conv2 = std_conv(
                3,
                3,
                mid,
                mid;
                stride = stride,
                pad = 1,
                init_weight = _BIT_CONV_INIT,
            ),
            norm3 = gn_act(mid),
            conv3 = std_conv(1, 1, mid, out_ch; init_weight = _BIT_CONV_INIT),
        ) do x
            x_pre = norm1(x)
            y = conv1(x_pre)
            y = conv2(norm2(y))
            y = conv3(norm3(y))
            @return y .+ x
        end
    end
end

# -- resnet_stage ---------------------------------------------------------

function resnet_stage(in_ch::Int, out_ch::Int, depth::Int, stride::Int)
    blocks = []
    push!(blocks, preact_bottleneck(in_ch, out_ch, stride; downsample = true))
    for _ = 2:depth
        push!(blocks, preact_bottleneck(out_ch, out_ch, 1; downsample = false))
    end
    return Chain(blocks...)
end

# -- Pretrained-weight loading -------------------------------------------

"""
    bit_resnetv2_mapping(state_dict, variant;
                          load_classifier=false, in_chans=3, prefix=()) -> Vector

Build the `(pytorch_key, lux_path, transform)` triples that move a timm
`resnetv2_<variant>_bit` state_dict into the Lux tree produced by
[`bit_resnetv2`](@ref). When `load_classifier=true`, the head keys
(`head.fc.weight`, `head.fc.bias`) are also mapped. Pass `prefix` to
address a backbone nested inside a larger `@compact` model (e.g.
`prefix = (:backbone,)`). Suitable for `apply_state_dict`.

Assumes the state dict was loaded with `load_safetensors_state_dict`
(default `reverse_axes=true`) or read from a parity HDF5 fixture: both
deliver conv weights in `(kW, kH, in, out)` order, which is Lux's `Conv`
layout, so the `identity` transform is correct for every leaf.
"""
function bit_resnetv2_mapping(
    state_dict::Dict,
    variant::Symbol;
    load_classifier::Bool = false,
    in_chans::Int = 3,
    prefix::Tuple{Vararg{Symbol}} = (),
)
    cfg = get(BIT_VARIANTS, variant) do
        error(
            "Unknown BiT variant: $variant. Known variants: " *
            "$(sort(collect(keys(BIT_VARIANTS))))",
        )
    end
    mapping = Tuple{String,Tuple{Vararg{Symbol}},Function}[]

    # The released checkpoint always has the 3-channel stem weight; adapt it
    # on the fly when the model was built with a different in_chans, matching
    # timm's adapt_input_conv.
    stem_w_transform = in_chans == 3 ? identity : adapt_input_conv(in_chans)
    push!(
        mapping,
        ("stem.conv.weight", (prefix..., :stem_conv, :conv, :weight), stem_w_transform),
    )

    for (s, depth) in enumerate(cfg.layers)
        stage_sym = Symbol("stage", s)
        for b = 1:depth
            layer_sym = Symbol("layer_", b)
            py_block = "stages.$(s - 1).blocks.$(b - 1)"
            for n = 1:3
                norm_sym = Symbol("norm", n)
                conv_sym = Symbol("conv", n)
                push!(
                    mapping,
                    (
                        "$(py_block).norm$(n).weight",
                        (prefix..., stage_sym, layer_sym, norm_sym, :gn, :scale),
                        identity,
                    ),
                )
                push!(
                    mapping,
                    (
                        "$(py_block).norm$(n).bias",
                        (prefix..., stage_sym, layer_sym, norm_sym, :gn, :bias),
                        identity,
                    ),
                )
                push!(
                    mapping,
                    (
                        "$(py_block).conv$(n).weight",
                        (prefix..., stage_sym, layer_sym, conv_sym, :conv, :weight),
                        identity,
                    ),
                )
            end
            if b == 1
                push!(
                    mapping,
                    (
                        "$(py_block).downsample.conv.weight",
                        (prefix..., stage_sym, layer_sym, :ds_conv, :conv, :weight),
                        identity,
                    ),
                )
            end
        end
    end

    push!(mapping, ("norm.weight", (prefix..., :final_norm, :gn, :scale), identity))
    push!(mapping, ("norm.bias", (prefix..., :final_norm, :gn, :bias), identity))

    if load_classifier
        push!(mapping, ("head.fc.weight", (prefix..., :head_fc, :weight), identity))
        push!(mapping, ("head.fc.bias", (prefix..., :head_fc, :bias), identity))
    end

    for (pykey, _, _) in mapping
        haskey(state_dict, pykey) ||
            error("mapping references missing state_dict key: $pykey")
    end
    return mapping
end

"""
    _load_bit_resnetv2(ps, st, variant; in_chans, num_classes,
                       revision, cache_dir, prefix) -> (ps, st)

Private back-end for `create_pretrained` on BiT ResNetV2 variants.
Resolves `model.safetensors` for `variant` on HuggingFace (cached on
disk under the standard HF Hub layout, so the same blob is shared
with any `timm` / `huggingface_hub` install on the machine), loads it
via `load_safetensors_state_dict`, and rebuilds `ps` with the BiT
weights applied at `ps.<prefix...>`. `st` is returned unchanged
(GroupNorm has no running statistics); the uniform `(ps, st)` shape
mirrors the other family loaders.

`in_chans` and `num_classes` are forwarded from the closure captured
at `create_pretrained` time. Three classifier-head cases:

- `num_classes == 0`: backbone-only load, feature extractor.
- `num_classes == default_num_classes(variant)`: full load including
  classifier.
- `num_classes` differs from the variant's default: the backbone is
  loaded and a `@warn` is emitted; the user's custom classifier is
  left at its `Lux.setup` random initialization for them to train.
"""
function _load_bit_resnetv2(
    ps,
    st,
    variant::Symbol;
    in_chans::Int,
    num_classes::Int,
    revision::AbstractString,
    cache_dir::AbstractString,
    prefix::Tuple{Vararg{Symbol}},
)
    cfg = BIT_VARIANTS[variant]
    load_classifier = num_classes > 0 && num_classes == cfg.default_num_classes
    if num_classes > 0 && num_classes != cfg.default_num_classes
        @warn "variant $variant ships $(cfg.default_num_classes)-class pretrained weights, " *
              "but the model has a $num_classes-class head. Loading the backbone only; " *
              "the classifier head is left at its Lux.setup random initialization for you to train."
    end
    path = hf_hub_download(
        cfg.hf_repo,
        "model.safetensors";
        revision = revision,
        cache_dir = cache_dir,
    )
    sd = load_safetensors_state_dict(path)
    ps = apply_state_dict(
        ps,
        sd,
        bit_resnetv2_mapping(
            sd,
            variant;
            load_classifier = load_classifier,
            in_chans = in_chans,
            prefix = prefix,
        ),
    )
    return ps, st
end
