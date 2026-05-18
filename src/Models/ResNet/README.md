# ResNet

Classic ResNet ports matching timm.

| Variant key | timm / HuggingFace repo | Block | Features | Classes | Input |
|-------------|--------------------------|-------|----------|---------|-------|
| `:resnet18_a1_in1k` | `timm/resnet18.a1_in1k` | BasicBlock | 512 | 1000 | 224 |
| `:resnet34_a1_in1k` | `timm/resnet34.a1_in1k` | BasicBlock | 512 | 1000 | 224 |
| `:resnet50_a1_in1k` | `timm/resnet50.a1_in1k` | Bottleneck | 2048 | 1000 | 224 |
| `:resnet101_a1_in1k` | `timm/resnet101.a1_in1k` | Bottleneck | 2048 | 1000 | 224 |
| `:resnet152_a1_in1k` | `timm/resnet152.a1_in1k` | Bottleneck | 2048 | 1000 | 224 |

```julia
using Jimm, Lux, Random

model = resnet(:resnet18_a1_in1k; in_chans = 3, num_classes = 0)
ps, st = Lux.setup(Xoshiro(0), model)
ps, st = load_resnet_pretrained(ps, st, :resnet18_a1_in1k; num_classes = 0)
st = Lux.testmode(st)

x = randn(Float32, 224, 224, 3, 1)
features, _ = model(x, ps, st)  # (7, 7, 512, 1)
```
