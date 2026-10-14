# 自定义无标签图像数据集：自监督继续预训练

本流程从官方 LVD-1689M backbone 权重开始，在无标签图像上继续训练。
训练入口为 `dinov3.train.custom`，适用于任意符合下述格式的无标签 RGB 图像数据集。
ViT 使用 DINO + iBOT + KoLeo；ConvNeXt 使用 DINO + KoLeo。
ConvNeXt 当前实现忽略输入 mask，因此不能将它的同图像 patch 一致性训练称为 iBOT 遮挡学习。
本配置是小算力领域适配的实验起点，不是官方大规模预训练配方，也不保证下游性能提升。
默认增强参数沿用此前内镜实验的温和设置，迁移到其他领域时应按图像特点调整裁剪、颜色扰动等参数。

## 数据目录

不需要类别目录、COCO JSON、框或分割 mask。以下两种组织方式都可以：

```text
my_dataset/images/
├── frame_000001.jpg
├── frame_000002.png
└── 检查_001/
    ├── frame_000003.jpg
    └── frame_000004.jpg
```

`--data-root` 指向 `my_dataset/images`，递归查找 jpg/jpeg/png/bmp/tif/tiff/webp（扩展名不区分大小写）。
下面以此前本地验证使用的内镜图像目录为例，可替换成自己的图像根目录：

```text
/media/inno/data/花环征/TrainData/COCO_FORMAT_Barrett_V1/images/train
```

可选：使用 `--manifest /path/to/train.txt`，每行一个图像路径，UTF-8 编码。
相对路径以 `--data-root` 为基准；也支持绝对路径。清单中不写标签或其他列。

```text
frame_000001.jpg
检查_001/frame_000003.jpg
```

大量图像建议预先生成清单，避免每次启动都递归扫描。Dataset 不会自动去重、过滤坏图或按样本来源划分数据；
对于内镜数据，应自行按患者或检查划分。
坏图会报错并停止训练，先做数据质检。路径支持中文、空格和等号，但数据路径不能包含冒号。

## 环境和权重

本机已使用 `/home/inno/anaconda3/envs/torch2.7.1/bin/python`（PyTorch 2.7.1+cu118、torchvision 0.22.1+cu118）运行。
可以在仓库根目录激活对应环境：

```bash
conda activate torch2.7.1
```

服务器使用兼容 CUDA 的 PyTorch/torchvision 和仓库依赖；本入口复用官方 FSDP2，旧版 PyTorch 可能没有所需 API。
不需要为了直接使用 torchrun 启动而安装 SLURM。

默认 `--weights-root /media/inno/Pretrained/DINOV3`，递归寻找所选模型的一个
`dinov3_<model>_pretrain_lvd1689m-*.pth` 文件。
找到多个或未找到时，使用 `--weights '/完整路径/model.pth'` 显式指定。

支持 `vitb16`（默认）、`vits16`、`vits16plus`、`vitl16`、`vith16plus`，以及
`convnext_tiny/small/base/large`。当前配置面向 web/LVD 权重；SAT ViT-L 的结构设置不同，不能直接套用。
ViT-S+/16 和 ViT-H+/16 会自动选择 SwiGLU、FFN ratio=6，匹配官方权重结构。
对应的 `--model` 参数分别为 `vits16plus`、`vith16plus`，权重查找方式与其他型号一致。

启动新训练时，先用官方权重初始化 student 的 backbone，并随机初始化它的 DINO/iBOT 投影头，
再将 student 的完整状态（backbone + 投影头）复制一次到 EMA teacher，使两者从相同状态开始。
这不是将已有预训练 teacher 替换成随机网络：复制时 student 的 backbone 已加载官方权重。
加载严格校验键名与形状；不会用 `strict=False` 静默跳过不匹配的参数。

DINO 使用两个同结构网络进行自监督学习。student 通过梯度下降训练；teacher 不反向传播，
为同一图像的不同视角提供目标分布。每次 student 更新后，teacher 的参数按
`teacher = m * teacher + (1 - m) * student` 做指数移动平均，而不是直接覆盖。
投影头也需要一起初始化复制，使初始 prototype 输出空间一致。
即使初始参数相同，不同裁剪、ViT 的输入遮挡和 teacher/student 的目标处理仍会产生学习信号。
EMA 使目标变化更平缓；避免特征坍塌还依赖目标分配、温度和损失等机制，不能仅靠 EMA 保证。
恢复训练时，完整 checkpoint 会恢复此前各自的 student、teacher 和 optimizer 状态。

## 本地单卡：验证真实训练流程

先检查路径、图像数量和最终配置，不使用 GPU：

```bash
python -m dinov3.train.custom \
  --data-root '/media/inno/data/花环征/TrainData/COCO_FORMAT_Barrett_V1/images/train' \
  --model vitb16 --output-dir /tmp/dinov3_custom_vitb_demo \
  --batch-size 2 --num-workers 0 --smoke-test --steps 5 --dry-run
```

ViT-B 运行 5 个 optimizer step，包含反向传播、EMA 更新、checkpoint 保存和 backbone 导出：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  -m dinov3.train.custom \
  --data-root '/media/inno/data/花环征/TrainData/COCO_FORMAT_Barrett_V1/images/train' \
  --model vitb16 --output-dir /tmp/dinov3_custom_vitb_demo \
  --batch-size 2 --num-workers 0 --smoke-test --steps 5
```

ConvNeXt-Tiny 使用同一个入口：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  -m dinov3.train.custom \
  --data-root '/media/inno/data/花环征/TrainData/COCO_FORMAT_Barrett_V1/images/train' \
  --model convnext_tiny --output-dir /tmp/dinov3_custom_convnext_demo \
  --batch-size 8 --num-workers 2 --smoke-test --steps 5
```

单卡也通过 torchrun 创建分布式进程组。`--batch-size` 是每卡原始图像数，
每张图像还会产生 2 个 global crop 和多个 local crop。
`--smoke-test` 默认缩短为 5 步、2 个 local crop，关闭学习率/teacher 温度预热，便于验证实际参数更新。
这只能验证工程流程，不能据此判断模型适配效果。

## 服务器四卡：自定义数据集训练

建议先从 ViT-B/16 做主干适配，完成独立评估后再决定是否扩大到 ViT-L/16。
4 张 3090 的总显存不等于单卡 96 GB；需要同时考虑每卡激活、teacher 和 heads 的开销。
ViT-L 已在本机单卡 3090 上以 batch=2 完成 3 步训练；更大的 batch 和四卡运行仍需实测。

在服务器仓库根目录执行（替换真实数据、权重和输出路径）：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m dinov3.train.custom \
  --data-root /path/to/my_dataset/images \
  --weights-root /path/to/Pretrained/DINOV3 \
  --model vitb16 --output-dir /path/to/output/custom_vitb16 \
  --batch-size 8 --num-workers 4 --epochs 10
```

四卡 ConvNeXt 可以换为 `--model convnext_base` 或 `convnext_tiny`，并使用不同输出目录。
先加 `--smoke-test --steps 20` 验证服务器环境和显存，然后用新的输出目录开启正式训练。
四卡分布式本身需要在服务器实测；本地单卡验证不等于已经验证四卡通信。

默认全局 batch = 4 × 8 = 32。这个 batch 较小，DINO/Sinkhorn 分配的训练效果仍需实际验证。
`iterations/epoch = ceil(图像数 / 全局 batch)`，80 万张时约 25,000 步/epoch，10 epoch 约 250,000 步。
底层使用无限采样器，因此这里的 epoch 是近似数据遍历计数，不保证每个区间恰好遍历所有图像。
先测真实吞吐再决定步数和 epoch 数；不要把官方固定 1,250 步/epoch 当成自定义数据集的完整一轮。

默认设置来自 `dinov3/configs/train/custom.yaml`：

| 参数 | 初始值/含义 |
| --- | --- |
| 精度 | bf16 参数计算、fp32 梯度规约，无 FP8 |
| 激活 checkpoint | 开启，compile 默认关闭 |
| global / local 尺寸 | 224 / 96，正式训练 4 个 local crop |
| backbone 基础峰值学习率 | 1e-5；ViT 使用层间衰减，patch embedding 额外乘 0.2 |
| 新投影头学习率倍率 | 10，即峰值 1e-4 |
| 学习率 / 温度预热 | 0.1 epoch（折算为整数步） |
| prototype 数 | DINO/iBOT 各 4096，降低小算力实验成本 |
| 保存周期 | 5000 步，保留最近 2 个完整训练 checkpoint |
| 增强 | 温和颜色扰动，不做灰度化或 solarization；仍保留裁剪、翻转和模糊 |

命令末尾可覆盖配置，例如：

```text
crops.global_crops_size=256 crops.local_crops_number=6 optim.lr=0.00002
```

ViT 的 crop 尺寸需为 16 的倍数，ConvNeXt 为 32 的倍数。
batch 使用 `--batch-size` 修改，使 epoch 长度同步计算。调整其他训练参数时仍需自己评估配方一致性。
当前默认没有启用 Gram anchoring；用于检测/分割时可以作为后续对照实验，但需另外配置 teacher 初始化和显存预算。

## 保存、恢复及下游加载

```text
output/
├── config.yaml                 # 本次训练配置
├── training_metrics.json       # JSON Lines；周期记录训练指标
├── ckpt/<iteration>/           # 分布式完整训练状态，包含 optimizer；用于恢复训练
└── eval/
    ├── training_<iteration>/   # 周期 teacher 导出
    └── final/
        ├── teacher_checkpoint.pth  # backbone + SSL heads；不包含 optimizer
        └── backbone.pth            # 独立 EMA backbone state_dict
```

重复相同命令和输出目录会恢复最近的完整 checkpoint。
保持模型、数据、batch 和训练计划一致；`--steps`/`--epochs` 改变会重算 schedule，并不是严格延续原计划。
若要开始新实验，换一个输出目录。入口拒绝在已有 checkpoint 的目录里用 `--no-resume` 覆盖旧训练，
也检查恢复时的模型结构及数据路径是否变化。
`ckpt` 用于恢复训练；`backbone.pth` 用于提取特征或下游任务，两者不能互换。

从导出 ViT-B 提取全局和局部特征：

```python
import torch
from dinov3.hub.backbones import dinov3_vitb16

model = dinov3_vitb16(pretrained=False)
state = torch.load('/path/to/output/eval/final/backbone.pth', map_location='cpu', weights_only=True)
model.load_state_dict(state, strict=True)
model = model.cuda().eval()

# x: [B, 3, H, W]，RGB，按训练使用的 ImageNet mean/std 归一化。
with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
    features = model.forward_features(x)
    global_features = features['x_norm_clstoken']
    patch_features = features['x_norm_patchtokens']
    intermediate_features = model.get_intermediate_layers(x, n=4, reshape=True)
```

ConvNeXt 可以将构造器换为 `dinov3_convnext_tiny` 等，严格加载对应导出的权重。
分类可用全局特征；检测/分割需要适配任务 head 和空间特征接口，不是仅导出 backbone 就完成下游模型。
ConvNeXt 的多阶段特征通道数不同，ViT 的中间层通常保持相同空间分辨率，下游 neck 需相应设计。

## 后续轻量化蒸馏

建议分两阶段：

1. 用自己的无标签数据集适配 ViT-B（或者经验证能训练的 ViT-L），保留原始官方权重作为对照。
2. 冻结经过评估的领域适配 teacher，在同一训练数据集上训练轻量 student，如 ViT-S、ConvNeXt-Tiny。

第一阶段 teacher 随 student 做 EMA 更新；第二阶段是固定 teacher 向另一架构蒸馏，机制不同。
考虑同时蒸馏全局语义和对齐后的空间特征；只蒸馏 CLS/全局向量不足以直接约束分割需要的局部特征。
ViT 的 stride 16 与 ConvNeXt 最后一层 stride 32 不同，通道数也可能不同，需投影和空间对齐。
第二阶段训练尚未在本入口实现；官方 distillation/multidistillation 代码可以作为基础，不能把一个
`backbone.pth` 直接当作包含 SSL heads 的固定 teacher checkpoint。

最终分别比较官方 backbone、领域适配 backbone、蒸馏 student 在独立下游评估集上的表现；
对于内镜数据，应按患者或检查独立划分。
分类、检测、分割都需要各自的评估，不能用 SSL loss 是否下降替代。

## 自动检查

```bash
python -m unittest discover -s tests -v
git diff --check
```

单元检查覆盖无标签目录/清单、中文及等号路径、严格权重加载、模型不匹配拒绝、
四卡 epoch 长度计算、分数 epoch 预热和 ConvNeXt crop/mask 配置。
GPU smoke-test 另行覆盖真实训练、FSDP 保存、恢复和导出权重的重新加载。

本机验证使用上述目录的 177 张真实图像，未读取 COCO 标签：

| 模型 | 单卡 batch / optimizer steps | 导出后全局特征 | 导出后 patch 特征（224 输入） |
| --- | --- | --- | --- |
| ViT-B/16 | 2 / 5 | `[2, 768]` | `[2, 196, 768]` |
| ConvNeXt-Tiny | 8 / 5 | `[2, 768]`（推理检查 batch=2） | `[2, 49, 768]` |

两者官方权重初始化、导出权重重新加载均严格匹配全部键，导出的 EMA 参数相比官方权重已有更新。
ViT 提取了 4 个 `[2, 768, 14, 14]` 中间层；ConvNeXt 提取了
`[2, 96, 56, 56]`、`[2, 192, 28, 28]`、`[2, 384, 14, 14]`、`[2, 768, 7, 7]` 四阶段特征。
完整训练状态从第 4 步前的 checkpoint 恢复后，继续完成了第 5 步。

本机验证日志与导出文件位于 `/tmp/dinov3_d2_vitb_smoke`、`/tmp/dinov3_d2_convnext_smoke`，
以及对应的 `/tmp/dinov3_d2_*_resume` 恢复测试目录。
`/tmp` 是临时位置；需要长期保存时使用持久存储的输出目录。

补充单卡容量验证：ViT-L/16 和 ConvNeXt-Large 均以 batch=2、224 global crop、96 local crop、
4 个 local crop、bf16、激活 checkpoint、4096 prototypes 完成 3 个 optimizer step，loss/梯度为有限值。
PyTorch 峰值 allocated 显存分别约 6.13 GiB 和 4.48 GiB（不含 CUDA context、缓存及桌面占用）。
这次容量验证关闭 checkpoint 保存和导出，未测试最大 batch 或长期训练效果。
日志分别为 `/tmp/dinov3_d2_vitl_capacity.log`、`/tmp/dinov3_d2_convnext_large_capacity.log`。
当前 custom 入口支持的最大 ViT 是 ViT-H+/16；此前的单卡容量验证只覆盖到 ViT-L，
新的 ViT-S+/16 和 ViT-H+/16 验证结果如下。

ViT-S+/16、ViT-H+/16 均已使用本机官方 LVD 权重通过严格加载和 CPU 前向检查。
单元检查还对照了官方 Hub 构造器的全部 state_dict 键名、形状及归一化等设置。
两者随后均在单卡 3090 上以 batch=2、224 global crop、96 local crop、4 个 local crop、
bf16、激活 checkpoint、4096 prototypes 完成 3 步 DINO+iBOT 训练，loss/梯度为有限值：

| 型号 | `--model` | PyTorch 峰值 allocated 显存 |
| --- | --- | --- |
| ViT-S+/16 | `vits16plus` | 约 0.90 GiB |
| ViT-H+/16 | `vith16plus` | 约 16.63 GiB |

这些容量检查关闭了保存和导出，不包含 CUDA context、缓存等显存，也没有验证最大 batch、长期训练或四卡通信。
新测试日志为 `/tmp/dinov3_custom_vits16plus_smoke.log` 和 `/tmp/dinov3_custom_vith16plus_smoke.log`。
