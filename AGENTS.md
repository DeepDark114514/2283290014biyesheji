# AGENTS.md — 基于残差U-Net的视频压缩伪影去除

> **项目类型**：本科毕业设计（算法研究型）  
> **课题**：基于残差U-Net的视频压缩伪影去除算法研究  
> **核心任务**：A/B对比实验，探索盲QP场景下轻量退化感知机制的可行性边界  
> **作者学号**：202283290014

---

## 1. 项目核心背景

### 1.1 研究问题
H.265/HEVC有损压缩引入块效应、振铃效应、模糊等伪影。现有深度学习方法大多假设已知QP（量化参数），需为每个QP单独训练模型。本课题聚焦**盲QP场景**（QP未知或动态变化），探索在标准残差U-Net中嵌入轻量退化感知模块的有效性与能力边界。

### 1.2 A/B方案定义

| 方案 | 模型名 | 核心特征 | 参数量 | 角色 |
|:---|:---|:---|:---|:---|
| **A** | `PureResUNet` | 纯Pre-activation残差U-Net，4级Encoder-Decoder，全局残差 | 12.16M | 基线 |
| **B** | `DegFiLMResUNet` | A + `DegEstimator`(退化估计器) + `FiLM`(特征线性调制) | 12.44M (+2.34%) | 改进 |

**B方案的关键设计哲学**："极简条件化"——用极少的额外参数（~284K）验证特征级退化感知的可行性，同时诚实地画出其能力边界（高QP边际收益递减）。

### 1.3 核心结论（实验已验证）
- B方案在中低QP（22/27/32）相对A有**6%~10%增益提升**，验证"轻手轻脚"自适应策略有效
- QP37时边际收益降至**+2.4%**，QP42时微降**-1.3%**，揭示轻量FiLM的能力边界
- 混合QP训练（22/32/42）对未见过QP（27/37）具有泛化能力

---

## 2. 项目结构

```
├── models/                 # 模型定义
│   ├── base_unet.py        # ResBlock + BaseUNet（4级U-Net骨架）
│   ├── pure_resunet.py     # A方案：继承BaseUNet，无修改
│   ├── deg_film_blocks.py  # DegEstimator + FiLM + FiLMResBlock
│   └── degfilm_resunet.py  # B方案：替换Bottleneck和部分Decoder为FiLM版本
├── datasets/
│   ├── mfqev2_dataset.py   # MFQEv2格式YUV数据集，支持单QP
│   ├── multi_qp_dataset.py # 多QP训练用，每batch随机QP
│   ├── __init__.py         # 统一导出MFQEv2Dataset、build_dataloader、tile_predict等接口
│   ├── yuv_io.py           # YUV420原始文件读写
│   └── inference_utils.py  # tile_predict（大图分块推理）
├── losses/
│   └── l1_ssim_loss.py     # L1 + SSIM组合损失（1:1权重）
├── utils/
│   ├── metrics.py          # PSNR/SSIM计算
│   ├── early_stopping.py   # 早停（保存最佳模型+optimizer状态）
│   └── process_priority.py # Windows进程优先级设置
├── scripts/                # 评估、可视化、辅助脚本
│   ├── cross_qp_eval.py    # 跨QP评估主脚本（生成JSON/CSV/曲线图）
│   ├── compare_ab_cross_qp.py  # A/B跨QP对比图表
│   ├── draw_training_curves.py # 训练过程对比图
│   ├── generate_thesis_visualization.py  # 论文图5-3/5-4（全局+局部对比）
│   └── ...（benchmark、smoke test、acceptance test等）
├── docs/                   # 论文相关
│   ├── 论文目录大纲_最终版.md  # 完整大纲（~26,000字规划）
│   ├── chapters/           # 各章节Markdown草稿
│   │   ├── 摘要.md
│   │   ├── 第一章_绪论.md
│   │   ├── 第二章_相关工作.md
│   │   ├── 第三章_基于残差U-Net的压缩视频伪影去除方法.md
│   │   ├── 第四章_实验设置与数据集构建.md
│   │   ├── 第五章_实验结果与分析.md
│   │   ├── 第六章_结论与展望.md
│   │   └── 致谢.md
│   ├── figures/            # 论文插图（命名规范：图X-X 描述.png）
│   ├── tables/             # 论文表格源数据
│   └── equations/          # 公式相关素材
├── MFQEv2_processed/       # 数据集（YUV 4:2:0格式）
│   ├── gt/                 # 原始无损YUV
│   └── compressed/         # 压缩后YUV（命名：xxx_qp{QP}.yuv）
├── logs/                   # 训练日志
│   ├── A_20260517_105612/  # A方案训练记录
│   └── B_20260519_121523/  # B方案训练记录
├── result/                 # 实验结果输出
│   ├── cross_qp/           # 跨QP评估结果（JSON/CSV/图）
│   ├── baseline/           # 输入LQ基线指标
│   ├── training_curves/    # 训练曲线图
│   └── ab_visualization_v2/ # A/B可视化对比图
├── config.py               # 全局配置字典CONFIG
├── train.py                # 统一训练入口（支持-m A/B）
└── README.md               # 项目说明（含ffmpeg压缩命令示例）
```

---

## 3. 代码规范与风格

### 3.1 编码风格
- **Python 3.10**，PyTorch深度学习框架
- 使用`pathlib.Path`处理路径，但配置中保留字符串路径以兼容JSON序列化
- 注释多为中文，包含设计决策说明（如"为什么不用FP16""为什么Decoder第4层不加FiLM"）
- 模型类继承清晰：B方案继承`BaseUNet`，只覆盖`__init__`和`forward`

### 3.2 关键工程约束（**必须遵守**）

| 约束 | 原因 | 代码体现 |
|:---|:---|:---|
| **FP32训练** | Pre-activation ResBlock在FP16下数值漂移，loss变NaN | `CONFIG['amp'] = False` |
| **Windows pin_memory=False** | Windows多进程机制不同，DataLoader易死锁 | `CONFIG['pin_memory'] = False` |
| **cuDNN benchmark=True** | 输入尺寸固定（256×256 patch），自动选最快卷积算法 | `torch.backends.cudnn.benchmark = True` |
| **eval时num_workers=0** | Windows下eval开多进程易崩 | `build_dataloader`中`nw = 0 if not is_train else cfg['num_workers']` |
| **梯度裁剪1.0** | 防梯度爆炸 | `clip_grad_norm=1.0` |
| **pred_clamp=True** | 模型输出先clamp到[0,1]再算loss | `torch.clamp(pred, 0.0, 1.0)` |

### 3.3 模型相关常量
- `base_ch = 32`（12M参数，64会涨到48M，5-15M范围合适）
- Encoder/Decoder均为4级，Bottleneck 2个ResBlock
- 下采样4次 → 特征图尺寸需被16整除 → `_pad_to_multiple(x, 16)`
- FiLM初始化为零（`nn.init.zeros_`），保证训练初期恒等映射
- `(1+gamma)*out + beta`而非`gamma*out + beta`，gamma=0时为恒等

---

## 4. 训练与评估流程

### 4.1 训练命令
```bash
# A方案
python train.py -m A --epochs 100

# B方案
python train.py -m B --epochs 100
```
训练集相同：均为混合QP22/32/42。区别仅在于B的DataLoader会返回qp_tensor，且B模型内部有DegEstimator分支。

### 4.2 评估命令
```bash
# 跨QP泛化评估（核心实验）
python scripts/cross_qp_eval.py \
    --model_path logs/B_20260519_121523/best_model.pth \
    --model_type B \
    --qp_list 22 27 32 37 42
```

### 4.3 关键配置（config.py）
```python
CONFIG = {
    'model_type': 'A',      # 或 'B'
    'base_ch': 32,
    'loss_type': 'L1_SSIM',
    'l1_weight': 1.0,
    'ssim_weight': 1.0,
    'optimizer': 'Adam',
    'lr': 1e-4,
    'scheduler': 'StepLR',
    'step_size': 30,
    'gamma': 0.1,
    'epochs': 100,
    'batch_size': 32,
    'patch_size': 256,
    'num_workers': 6,
    'pin_memory': False,
    'early_stop': True,
    'early_stop_patience': 6,
    'val_interval': 5,
    'clip_grad_norm': 1.0,
    'amp': False,
    'qp_list': [22, 32, 42],    # 训练QP
}
```

---

## 5. 论文写作规范

### 5.1 图表命名规范
- 论文图：`图X-X 描述.png`（如`图3-1 PureResUNet网络架构.png`）
- 存放在 `docs/figures/`
- 可视化脚本生成的对比图命名：`图5-3 {seq_name}_QP{qp}_frame{idx:04d}_global.png`

### 5.2 核心数据（**已确定，勿随意改动**）

**A方案跨QP性能（Test集）**：
| QP | Baseline PSNR | Model PSNR | ΔPSNR | Baseline SSIM | Model SSIM | ΔSSIM |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 22 | 36.62 | 36.79 | +0.17 | 0.9346 | 0.9384 | +0.0038 |
| 27 | 34.22 | 34.47 | +0.25 | 0.9055 | 0.9113 | +0.0059 |
| 32 | 32.04 | 32.39 | +0.35 | 0.8715 | 0.8819 | +0.0104 |
| 37 | 29.38 | 29.74 | +0.36 | 0.8123 | 0.8266 | +0.0144 |
| 42 | 27.49 | 27.87 | +0.38 | 0.7625 | 0.7825 | +0.0201 |

**B方案相对A方案的改进幅度**：
- QP22: PSNR +0.018dB（相对提升~10%）
- QP27: PSNR +0.016dB（相对提升~6%）
- QP32: PSNR +0.027dB（相对提升~8%）
- QP37: PSNR +0.009dB（相对提升~2.4%）
- QP42: PSNR -0.005dB（相对下降~1.3%）

**训练最佳指标**：
- A最佳Val PSNR: **32.7917 dB** @ Epoch 45
- B最佳Val PSNR: **32.9297 dB** @ Epoch 45

### 5.3 论文章节字数规划
- 摘要：~800字
- 第一章（绪论）：~3,000字
- 第二章（相关工作）：~4,000字
- 第三章（方法）：~5,000字
- 第四章（实验设置）：~3,500字
- 第五章（结果分析）：~6,000字
- 第六章（结论展望）：~1,500字
- 附录：~2,000字
- **正文总计目标：~23,500字**

---

## 6. 用户偏好与注意事项

### 6.1 技术偏好
- 追求**轻量、可部署**的解决方案，不追求SOTA刷点
- 重视**A/B控制变量**的公平性：A和B训练集完全相同，仅模型结构差异
- 喜欢用**表格**清晰呈现结构配置和对比结果
- 论文强调**诚实划定能力边界**，不夸大B方案效果

### 6.2 工程习惯
- 用ffmpeg+libx265而非HM参考软件（效率优先，且认为深度学习对编码器细节不敏感）
- Windows开发环境，因此处理了多个Windows特有问题（pin_memory、num_workers、进程优先级等）
- 训练日志和配置均自动保存到`logs/实验名/`目录，便于复现
- 喜欢在代码注释中写明"为什么这样设计"（设计决策追溯）

### 6.3 常见陷阱（新Agent必看）
1. **不要开启AMP/FP16**：会导致Pre-activation ResBlock训练不稳定
2. **不要修改A/B训练集差异**：两方案必须用完全相同的混合QP训练数据
3. **eval时别改batch_size**：整帧推理batch_size=1，大图自动tile
4. **720p推理B方案慢87%→+2.8%**：旧数据155.82ms（+87%）为WDDM模式异常值，已更新为稳定测量值61.19/62.92ms（+2.8%）
5. **不要删除`docs/figures/`中的中文命名文件**：论文引用路径已固定
6. **训练集Cactus_189 ≠ 测试集Cactus_500**：经逐帧像素级验证（PSNR≈10dB），两者为完全不同的视频内容。训练集中的Cactus来源于MFQEv2.0原生训练集构成，不存在数据泄露。论文已在第四章主动说明此命名重合问题

### 6.4 后续Agent协作建议
- 修改代码后尽量跑`scripts/smoke_test.py`或`scripts/acceptance_test.py`验证
- 涉及论文数据（PSNR/SSIM数值）的修改需与`result/cross_qp/cross_qp_results.json`交叉核对
- 新增图表建议遵循现有配色：A方案用`#2E86AB`（蓝），B方案用`#E94F37`（红），Baseline用`#888888`（灰）
- 论文章节文件使用UTF-8编码，Markdown格式

---

## 7. 关键参考文献方向

论文引用的核心文献领域（已在`docs/论文目录大纲_最终版.md`中规划）：
- 非盲去伪影：ARCNN、DnCNN、VRCNN、MFQE/MFQEv2
- 盲去伪影：QP分类+通道注意力、输入级QP编码（DREFNet）、重型条件化、Prompt Learning
- 网络结构：U-Net、ResNet、Pre-activation ResBlock
- 条件化机制：FiLM（Feature-wise Linear Modulation）
- 轻量化：Scalable Residual Laplacian Network等

---

## 8. 快速检查清单

在修改以下任何内容时，请同步检查关联文件：

- [ ] 修改模型结构 → 检查`train.py`的`build_model`、`count_parameters`输出、论文第三章
- [ ] 修改损失函数 → 检查`train.py`的`loss_fn`构建、论文3.5节
- [ ] 修改数据加载 → 检查`config.py`的`qp_list`、`train.py`的`build_dataloader_for_split`
- [ ] 修改训练配置 → 检查`config.py`、论文4.2节表4-3
- [ ] 修改评估方式 → 检查`scripts/cross_qp_eval.py`、论文第五章数据
- [ ] 新增/修改图表 → 检查`docs/figures/`命名、论文对应章节引用
- [ ] 修改论文数据 → 必须与`result/cross_qp/cross_qp_results.json`一致
