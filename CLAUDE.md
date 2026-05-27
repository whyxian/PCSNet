# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中操作时提供指引。

## 环境配置与依赖

- PyTorch 1.12, torchvision 0.13, numpy, scipy, matplotlib, tqdm
- opencv-python, scikit-learn, scikit-image, einops
- 数据集放置于 `./datasets/` 目录下，子目录以类别名称命名
- 主干网络：torch hub 预训练的 Wide ResNet-50-2（`cnn/resnet.py`）

## 运行命令

```bash
# 训练并评估单个类别
python main.py --class_name class_name

# 可选参数
python main.py --data_path ./datasets --save_path ./result --size 224
```

## 架构

**PCSNet** — 原型学习引导的上下文感知分割网络（Prototypical Learning Guided Context-Aware Segmentation Network），用于少样本异常检测。

### 流水线

1. **主干网络** ([cnn/resnet.py](cnn/resnet.py))：在 ImageNet 上预训练的 Wide ResNet-50-2。`forward()` 从 layer1/layer2/layer3 返回多尺度特征 `[x_1, x_2, x_3]`（应用 leaky relu）。训练时作为冻结的特征提取器使用。

2. **PFA 子网络** ([utils/adaptor.py](utils/adaptor.py))：`adaptor` 类封装了一个 `Descriptor` 模块，通过 CoordConv 层和 1x1 卷积投影将多尺度特征融合为统一特征图（旋转对象为 256 通道，其他为 896 通道）。存储从训练数据计算得到的一组正常原型（质心）。训练时应用软边界损失：将正常特征拉入超球体（半径 `r`）内，将异常特征推至外部（`r+alpha`）。评分基于到 k 近邻原型的距离。

3. **CAS 子网络** ([casnet.py](casnet.py))：上下文感知分割模块，采用多尺度金字塔池化架构（bins: 60, 30, 15, 8）。接收 PFA 特征 + 评分图作为输入，通过多尺度相关性和特征增强（受 PFENet 特征增强模块启发）在像素级别细化异常定位。

4. **自监督训练** ([self_sup_tasks.py](self_sup_tasks.py))：通过 OpenCV 无缝克隆（`NORMAL_CLONE`/`MIXED_CLONE`）在图像间克隆补丁来合成异常图像，支持随机补丁放置、缩放和可选的背景感知跳过。标签为二值补丁掩码。

### 训练循环 ([main.py](main.py))

- 对每个类别，通过混合补丁创建正常图像和合成异常的混合
- 优化器：Adam（adaptor 学习率 0.001，CAS 网络学习率 0.0001）
- 损失函数：`L_NFC + L_AFS + L_PDC*50 + L_SEG*40`
  - `L_NFC`：正常特征紧凑性（将特征拉入半径内）
  - `L_AFS`：异常特征分离（将特征推出半径外）
  - `L_PDC`：像素级差异分类（异常评分上的交叉熵）
  - `L_SEG`：预测异常掩码上的分割 MSE
- 训练轮数：50；每 10 轮评估一次
- 指标：图像级 AUROC、像素级 AUROC、像素级 AUPRO

### 每类别配置 ([main.py](main.py) 全局字典)

- `WIDTH_BOUNDS_PCT`、`INTENSITY_LOGISTIC_PARAMS`、`NUM_PATCHES`、`MIN_OBJECT_PCT`、`MIN_OVERLAP_PCT` — 每类合成异常生成的超参数
- `BACKGROUND`：每类前景/背景掩码的背景阈值
- `Descriptor` 中对旋转对称类别的特殊处理（`utils/adaptor.py` 中 `ROTATION_OBJECTS`） — 使用 CoordConv 且通道数为 256/512/1024 而非 128/256/512

### 关键文件

| 文件 | 用途 |
|------|------|
| [main.py](main.py) | 入口：训练编排、评估、每类配置 |
| [cnn/resnet.py](cnn/resnet.py) | Wide ResNet-50-2 主干网络（多尺度特征提取） |
| [utils/adaptor.py](utils/adaptor.py) | PFA 子网络：基于原型的特征自适应和评分 |
| [casnet.py](casnet.py) | CAS 子网络：上下文感知金字塔分割 |
| [self_sup_tasks.py](self_sup_tasks.py) | 通过补丁克隆/混合合成异常图像 |
| [datasets/mvtec_train.py](datasets/mvtec_train.py) | 自监督训练数据集（合成异常即时生成） |
| [datasets/mvtec_test.py](datasets/mvtec_test.py) | 测试数据集 |
| [loss.py](loss.py) | FocalLoss 和 SSIM 损失函数 |
| [utils/metric.py](utils/metric.py) | AUROC、像素 AUROC、PRO AUC 评估 |
| [resnet.py](resnet.py) | （遗留）带有深度基础茎的自定义 ResNet |

---

## 论文总结

**标题**: Prototypical Learning Guided Context-Aware Segmentation Network for Few-Shot Anomaly Detection (PCSNet)

**作者**: Yuxin Jiang, Yunkang Cao, Weiming Shen (华中科技大学)

**发表**: IEEE Transactions on Neural Networks and Learning Systems (TNNLS), 2024

### 问题背景

少样本异常检测（FSAD）旨在利用目标类别中极少量正常样本识别异常。现有方法大多依赖预训练特征表示，但**预训练表示与目标 FSAD 场景之间存在固有的域间隙**（domain gap）：如图 1 所示，仅使用预训练模块作为特征提取器，会导致正常样本的嵌入空间分散，测试阶段正常与异常特征常交织在一起。

### PCSNet 总体架构

PCSNet 包含两个子网络：

1. **PFA（Prototypical Feature Adaptation）子网络** — 特征自适应
2. **CAS（Context-Aware Segmentation）子网络** — 上下文感知分割

自监督训练阶段通过 OpenCV 无缝克隆在正常图像上合成异常，生成带伪标签的训练数据。

### PFA 子网络（[utils/adaptor.py](utils/adaptor.py)）

PFA 利用原型学习策略将预训练特征适配到目标域，创建高判别性的嵌入空间。核心组件：

- **Descriptor 模块**：将 Wide ResNet-50-2 的三个层输出的多尺度特征 `[x_1, x_2, x_3]`（通道数 256/512/1024 或对旋转对称物体的 128/256/512）通过 CoordConv + 1x1 卷积投影 + 上采样融合为统一分辨率（64x64）的特征图
- **原型集 C**：训练时从所有正常图像中提取特征块并计算逐像素均值，作为正常原型质心。评分时计算每个特征块到其 k 近邻原型的距离作为异常分数

PFA 使用三种损失函数（参见论文 Fig. 2, Fig. 3）：

1. **NFC Loss（正常特征紧致性损失）** — 将正常特征块拉入以原型为中心的半径为 r 的超球体内：
   $$L_{NFC} = \frac{1}{J \times K} \sum_{j=1}^{J} \sum_{k=1}^{K} \max\left(0, D(F_j^n, C_j^k) - r^2\right)$$

2. **AFS Loss（异常特征分离损失）** — 将合成异常特征推出到半径 r+α 之外，压缩正常特征空间：
   $$L_{AFS} = \frac{1}{J \times K} \sum_{j=1}^{J} \sum_{k=1}^{K} \max\left(0, (r+\alpha)^2 - D(F_j^a, C_j^k)\right)$$

3. **PDC Loss（像素级差异分类损失）** — 针对困难样本（subtle anomalies），计算相似度图后选取最异常的特征块，通过交叉熵损失分类，放大异常与正常之间的局部像素差异。该损失作用于整个图像的异常分数向量而非全局特征，使得网络能够关注细微差异。

### CAS 子网络（[casnet.py](casnet.py)）

CAS 是一个类似 FPN 的分割网络，用于生成像素级异常定位图：

- **输入**：PFA 提取的特征图 `F`（896 通道）+ 相似度图 `S^k(F)`（200 通道，通过特征与 k 个最近原型间的距离计算）
- **多尺度金字塔池化**：bins = [60, 30, 15, 8]，每个尺度上拼接特征和相似度图后，通过 1x1 卷积降维
- **特征增强**：对每个金字塔层逐层处理，当前层融合上一层特征，加残差连接
- **特征融合**：将所有尺度的特征拼接后经 `res1` + `res2`（1x1 + 3x3 卷积+残差）精炼，最终由 `cls` 卷积（→1 通道）输出分割结果
- **设计灵感**：受 PFENet 特征增强模块启发，通过多尺度上下文相关性实现细化定位

### 数据增强与自监督训练（[self_sup_tasks.py](self_sup_tasks.py)）

采用基于补丁克隆的 NSA（Normal-to-Seamless-Anomaly）策略：

- 通过 OpenCV `seamlessClone`（NORMAL_CLONE / MIXED_CLONE）从同一图像（`same=True`）或不同图像中裁剪补丁并泊松融合到目标位置
- 支持随机补丁尺寸（根据 `WIDTH_BOUNDS_PCT` 按类别配置）、随机位置、缩放
- 背景感知跳过（`skip_background`）：对于特定类别（如 `bottle`、`capsule`），仅在前景物体区域内放置补丁
- 生成的标签为二值补丁掩码（经过中值滤波去噪）
- 训练时每个 batch：正常图像（label=0）+ 合成异常图像（label=1）+ 混合图像（部分正常部分异常，label=2）

### 训练流程与配置（[main.py](main.py)）

- **每类别配置**：为 MVTec AD 和 MPDD 的每个类别单独配置补丁尺寸（`WIDTH_BOUNDS_PCT`）、补丁数量（`NUM_PATCHES`）、前景占比（`MIN_OBJECT_PCT`）、强度逻辑参数等
- **优化器**：Adam；adaptor 学习率 0.001，CAS 学习率 0.0001
- **总损失**：$L_{total} = L_{NFC} + L_{AFS} + L_{PDC} \times 50 + L_{SEG} \times 40$
- **训练**：50 个 epoch，每 10 epoch 评估一次
- **评估**：image-level AUROC、pixel-level AUROC、pixel-level AUPRO

### 实验成果（论文核心数据）

| 数据集 | 场景 | 图像 AUROC（8-shot） |
|--------|------|-------------------|
| MVTec AD | 15 类 | 94.9% |
| MPDD（金属缺陷检测） | 6 类 | 80.2% |
| APPD（汽车塑料件检测，真实应用） | 4 类异常 | AUPRO 92.8% |

- 在 MVTec AD 上，2/4/8-shot 场景分别比 RegAD 提升 4.7%/3.9%/3.7%
- t-SNE 可视化验证：PFA 适配后的特征相比原始预训练特征具有更好的类内紧致性和类间可分性

### 关键消融结论

- PDC 损失单独带来约 12-16% 提升（MVTec AD）
- NFC+AFS 进一步带来 6-10% 提升
- CAS 子网络贡献约 2-4% 提升
- NSA 合成异常方法优于 CutPaste、FPI、PII 等方法
- 引入少量真实异常样本（1-4 个）可显著提升性能

### 论文方法 → 代码映射

| 论文组件 | 代码文件 | 对应类/函数 |
|---------|----------|------------|
| PFA 子网络 | [utils/adaptor.py](utils/adaptor.py) | `class adaptor`, `class Descriptor` |
| PFA: NFC loss | [utils/adaptor.py](utils/adaptor.py):53-57 | `_soft_boundary()` 中 `label==0` 分支 |
| PFA: AFS loss | [utils/adaptor.py](utils/adaptor.py):59-66 | `_soft_boundary()` 中 `label==1` 分支 |
| PFA: PDC loss | [main.py](main.py):350 | `tr_entropy_loss_func(score_1, target)` |
| PFA: 原型集初始化 | [utils/adaptor.py](utils/adaptor.py):76-82 | `_init_centroid()` |
| PFA: k-NN 评分 | [utils/adaptor.py](utils/adaptor.py):41-43 | `dist.topk(n_neighbors, largest=False)` |
| CAS 子网络 | [casnet.py](casnet.py) | `class Model` |
| CAS: 多尺度金字塔 | [casnet.py](casnet.py):50-51 | `pyramid_bins = [60, 30, 15, 8]` |
| CAS: 特征增强（PFENet 启发） | [casnet.py](casnet.py):59-79 | `corr_conv`, `beta_conv`, `alpha_conv` 模块 |
| CAS: 分割损失 L_SEG | [main.py](main.py):349 | `MSE_loss(out, mask)` |
| 自监督数据增强 | [self_sup_tasks.py](self_sup_tasks.py) | `patch_ex()` / `_patch_ex()` |
| 合成异常生成（NSA） | [self_sup_tasks.py](self_sup_tasks.py):206-227 | `cv2.seamlessClone`（Poisson blending） |
| 训练循环 | [main.py](main.py):326-353 | `run()` 中 epoch 循环 |
| 骨干网络（Wide ResNet-50-2） | [cnn/resnet.py](cnn/resnet.py) | `wide_resnet50_2()` |
