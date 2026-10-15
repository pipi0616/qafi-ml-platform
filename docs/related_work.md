# 第二阶段 F · 第 0 步：文献核对

> 2026-10-15。目的：在动手写 14a 之前，确认"把一个分数拆成折叠能和功能能"这件事还没有被别人做完。
> 结论：**有一篇很接近的新工作（DETANGO，2026 年 2 月预印本），但它和我们的做法有三处本质区别。角度要调整，不用放弃。**

---

## 1 · 最接近的工作：DETANGO（2026 预印本）

Deconvolving mutation effects on protein stability and function with disentangled protein language models
([bioRxiv](https://www.biorxiv.org/content/10.64898/2026.02.03.703560v1.full) · [PMC12889624](https://pmc.ncbi.nlm.nih.gov/articles/PMC12889624/))

它做的事和我们想做的听起来一样：把突变效应拆成"稳定性"和"功能"两部分。

**它怎么做的**：拿 ESM-1v 的残基向量（1280 维），用一个小网络投影出"稳定性部分"，**功能部分 = 原向量 − 稳定性部分**（直接相减）。两部分各自预测一个值，两者之和训练成去还原 ESM-1v 的 log-odds。

**它拿什么当监督**：
- 主监督是 **ESM-1v 自己的 log-odds**（模型在学着复现语言模型）；
- 稳定性那一路用 **FoldX 预测的 ΔΔG**（也可换成 SPURS 的预测或实测的蛋白丰度）。

**它验证了什么**：
- "稳定但没活性"分类（11 个蛋白，对照 Cagiada 2023）；
- Human Domainome 1 的 408 个人类蛋白上找功能位点；
- BioLiP2（24,252 个蛋白）的配体位点、PocketMiner 的隐藏口袋；
- KRAS 的变构图谱（6 个结合伙伴）、17 个变构蛋白。

### 对我们的影响（要正视）

**原来的 14c 基本被它做掉了。** 功能位点富集、BioLiP、KRAS 变构口袋，这几条它都做了，而且规模比我们大。14c 不能再当成"新发现"来写，只能当成"和 DETANGO 的对照"。

### 但它有三处本质缺陷，正是我们的位置

| | DETANGO | 我们要做的 |
|---|---|---|
| **1. 一个蛋白一个模型** | 论文原文："Models are trained per protein"，交叉验证也是**在同一个蛋白内部把残基分成训练和验证**。没有留出蛋白、更没有留出家族的切分。换句话说，**它不能预测一个全新的蛋白**，必须先有这个蛋白的数据 | 一个模型，跨蛋白。考试规则不变：按序列相似度留出整个家族，测试集只开一次 |
| **2. 不是物理模型** | 方法里写了玻尔兹曼关系 s = −ΔΔG/RT，然后**换成了一个学出来的 MLP**。没有"折好的比例"这一层，也没有"折好 × 还能工作"。输出是 log-odds 尺度的分数，不是 kcal/mol | 真的用物理读数层：测蛋白量的实验只看折叠，测活性的实验看折叠 × 功能。输出带单位，可以和 ddPCA 实测的能量直接对账 |
| **3. 在复现语言模型，不是在拟合实验** | 主监督是 ESM-1v 的 log-odds。稳定性那一路默认用 **FoldX 的预测**（预测去学预测）。作者自己承认："the separation between stability- and function-specific representations is operational rather than strictly guaranteed"（这个分离是操作上的，不保证真的分开了） | 直接拿实验数据监督：测蛋白量的实验钉折叠能，测活性的实验钉功能能。分离是由**实验类型**强制的，不是靠网络结构暗示的 |

其他限制：只支持 ESM-1v；超过 1,022 个氨基酸的蛋白直接排除（我们 09c 跑的是全蛋白组，含 TTN）；未经同行评审。

---

## 2 · 物理建模那一支：MoCHI / ddPCA（Lehner 实验室）

- **MoCHI**（Faure & Lehner, *Genome Biology* 2024, [doi](https://doi.org/10.1186/s13059-024-03444-y)，[lehner-lab/MoCHI](https://github.com/lehner-lab/MoCHI)）：用 PyTorch 把生物物理模型拟合到 DMS 数据上，输出自由能变化和能量耦合。
- **The genetic architecture of protein stability**（*Nature* 2024, [s41586-024-07966-0](https://www.nature.com/articles/s41586-024-07966-0)）：两态平衡 + 全局非线性，从 AbundancePCA 数据推折叠自由能。
- **KRAS**（Weng et al., *Nature* 2024, [PMC10866706](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10866706/)）：2.6 万个突变，推出 2.2 万个折叠能和结合能。

**这正是我们要用的物理层**，但它们是**拟合**不是**预测**：每个蛋白单独拟合，而且需要双突变体数据来解耦。拿到一个新蛋白、只有单突变数据，它们给不出答案。

> **我们的定位一句话**：MoCHI 在单个蛋白上**拟合**出两个能量；DETANGO 跨蛋白**分离**但不是物理量、且一个蛋白一个模型；我们要做的是**跨蛋白预测这两个能量**，用物理层连接，用留出家族验证。

---

## 3 · 只做稳定性的那一支（14a 的对手）

| 方法 | 出处 | 性质 |
|---|---|---|
| **SPURS** | [Nat Commun 2025](https://www.nature.com/articles/s41467-025-67609-4) | ESM2 + ProteinMPNN 改线，Megascale 微调；声称"exceptional generalization to unseen proteins"。**最强对手**，但公开材料没说清留出切分怎么做的，需要看正文方法 |
| **ThermoMPNN** | [PMC10402116](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10402116/) | ProteinMPNN 向量 + 迁移学习，Megascale 微调。你 IJMS 那篇已经跑过，切分方法可直接复用 |
| **RaSP** | — | 快速 ΔΔG 预测 |
| **StaB-ddG** | [ICML 2025](https://arxiv.org/abs/2507.05502) | 结合能 = 复合物折叠能 − 各自折叠能。做的是**蛋白-蛋白结合**，不是"功能" |
| **保守性 + 埋藏程度** | [bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.02.03.636212v1.full) | 标题就是"（几乎）只要这两个就够了"。**必须先赢它**，不然整条线站不住 |

这一支全部只输出**一个数**（稳定性），不做功能拆分。所以 14a 是"补齐对照"，不是创新点；创新点在 14b。

---

## 4 · 调整后的角度

**不变**：
- **14a**：稳定性换成 kcal/mol，和 SPURS / ThermoMPNN / RaSP / 保守性基线同题对照，按家族留出。这是入场券。
- **14b**：核心。跨蛋白 + 物理读数层 + 实验监督。三处都和 DETANGO 不同。

**要改**：
- **14c 降级**：功能位点富集不再当作新发现，改成"和 DETANGO 同题对照"：同样是找功能位点，我们的**跨蛋白**模型在**没见过的蛋白**上能不能做到它**每个蛋白单独训练**才能做到的事。这反而是更强的论点。
- **新增一条验证**：和 ddPCA 实测能量**按 kcal/mol 对账**（GRB2-SH3、PSD95-PDZ3、KRAS）。DETANGO 做不了这条，因为它的输出没有单位。这条是我们和它最硬的区别。

**风险**（写在前面，别自欺）：
1. DETANGO 已经占住了"分离稳定性和功能"这个叙事。我们必须把话说成"**跨蛋白预测物理能量**"，而不是"分离稳定性和功能"，否则就是在跟随。
2. 如果 14a 连保守性基线都赢不了，整条线作废，要及时停。
3. DETANGO 是预印本，可能在同行评审中补上跨蛋白实验。要盯着它的更新。

---

## 5 · 还没核实的

- SPURS 正文里"unseen proteins"具体怎么切的（是留出结构还是留出家族？）——做 14a 之前必须查清楚，否则对照不公平。
- DETANGO 全文后 1/3 没读（页面 16.5 万字符，只读了前 10 万）。
- ddPCA 三个数据集的实际可获得性和格式。
- Cagiada 2023 分类器的代码和数据是否公开，能否直接当基线跑。
