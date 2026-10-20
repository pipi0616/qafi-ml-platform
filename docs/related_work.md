# 第二阶段 F · 第 0 步：文献核对

> 2026-10-15。**目的不是抢新意**（这个项目不投论文，是求职作品集），而是搞清楚：这条路别人怎么走的、
> 有哪些现成的方法可以当对照、面试被问到时我们站在什么位置。
>
> 结论：**方向是对的，已经有人在做**（DETANGO，2026 年 2 月预印本），说明这不是自嗨。我们照着做，
> 并且补上它没做的两件事：**跨蛋白**（它一个蛋白一个模型）和**物理单位**（它输出 log-odds，不是 kcal/mol）。

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

### 对我们的影响

**好消息**：功能位点富集、BioLiP、KRAS 变构口袋这些验证，它都做过了，**说明这些验证是这个领域认的做法**。我们照着做同样的验证，结果可以直接对照它的数字，不用自己发明评价标准。14c 从"新发现"改成"同题对照"，工作量更小、说服力更强。

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

**风险**（只剩一条真的）：
如果 14a 连"保守性 + 埋藏程度"这个简单基线都赢不了，说明模型没学到东西，要停下来查原因，而不是硬往下做。

（"被人抢先"不算风险：这个项目不投论文。别人做过，我们实现得出来、讲得清楚、还能说出和它的区别，面试上就是加分。）

---

## 5 · 还没核实的

- SPURS 正文里"unseen proteins"具体怎么切的（是留出结构还是留出家族？）——做 14a 之前必须查清楚，否则对照不公平。
- DETANGO 全文后 1/3 没读（页面 16.5 万字符，只读了前 10 万）。
- ddPCA 三个数据集的实际可获得性和格式。
- Cagiada 2023 分类器的代码和数据是否公开，能否直接当基线跑。

---

# 第三阶段 G · 第 0 步：文献核对

> ⚠️ **2026-10-20 第一版：下面第 6–9 节的数字全部来自检索摘要，正文没读，不可用于任何决定。**
> 用户的规矩：**所有文献参考都必须读正文**，读完才决定怎么做。正文核对正在进行，结论会覆盖这一版。
> 在那之前，第 9 节的"章节调整"只是待议，不是定稿。

> 目的：别人怎么走的、哪些能当对照、面试时我们站在哪。
>
> 初步印象（**待正文确认**）：方向是对的，而且比 F 阶段时热闹得多——"反向折叠 + 一个功能/活性项来挑"
> 这件事 2025–2026 有好几组在做，其中一篇上了 Nature Communications 并带湿实验。如果成立，这是
> **好消息**：任务是真的，而且有发表级的对照数字可以抄作业。

## 6 · 两个能量的分解：又多了一篇，而且它用了反向折叠

| 工作 | 出处 | 做了什么 | 和我们的关系 |
|---|---|---|---|
| **DETANGO** | 2026 预印本 | 见上面第 1 节 | F 阶段的主要对照 |
| **Galpern et al.**《Blending physics-based and inverse folding models to disentangle variant effects on stability and function》 | [bioRxiv 2026.07.30.741764](https://www.biorxiv.org/content/10.64898/2026.07.30.741764v1) | 把变异的进化代价拆成**折叠能 + "dark energy"**（折叠之外的功能约束）。用反向折叠模型 + 粗粒化物理势 + 进化语言模型混合，**无监督** | **最接近 14c 的新工作**，而且它用的正是我们 15a 要用的反向折叠模型。但它**只打分、不做设计**，摘要里没给 kcal/mol 单位 |
| 生物物理上位效应（protein G） | [MBE 2018](https://academic.oup.com/mbe/article/35/10/2345/5063899) | bound / unbound / unfolded 三态热力学模型，靠能量近似可加来分开界面失稳和整体失稳 | 和 14c 读数层同源的老思路；**它也依赖可加性**，而我们第 15 章前提查验发现 Tsuboyama 的"非可加性"主要是量程压顶 |

**Galpern 里对我们最有用的一句**：它报告"reduced IF model bias at functional sites"——**反向折叠模型在功能位点上有偏**。
这不是我们自己猜的动机，是文献里的已知问题，可以直接写进 15b 的动机段，并且是一条要去打的数。

## 7 · 设计侧：提稳定性不丢功能，已经有人做了

| 工作 | 出处 | 关键数字 | 作者自己承认的缺口 |
|---|---|---|---|
| **ABACUS-T** | [Nat Commun 2025](https://www.nature.com/articles/s41467-025-65175-3) | 多模态反向折叠。重设计 xylanase 和 TEM β-内酰胺酶，**活性保持或超过野生型**，设计全部 **+10 °C 以上**热稳定性。**有湿实验** | "设计过程本身还没有主动针对催化效率的提升"。另外作者指出：只在**单一结构**上优化序列会损害蛋白采样其他构象的能力，这可能解释固定主链反向折叠设计丢活性的现象 |
| **CatIF-RL** | [bioRxiv 2026.05.11.724288](https://www.biorxiv.org/content/10.64898/2026.05.11.724288v1.full.pdf) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/42466941/) | 用 GRPO 把反向折叠导向更高预测 kcat。留出基准上预测 kcat 约 **4 倍**，序列恢复率 **0.55** | **增益是代理预测器给的，不是湿实验** |
| **EditorForge** | [bioRxiv 2026.06.08.730993](https://www.biorxiv.org/content/10.64898/2026.06.08.730993v1) | **✅ 正文已读，见 7.1。我原来写的那条是错的** | 见 7.1 |
| **From Structure to Function** | [KDD 2026](https://dl.acm.org/doi/10.1145/3770855.3818896) · [代码](https://github.com/nilufertamatgar/Function-Aware-Protein-Inverse-Folding) | 用偏好对齐让反向折叠 function-aware | 方法细节没核实 |
| 抗体反向折叠基准 | [bioRxiv 2025.08.05.668698](https://www.biorxiv.org/content/10.1101/2025.08.05.668698v1) | **✅ 正文已读，见下面 7.1 节的更正** | 见 7.1 |

> ⚠️ 这里原本有两段从摘要推出来的结论（"EditorForge 的审计证明注释清单不够"、"抗体基准预示
> ProteinMPNN 不如 ESM-IF"）。**读完正文发现两段都是错的，已删除，更正见 7.1 节。**

## 8 · 我们的位置：三件他们都没做的事

1. **功能项带单位，而且对过账。** Galpern 做分解但不设计、没单位；ABACUS-T 的功能项是隐式的多模态；
   CatIF-RL 的功能项是 kcat 代理。我们的功能能是 kcal/mol，并且 14c 已经和 ddPCA 实测能量按 kcal/mol
   对过账（GRB2-SH3、PSD95-PDZ3、KRAS）。
2. **评估规模。** 现有工作要么单个酶 + 湿实验（ABACUS-T，n 很小），要么代理预测器（CatIF-RL）。
   我们有 14x 的 **128 组留出 DMS 实验、92 个考试蛋白、22 万个实测突变**，而且已经证明过引导挑选的增益。
3. **消融。** 没有一篇报告"只用生成器"vs"生成器 + 功能项"。而对我们来说这不是加分项——
   生成器预训练见过考试蛋白，**这条消融是第 15 章列的第 3 号泄漏面的唯一解药**，所以它是过关标准。

## 9 · 对章节的调整

- **15a 定位下调成"复现 + 建基线"**：生成器同题对照文献里做得不少（抗体基准、EditorForge 的审计），
  不当新发现，当我们自己的对照组和入场券。照 portfolio-not-publication 的规矩，这是加分不是减分。
- **15c 明确和 ABACUS-T 同题对照**，并在 notebook 里写清我们能做什么它没做（上面第 8 节三条）。
  它的 +10 °C 和"活性保持"是我们要引的参考数字。
- **15b 的动机段引 Galpern** 的"IF 模型在功能位点有偏"，把消融从"我们谨慎"升级成"文献已知问题"。

## 10 · 还没核实的（第三阶段）

- 上面 2026 年的四篇**都是预印本**，数字来自检索摘要，**正文没读**。写 15a/15c 之前必须核对原文，
  尤其 ABACUS-T 的留出切分和 CatIF-RL 的"留出基准"到底是什么。
- ABACUS-T 和 CatIF-RL 的代码/权重是否公开，能不能直接当基线跑。
- 抗体基准里 ProteinMPNN 表现差，是方法问题还是抗体这个体系特殊——会影响 15a 的解读。
- ProteinGym 的多突变实验到底有多少可用（第 15 章查到我们手上只有 k=2，且只有 Tsuboyama 一个来源）。

---

## 7.1 · 正文已读（持续更新中）

| 文献 | 状态 |
|---|---|
| 抗体反向折叠基准 | ✅ 已读完整正文（20 页） |
| EditorForge | ✅ 已读完整正文（23 页） |
| CatIF-RL | ✅ 已读完整正文（32 页 + 完整 SI） |
| **Galpern（AWSEM-IF blend）** | ✅ 已读完整正文 + 补充方法 + 公式图 |
| **ABACUS-T** | ✅ 已读正文 + 补充 PDF + Source Data + 原分辨率图 |
| FPA（KDD 2026）+ EnzymeArt | ✅ 已读（FPA 读的是作者 camera-ready PDF，ACM 页 403） |
| SPURS + DETANGO（含后 1/3） | ✅ 已读正文 + 补充材料 + 一个 Author Correction |
| Cagiada 两篇 + Otwinowski 2018 | ✅ 已读（注意 arXiv 版和正式版数字不同） |
| **Poley-Gil**（Galpern 引的 [67]） | ✅ 已读正文 34 页 + 补充 34 页 |
| **新增**：Poley-Gil 引的 Sumida 2024（TEV 蛋白酶） | ⏳ 在读（15c 前提最直接的出处） |

**读完两篇，我原来从摘要写的结论两条全错。** 记录下来当教训。

### (1) 抗体反向折叠基准 — Janusz et al.（NaturalAntibody + Boehringer-Ingelheim）

[bioRxiv 2025.08.05.668698](https://doi.org/10.1101/2025.08.05.668698)，2025-08-07，未经同行评审。

测四个模型：ESM-IF1、ProteinMPNN、AntiFold（ESM-IF 的抗体微调版）、AbMPNN（ProteinMPNN 的抗体微调版）。
唯一的打分是 **perplexity**。三个数据集：

| 数据集 | 规模 | 真值 |
|---|---|---|
| HER2-aff-large（公开） | 52 万 Trastuzumab 变体 | 三档亲和力分类 |
| BI-PD1（内部，不公开） | **只有 118 条** | 59 个真 binder + **59 个随机打乱 CDR3 造的合成负样本** |
| AbDesign DB（作者自己的） | 14 抗体 / 7 靶点 / 658 个单点突变 | ELISA 突变/野生型比值 |

| 模型 | HER2（高 vs 低 / 中算非 binder） | BI-PD1 |
|---|---|---|
| **AbMPNN** | **96% / 88%** | 75%（top 12），之后迅速衰减 |
| **AntiFold** | 93% / 75% | 75%（top 12） |
| **ProteinMPNN** | **86% / 73%** | 最高 67%，无分离 |
| **ESM-IF** | 最差（没给数字） | 71%（top 24）→ **再往后掉到 54%** |

**我原来写错的两处：**

1. 我写"ProteinMPNN 最差，binder 和非 binder 完全没分离"并当成一般规律。**实际只发生在那个 118 条、
   负样本是合成的小数据集上**；在 52 万变体的 HER2 上 ProteinMPNN 拿 86%/73%，**反而赢 ESM-IF**。
   作者的归因是三条情境性的：没做抗体微调、数据太小、MPNN 系的采样随机性。**全文没有任何一句**
   说 ProteinMPNN 普遍排不好点突变，也**没有**任何能否推广到非抗体的说法（NOT STATED）。
2. 我把 ESM-IF 的 71% 当亮点。那是同一个小数据集上的，作者原话是 "surprisingly competitive"，
   而且紧接着 "dropping to about 54% at larger selections"。

**另外四条对我们有用的：**

- **ProteinMPNN 和 AbMPNN 根本没进抗原感知那组基准**："ProteinMPNN was unable to handle the antigen
  information"。所以这篇真正的负面结论是**抗原信息几乎没被用上**，和生成器高下无关。
- AbDesign 上的 Pearson 只有 **0.036–0.247**，而且 **single-chain 在 4 行里有 3 行赢过 multi-chain**
  ——加了抗原反而更差，正文没有解释这个矛盾。
- **用预测结构比晶体结构掉 5–10 个百分点**。我们平台用的是 AlphaFold 模型，这条是真约束。
- **这篇不报序列恢复率，而且是故意的**：作者认为 AAR 就是模型的训练目标，不能拿来当评价指标。
  我原来把"序列恢复率约 50%"当 15a 的过关标准，这个指标本身在这篇里是被否定的。
- **oracle 自己就不可靠**：随机生成的 CDR-H3 被分类器判成 binder 的比例是 **5.6% / 38.0% / 10.6%**，
  作者自己说 "quite extraordinary"。这对任何"用代理打分器评设计"的章节都是直接警告。

**可用性**：全文**没有**代码/数据可用性声明，内部数据集不公开，**全文没有任何统计检验**（无 p 值、
无置信区间、无重复种子）。

### (2) EditorForge — Chen et al.

[bioRxiv 2026.06.08.730993](https://doi.org/10.64898/2026.06.08.730993)，2026-06-11，未经同行评审。

**先说来源**：作者单位是 "St. Andrew's College / SAC Scientific and Mathematical Research Collective"
——一个**高中生研究团体**，13 位作者加一位本科生，致谢里感谢"团体的主任"。不是说高中生做的就不算，
但引用它当证据之前必须知道这件事。

**我原来写错的那处（最严重）**：我写"审计发现活性位点附近的失败模式，**尽管固定位点的置信分看起来
很高**"。**全文不报任何置信分**。原文用的词是 "fixed-position integrity"，意思是"零违规"，
我把它编成了"置信分很高"。这是我凭空加的。

**而那个"关键发现"本身是循环的**：所谓的失败是**他们自己写的 mask 启发式**违反了**他们自己写的
shield 规则**（25 个可设计位点里有 5 个落在他们定的 16 Å 禁区内）。**预生成那一轮的候选一个都没报**
——没有数量、没有序列、没有指标——所以**没有任何证据表明 ProteinMPNN 的输出有问题**。

**活性位点是怎么定的（我最想知道的那个问题）：手工写死三个残基 `A = {150, 224, 225}`，全文从头到尾
没有说这三个数从哪来。** 不是数据库注释、不是配体距离（4MH8 是 apo 结构）、不是模型、没有引文。
作者自己承认："These values define the exclusion geometry used for the MMLV RT run rather than
universal biological constants."

**其余要点：**

- **单个蛋白**（MMLV 逆转录酶 4MH8，453 残基），25 个可设计位点 / 428 固定（只有 5.5% 可动）。
- **没有任何基线，没有任何消融。** 8/8 和 24/24 的"通过率"是对着作者自己**没有公开**的阈值算出来的。
- **论文明确没有证明那个 shield 有用**——作者把这个对照推给未来工作："a future comparison could
  generate one candidate set using ordinary fixed-position inverse folding and another using EditorForge"。
- 没有湿实验，作者反复声明："we do not claim that any generated candidate improves ... activity"。
- 大量关键参数 NOT STATED：接触距离阈值、RefoldQC 的验收阈值、外围复查区半径、**到底用了哪个结构
  预测器**、ProteinMPNN 的版本/采样温度/随机种子、24 个候选里怎么挑出 8 个去重折叠。

**结论：这篇不能当证据用。** 它只能当"有人想到了同一个问题"的存在性证明。它反而说明一件事——
**"把活性位点固定住"这件事在文献里目前就是手工画圈 + 欧氏距离**，没有人用一个学出来的功能代价模型。
这对我们是个空位，但要等其余几篇读完才能确认。

### (3) CatIF-RL — Li et al.（北京理工 + 人民大学）

[bioRxiv 2026.05.11.724288](https://doi.org/10.64898/2026.05.11.724288)，2026-05-15，未经同行评审。
正文 32 页 + 完整 SI（表 S1–S12）全部读完。代码和权重：https://github.com/lynk-101-li/CatIF-RL

**我上一轮漏掉的，比我写的那条严重得多。**（我只说了"增益来自代理预测器"，方向对但远远不够。）

先更正两个事实错误：

1. 被导向的基模型**不是** ProteinMPNN / ESM-IF，是 **GraDe-IF**——一个图去噪离散扩散的反向折叠模型
   （BLOSUM 转移核，500 步扩散）。ProteinMPNN / ESM-IF / LigandMPNN / ABACUS-T 在这篇里是**基线**。
2. 序列恢复率 **0.55 不是"保持"，是掉下来的**。同一架构早一个阶段的 EnzymeIF 是 **0.7088**。
   摘要写 "maintaining sequence recovery (0.55)"，实际是用 0.155 的恢复率、5.9 的 pLDDT、
   1.47 Å 的 RMSD 换来的活性增益。

**真正的问题：标尺就是靶子。**

奖励是三个现成冻结的 kcat 预测器（DLKcat、UniKP、CataPro）的集成。而 SI §S1.6 原文承认这三个
"used both for activity reward (manuscript §2.5) **and held-out evaluation** (manuscript §2.7)"。
更要紧的是，**表 2 那个头条数字是 DLKcat 单独算的**——奖励成员之一。全文和 SI 里**没有任何一张表**
报 UniKP 或 CataPro 下的基准结果。那个"留一个预测器出来评"的检查，模型都装好了，**没做**。

而且这个 oracle 的精度**全文没有报过**——没有任何 R²、RMSE、Pearson 对实测 kcat。唯一的量化是
**三个预测器之间的一致性，而且很差：Pearson r = 0.13–0.59**（25,970 组配对）。三个号称测同一个物理量
的模型互相只有这个一致性。

所谓"约 4 倍"= 10^0.6049 ≈ 4.03×，**是策略在它被训练去最大化的那个预测器上的得分**。

**所有可独立验证的指标都朝反方向走：**

| RL 轮次 | 奖励指标 Δlog₁₀kcat | 恢复率 | pLDDT | RMSD (Å) | 结构+功能联合通过率 |
|---|---|---|---|---|---|
| CatIF（RL 前） | 0.3647 | 0.5719 | 79.16 | 4.885 | 0.1145 |
| 第 1 轮 | 0.4063 | ↓ | ↓ | ↑ | — |
| 第 2 轮 | 0.5186 | ↓ | ↓ | ↑ | — |
| 第 3 轮 | **0.6049** | **0.5533** | **77.87** | **5.322** | **0.1266** |
| 对照：EnzymeIF（早三个阶段） | 0.2359 | **0.7088** | 83.74 | 3.857 | **0.1252** |

**联合通过率 0.1266 vs EnzymeIF 的 0.1252——三个阶段的努力等于零。** 严格阈值下
CatIF-RL（0.0345）和 LigandMPNN（0.0338）打平，而 LigandMPNN 还是**在不给配体输入**的条件下跑的
（作者自己承认基线被喂得不全）。

奖励单调上升、每一个可验证指标单调下降、联合通过率持平——这是**奖励攻击的典型签名**。
作者的读法是 "guided exploration of dark sequence space"。他们准确地描述了这个现象，
然后选了一个数据支持不了的解释。全文**只出现过一次 "reward hacking"**，是在算法注释里声称
KL 锚定"prevents reward hacking"，没有任何证据。

**切分：只到序列/蛋白簇级别，没有同源切分。** 直接继承 DLKcat 的划分；训练/验证是
**随机打乱（seed 1234）**。全文和 SI 里 **homology / sequence identity / MMseqs / CD-HIT / BLAST
这些词一次都没出现**，没有任何同一性阈值。唯一的聚类 `distc_pro_num` 是"同一个蛋白的不同变体"，
它挡得住同一个蛋白两边都有，**挡不住同家族、同 fold 的近亲**。EC 类只用于事后分层报告，不用于切分。

**其他要点：**

- **完全不涉及稳定性**，纯活性。pLDDT 和 RMSD 是重折叠自洽性，**不是热力学稳定性**，当稳定性代理会读错。
- 平均改掉约 **45% 的残基**（恢复率 0.553 推出）——整条序列重设计，不是点突变。
  突变预算只有一个软惩罚，而且**前 30% 免费**。
- **没有湿实验**，作者明说 "all reported improvements are computational"。所有结构都是 ESMFold 预测的。
- 有一个**做得不错的阶段消融**（GraDe-IF → EnzymeIF → CatIF → CatIF-RL），带配对 Wilcoxon 和 BH 校正。
  但**没有**奖励权重消融、**没有**单预测器 vs 集成的对照。

**对我们最有用的一点**：这篇是**我们那两条防泄漏规矩的现成反面教材**——
"标尺不能是靶子"和"切分必须到同源级别"。它还顺手给了一张 backbone-only 条件下的基线表
（ProteinMPNN 恢复率 0.4704、ESM-IF 0.5359、ABACUS-T 0.6042），虽然基线都被喂得不全，引用时要说明。

### (4) Galpern et al. — AWSEM-IF blend（CRG 巴塞罗那，Dias / Frazer 组）

[bioRxiv 2026.07.30.741764](https://doi.org/10.64898/2026.07.30.741764)，2026-08-04，未经同行评审。
正文 + 补充方法 + 公式图 + 表 S1/S2 全部读完。

**这是 F 阶段以来最接近 14c 的工作，而且我上一轮漏判了一处要紧的。**

#### 我写错的那处

我写"摘要里没给 kcal/mol 单位"。**错了。它有单位，而且和实测能量对过账。**

- 明确换算到 kcal/mol：`ΔE_fold[kcal/mol] = ΔE_fold · σ_ΔΔG / σ_ΔE_fold`，其中 **σ_ΔΔG = 0.9777 kcal/mol**（取自 Megascale 天然蛋白）。
- 而且它对账用的是**和我们 14c 完全同一批双读数实验**：**KRAS（Weng）、PSD95-PDZ3、GRB2-SH3、Src kinase**，拿 MoCHI 推出来的 ΔΔG_fold / ΔΔG_bind / ΔΔG_activity 当真值。

所以我们原来写的"和它最硬的区别是按 kcal/mol 对账"——**这条差别不成立了**，它也做了，用的还是同一批数据。

#### 方法（正文确认）

```
ΔE_fold = (1−λ)·ΔE_AWSEM/σ_AWSEM  +  λ·ΔE_IF/σ_IF        (公式 1)
ΔE_dark = ΔE_evo − ΔE_fold                                  (公式 2)
ΔE_evo  = k_B·T_sel·ΔΨ_evo   （ESM-1v masked marginals，按蛋白定选择温度）
```

物理项是 **AWSEM** 粗粒化力场（只用 burial + contact 项，backbone 固定，忽略熵）。
IF 项试了三个：**ESM-IF1、ProteinMPNN、PottsMPNN**。混合就是**一个标量 λ 的加权和**，不是学出来的组合。
对照里还放了有监督的 **ThermoMPNN**。

#### 它最硬的结果：IF 模型在功能位点上确实有偏，而且量化了

| 在"dark energy 最高的那 1% 位点"上的 RMSE（kcal/mol） | |
|---|---|
| 纯 ESM-IF | ≈ 1.11 |
| 纯 ProteinMPNN | ≈ 0.89 |
| 纯 PottsMPNN | ≈ 0.80 |
| **AWSEM-PottsMPNN / AWSEM-ProteinMPNN 混合** | **0.4–0.6，而且在全程基本平的** |

在 M-CSA 的 785 个酶的**实测催化残基**上：纯 IF 预测 **1–2 kcal/mol 的失稳**，混合后 **< 1**。
而纯 AWSEM 反而把活性位点突变预测成**稳定化**（负值）。作者的结论：
"IF models are biased against the experimentally observed stability-function trade-off,
leading them to overestimate ΔE^fold at catalytic residues."

这条在 Domainome 和那四个双读数 DMS 上都复现了。**所以"IF 在功能位点有偏"现在是有正文、有数字的已知问题。**

Megascale 相关性（表 S1，精确值）：最好是 **AWSEM-PottsMPNN，189 个天然结构域上 Pearson r = 0.7377（λ = 0.6674）**。
纯模型端点〔读自图〕：天然蛋白上纯 AWSEM ≈ 0.46、纯 PottsMPNN ≈ 0.67；de novo 设计上反过来，纯 AWSEM ≈ 0.59 > 纯 IF ≈ 0.42–0.53。

#### 它的方法学缺口：**根本没有留出切分**

这是正文里最值得记的一条。全文**没有任何一句**描述自己方法的 train/test 切分、交叉验证或留出蛋白；
"cross-validation"、"held-out"、"training set" 这些词**只在描述 ThermoMPNN 时出现**
（"Restricting the analysis to its test set"）。

那个唯一的拟合参数 λ 是**在报告性能的同一批 Megascale 数据上取 Pearson r 最大值**定出来的，
然后原封不动用到所有下游分析。σ_ΔΔG 也取自 Megascale 并到处复用。

只拟合一个标量，过拟合风险确实小，**但这不是留出评估**，而且论文没有声称下游数据集和 Megascale
不同源。唯一和家族有关的处理是**聚合而非切分**：Domainome 的 AUPRC 先在 PFAM 家族内平均再跨家族平均。

#### 其余要点

- **只打分，不做设计**（这条我上一轮判对了）。所有分析都是穷举单点替换。
- **只做单点突变，明确把上位效应和多突变排除在外**："leaving epistatic and multi-site effects out of scope"。
- 作者自己说 dark energy **不是严格意义的物理功能能**："its thermodynamic interpretation is clearest
  when the selection temperature for function equals that for folding, as in binding, but the scale may
  differ for other chemical activities such as catalysis"。
- **代码和数据都公开、可直接当基线跑**：
  - 代码 https://github.com/DiasFrazerGroup/blend_and_disentangle（MIT，带 CLI 和复现 notebook）
  - AWSEM 实现 https://github.com/HanaJaafari/Frustratometer/tree/galpern2025
  - **预计算分数 https://doi.org/10.5281/zenodo.21241135**，含 megascale / mcsa / cancer /
    human_subset / human_gof_lof，**单位就是 kcal/mol**
- 下游还做了癌症体细胞突变（癌基因 vs 抑癌基因）和 HGMD 的 GOF/LOF 分类，发现
  **GOF 变异是唯一 dark energy 大于折叠能的一类**。

#### 一条要追的新文献

讨论里指向设计方向时引了一篇：**"Recent observations that functional sites resist stability
optimization in backbone-conditioned sequence generation [67]"**——这正是我们 15c 的前提，而且已经有人
发表了观察。**这篇必须找出来读正文**，加进待读清单。

### (5) SPURS — Li & Luo（Georgia Tech）· **F 阶段第 5 节那个问题的答案**

"Generalizable and scalable protein stability prediction with rewired protein generative models"，
*Nature Communications* **17:891 (2026)**，线上 2025-12-20，DOI 10.1038/s41467-025-67609-4。
**注意有一个 Author Correction**（2026-03-16，DOI 10.1038/s41467-026-70929-8）：图 5（ClinVar 那张）
上传了旧版本，已替换；切分和基准那几张图不受影响。

#### 「unseen proteins」到底怎么切的 — 答案

> "We used the training, validation, and test splits of the Megascale dataset created by Dieckhaus et al.,
> **which ensured no sequences sharing >25% identity across splits**."

> "To prevent data leakage across evaluation sets, any training sequences with >25% sequence identity to
> sequences in the independent test sets were removed using **MMseqs2**."

**所以是 25% 序列同一性的聚类留出，到结构域/序列级别——不是家族级、不是 fold 级。** 全文没有任何
PFAM / CATH / SCOP 家族划分，唯一的同源控制就是 25% 两两同一性。没给覆盖度、E 值或聚类模式参数。

**用的就是 ThermoMPNN 那个切分**（数据可用性直接指向 Kuhlman-Lab/ThermoMPNN），而且**测试集一行没动**，
只是把训练/验证里和十个独立测试集 >25% 同一的拿掉了：

| | ThermoMPNN 原版 | SPURS 过滤后 |
|---|---|---|
| 训练 | 216,919 突变 / 239 结构域 | **203,789 / 212**（去掉 27 个结构域） |
| 验证 | 27,481 / 31 | 26,548 / 29 |
| 测试 | 28,312 / 28 | **28,312 / 28（不变）** |

**对我们的意义**：我们 14a 用的是同一个切分，所以对照是公平的。而且**我们的同源规则比它严**
（局部比对 + 30% 同一性 + 50% 覆盖度 + 跨数据集查到 56 个同源结构域），它只有一个 25% 同一性。

#### 它的数字（表 S1/S3）

**留出测试集 28,312 个突变 / 28 个结构域上，median per-domain Spearman 0.83 vs ThermoMPNN 0.77**，
24/28 个蛋白更好。混合 Spearman 0.74、Pearson 0.78。

**RMSE 全文零次出现**——只报 Spearman 和 Pearson。我们 14a 表里 SPURS 那行 RMSE 留空是对的。

#### ⚠️ 和我们第 15 章前提查验直接相关：**SPURS 已经做过双突变基准**

补充表 7 的双突变切分：训练 122,278/116、验证 10,672/17、**测试 22,081/20**。
**那个 22,081 / 20 和我第 ① 步数出来的完全一样**，互相印证。

而它在这同一批测试双突变上的结果（补充表 4）：

| 方法 | Spearman |
|---|---|
| **SPURS（学出来的）** | **0.59** |
| MutateEverything | 0.54 |
| **SPURS-additive（自己的单突变相加）** | **0.52** |
| ThermoMPNN-additive | 0.45 |
| ESM2-additive | 0.40 |
| DDGun-3D / DDGun-seq | 0.36 / 0.33 |
| MPNN-additive | 0.16 |

**这把第 ② 步的结论从"否"改成"头部空间存在但很小"**：学出来的模型比自己的可加基线只高
**0.07 Spearman**（0.59 vs 0.52）。这和我查到的"非可加性主要是量程压顶"是**一致的**——
正因为大部分差异是测量截断，可学的余量才这么薄。

**第 15 章的 notebook 和 plan.md 第 10 节要据此修订**：不能再写"15a 不能是多突变那一章"，
而要写"头部空间有已发表的数字，只有 0.07，而且 MutateEverything 已经占了这个题"。

#### 它没做的泄漏检查（我们可以做得更好的地方）

- **Megascale 和 Human Domainome 之间没有任何同一性过滤**——而那是它 522 个蛋白的头条结果之一。
- Megascale 和 ProteinGym 之间没有检查（141 个 DMS，只说排除了测稳定性的）。
- Megascale 和 ClinVar / gnomAD / CDD 之间没有检查。
- 没有任何结构/fold 级别的冗余检查。

它自己倒是很诚实地承认**基线被它占了便宜**："their training data are supersets of our filtered
Megascale training set"；MutateEverything "13 of 20 proteins in our test set were included in its
training set"。

**代码和权重都可用**：https://github.com/luo-group/SPURS ，权重在
https://huggingface.co/cyclization9/SPURS （`get_SPURS_from_hub()`）。许可证 NOT STATED。

### (6) DETANGO — 正文读完，我们第 1 节的三条定位全部成立

"Deconvolving mutation effects on protein stability and function with disentangled protein language
models"，Ding, Li, Tu, Luo, Luo（**和 SPURS 同一个组**），bioRxiv v2 2026-02-26，
DOI 10.64898/2026.02.03.703560，PMC12889624。

#### 三个悬着的问题，答案

**① 是不是一个蛋白一个模型、在蛋白内部按残基做交叉验证？是，原文逐字：**

> "**For each protein in the dataset, we individually trained DETANGO** using the loss function described
> above."

> "**For each protein, we performed a 10-fold cross-validation, where residues are partitioned into
> training and validation sets across folds, and constructed the final DETANGO model as an ensemble of
> the 10 independently trained replicas.**"

**关键推论（论文没说）**：最终报告的模型是 10 个复制的集成，所以它**合起来在被打分的那个蛋白的
每一个残基上都训练过**。论文里每一个数字都来自一个拟合过测试蛋白自身残基的模型，没有任何
留出残基的报告。

**② 有没有在任何地方留出整个蛋白或家族、去预测一个零数据的蛋白？没有，全文都没有。**
"held out"、"hold-out"、"unseen" 这些词零次出现。架构上也做不到——模型是拟合到目标蛋白身上的。

**③ 默认稳定性输入是什么？FoldX 预测，跑在 AlphaFold2 预测结构上。** 原文：
"Unless otherwise specified, DETANGO uses FoldX ΔΔG predictions as the default stability measurements."

而且它自己承认实测更好："**DETANGO (DMS) outperformed DETANGO (FoldX)**, highlighting the value of
incorporating experimentally measured stability data over computationally predicted scores"。

**那句"operational rather than strictly guaranteed" 的完整上下文**（在 Methods 公式 5 之后）：

> "**While the separation between stability- and function-specific representations is operational rather
> than strictly guaranteed**, our formulation aims to isolate components of the evolutionary signal that
> are preferentially associated with stability versus with functions."

这句话是在承认 `h_function = h_evolution − h_stability` 只是嵌入空间里的一个减法，没有任何独立性约束。
他们试过加互信息正则项，"modest but statistically non-significant performance gains"，最后**没有采用**。

#### 两个值得记的新发现

- **超参数是在 4 个蛋白上挑的：CYP2C9、NUDT15、PTEN、GRB2——而这 4 个全在它 11 个 SBI 基准蛋白里面。**
  论文没有承认这一点。
- **在唯一一个独立的变构基准上（Amor et al. 的 17 个蛋白 / 159 个位点），DETANGO 的 AUPRC 只和
  FoldX 打平**——而 AUPRC 正是论文自己论证过的、针对这种不平衡标签该用的指标。

#### 大部分数字只在图里

SBI 分类、Domainome 功能位点、BioLiP2、PocketMiner、KRAS 变构、17 蛋白变构、globin、RAS——
**正文和补充材料里都没有给 AUPRC / AUROC / nDCG / F1 的数值**。要引用必须去 Source Data 或读图。
正文里有数的只有：Domainome 408 蛋白 / 213,332 残基 / 17,381 个 CDD 功能位点（基线率 8.1%）、
168 个锌结合位点全部命中、PUM2 的 AUPRC 0.87。

#### 结论：我们第 1 节的三条定位全部被正文证实

| 我们当初写的 | 正文核对 |
|---|---|
| 一个蛋白一个模型，不能预测全新蛋白 | ✅ 逐字证实，而且比我们写的更严重——最终模型见过测试蛋白的每个残基 |
| 不是物理模型、输出没有单位 | ✅ 证实 |
| 在复现语言模型 + 用 FoldX 预测当监督 | ✅ 证实，而且作者自己承认实测比 FoldX 好 |

**同组套娃要注意**：DETANGO 用 SPURS 当稳定性输入（BioLiP2 的 24,252 个蛋白、PocketMiner 的 33 个），
**同时又把 SPURS 当基线**放在同几张图里。而 SPURS 自己那套功能位点方法（ESM1v vs −ΔΔG 的 sigmoid
拟合残差，239 个 Domainome 结构域上 AUROC 0.69）在 DETANGO 里变成了那个
"Logistic regression model" 基线。引用时要说清这层关系。

### (7) Otwinowski 2018 — **第 ② 步那个压顶发现的直接先例**

"Biophysical Inference of Epistasis and the Effects of Mutations on Protein Stability and Function"，
Jakub Otwinowski（单作者），*Mol Biol Evol* **35**(10):2345–2354 (2018)，DOI 10.1093/molbev/msy141。
代码数据 https://github.com/jotwin/ProteinGthermo （含 `energies.csv`）。
⚠️ arXiv v2 和**正式版数字不同**，正式版更保守，引用必须用正式版。

三态热力学模型（未折叠-未结合 / 折叠-未结合 / 折叠-结合），拟合 GB1 的 1,045 个单突变 + 约 53.6 万个
双突变（Olson 2014），推出 2,092 个能量参数。

#### 它已经发现了测量地板制造假上位效应，机制和方向和我们一模一样

> "A minimal amount of non-specific background binding, p0, **imposes a lower bound on measured binding
> fraction in the experiment**, estimated to be f0 = −5.69 by our three state model. **This threshold
> effect produces large amounts of non-biological positive pairwise epistasis**, e.g. when the double
> mutant has the same level of binding as one of the single mutants at the background level."

它的处理方式是：把 p0 显式写进似然，**并且把靠近地板的那些配对从上位效应分析里剔除**。

**我们第 ② 步的发现因此不是新现象，但量化方式是新的**：它从未把这件事叫做 saturation 或 dynamic
range（这两个词全文都没出现过这个意思），**没有推广到整个数据集**，而且是把受影响的配对**丢掉**而不是
**分解**。我们在 131,143 个双突变上做的"残差 vs 预测严重程度相关 −0.72、残差 vs 位置间距无关"这个
分离检验，它没做。**所以第 ② 步的结论要改成"有先例、我们做的是量化和推广"，不能写成新发现。**

它的量化：一个**完全没有能量耦合**的模型就能复现观测到的配对上位效应 **ρ = 0.66、符号准确率 0.73**
（剔除近地板配对之后）。也就是说，看起来的上位效应里**约三分之二只是测量非线性**。

#### 另一个结果直接支撑我们 14c 的做法

两态 vs 三态模型：对适应度读数的拟合几乎一样好（**96.4% vs 97.1%**），但三态模型的能量和 81 个
独立实测 ΔΔG 相关 **ρ = 0.91、RMSE 0.39 kcal/mol**，而两态模型的能量 **"have no relation to the
independently measured ε_f"**。

**拟合读数拟合得好，完全不保证推出来的潜在能量是物理的。** 这是"必须和实测 kcal/mol 对账"这件事
现成的、发表过的论证——正是我们 14c 做的那条验证。

#### 对我们 k=2 边界的直接警告

外推到四突变（Wu et al. 的 204 个变体）时：相关性掉到 **0.48**；分类上
**"classifying only 11% of functional quadruple mutants correctly"**（正式版数字，arXiv 版写的是
86% TPR，差别巨大）；约 **20%** 的变体适应度被严重低估，"some unaccounted for epistasis is restoring
binding or stability"。

**这给我们第 15 章"k≥3 必须标成外推"提供了一个发表过的量化依据。**

### (8) Cagiada 的两篇 — F 阶段第 5 节那个"能不能当基线"的问题：**能，而且很容易**

**注意我们原来记的"Cagiada 2023"其实是两篇，而且分在两条技术路线上：**

| | Cagiada et al. 2023 | FunC-ESMs 2025 |
|---|---|---|
| 出处 | *Nat Commun* **14**:4175，DOI 10.1038/s41467-023-39909-0 | *eLife* Reviewed Preprint，DOI 10.7554/eLife.108160.1 |
| 功能轴 | **GEMME**（比对/进化，不是语言模型） | **ESM-1b** |
| 稳定性轴 | **Rosetta** Cartesian ddG（或 RaSP） | **ESM-IF**（跑在 AlphaFold2 结构上） |
| 分类方式 | **训练过的 CatBoost 分类器**，8 个特征 | **两个固定阈值**（ESM-1b < −6.5，ESM-IF < −7），不训练 |
| 训练数据 | NUDT15 / PTEN / CYP2C9，9,945 个变体 / 923 个位置 | 不训练；阈值用 32 个 ProteinGym MAVE + Tsuboyama 的 17.4 万个 ΔΔG 定 |

**能不能当基线跑：能。** 2023 那篇的**训练好的权重直接提交在仓库里**
（`catboost_model/cat_trained_24jun22.cbm`，Apache 2.0），还带一个一键 Colab
（https://github.com/KULL-Centre/_2022_functional-sites-cagiada ，Zenodo 10.5281/zenodo.8046585）。
每个新蛋白需要：序列、结构、一个 kcal/mol 的 ΔΔG 文件（RaSP 是快路径）、一个 GEMME ΔΔE 文件。
**阈值不用重新拟合，已经烤进模型。** FunC-ESMs 更简单——两个发表过的数字，输入只要序列 + AF2 结构
（https://github.com/KULL-Centre/_2024_cagiada-jonsson-func ，全蛋白组预测在 KU ERDA）。

#### 它们的留出数字，我们 14d 对照时要用这个

2023 那篇的主模型是**在三个蛋白上全部训练**的，留出蛋白只是补充材料里的稳健性检查。
补充表 2 的留出蛋白 F1：

| | 留出蛋白上 | GRB2-SH3 上（全模型） |
|---|---|---|
| 功能性**变体** F1 | **0.13 – 0.33** | **0.32** |
| 功能性**位点** F1 | 0.51 – 0.58 | **0.60** |

**正文里那些漂亮数字（95%、90%）是训练集上的回代，不是留出。** 它和前人方法比的那条是
**10 个酶上 77/109 个已知功能位点，TPR 0.71**，对照 0.40（Cheng 2005）和 0.60（Chelliah）。

两条和我们直接相关的：

- **它把 ΔΔG 截断在 0–5 kcal/mol**："setting all values below/above a range of 0 – 5 kcal/mol, to 0 or
  5"。又一个处理量程的例子。
- **它自己承认的那条局限和我们 14d 的盲点是同一个**："some residues will have a direct role in
  function, but also be important for protein stability; **our analysis will miss those residues**" 以及
  "**the reverse is not true**"。我们 14d 的 SHP2 和变构口袋失败属于同一类。
- 它引 Tsuboyama 时说："Very recently, it has been shown that one can also use large-scale measurements
  of stability changes as input to an approach inspired by our work."

### (9) FPA（KDD 2026）— 偏好对齐让反向折叠 function-aware

"From Structure to Function: Preference Alignment for Function-aware Protein Inverse Folding"，
Tamatgar, Park, Yao, Chen, Pan（A*STAR 新加坡 + 延世大学），KDD'26 pp. 12066–12077，
DOI 10.1145/3770855.3818896。代码 MIT：https://github.com/nilufertamatgar/Function-Aware-Protein-Inverse-Folding

微调 **ProteinMPNN 和 ESM-IF**，用的是 **SPPO 推出来的损失（不是 DPO，论文明确反对 DPO）**。
"功能"的定义是**位置性的**：用 **SIFT**（保守性预测器）找关键位点，在那些位点上放 SIFT 判为
"不容忍"的替换当负样本，再要求**两个 IF 模型都给这个负样本比野生型更高的似然**（硬负样本挖掘）。
作者自己把这种监督叫 "**hypothetical**"、"**weak** but biologically grounded"。

**两条要记的：**

1. **它自己的表 1 里，普通 DPO 在它的头条指标（关键位点恢复率 CSR）上赢过它的 FPA**：
   ESM-IF+DPO 34.29 vs +FPA 31.22；ProteinMPNN+DPO 21.64 vs +FPA 20.08。它声称的胜利是"权衡"
   （DPO 的困惑度和序列恢复率崩了），不是 CSR。
2. **完全没有湿实验**，而且这是它的卖点："enabling scalable supervision without additional wet-lab
   measurements"。4.5 节那个酶的案例是**回溯重打分**一个已有数据集，没有一条 FPA 设计的序列被表达过。

切分是继承 ProteinMPNN 的 CATH 4.2 40% 非冗余结构簇 + 一个和 ESM-IF 预训练划分的重叠过滤，
**没有家族级留出**。另外它**完全不评估折叠结构**——没有 AlphaFold/ESMFold、没有 scTM、没有 pLDDT。
代码虽然是 MIT，但**没有 SIFT 流程的代码**，也就是说它真正的贡献（偏好对的构造）**没法从头复现**，
只放了产出的 CSV。

### (10) EnzymeArt（bioRxiv 2026）— **有真湿实验，57/60 条有活性**

"Function-guided design of active enzymes"，Hu, Wu, Yang, Li, Zhu（浙大 + 清华深圳），
bioRxiv 2026-06-29，DOI 10.64898/2026.06.27.735025。

**先更正分类：这篇不是反向折叠。** 它是**功能条件的从头酶生成**——输入是文字提示（EC 号、酶名、
InterPro 域），不需要主链。反向折叠（ProteinMPNN）只是它第 4 阶段精修里的一个子步骤。

**它是这一批文献里唯一有真湿实验的：合成 60 条基因，57 条有活性**（ADH 17/20、MDH 20/20、
脂肪酶 20/20），纯化后做了稳态动力学（ADH11 k_cat 223.7 s⁻¹、k_cat/K_M 22.24 s⁻¹mM⁻¹，
在同样条件下超过天然对照 P00330）。MDH19 和最近的 BLASTP 命中只有 **33% 同一性**。

**但是：**

- 三个酶类都是**成熟的天然酶类**，作者明说 "not ... a demonstration of unrestricted enzyme design"。
- **挑候选去合成的那个动力学排序模型用的是随机切分**（随机 10%、随机 9:1），**不是同源感知的**。
  这是整篇评估里最弱的一环，因为它决定了合成哪些。
- **代码和权重都还没放**（"upon publication"）。
- 它自己报告了一个新颖性比较的缺陷（查询文件 4,736 条里只有 926 个唯一标头）。

**一句话值得抄进我们的讲稿**：作者自己的结论是
**"Consequently, generation alone is insufficient."**——生成本身不够，必须有筛选。
这正是"生成器提方案、两个能量做选择"这个框架的论证。

### (11) ABACUS-T — **15c 最重要的对手，而且我从摘要写的那条错得最厉害**

"Enhancing functional proteins through multimodal inverse folding with ABACUS-T"，Liu, Wu, Wang 等，
*Nature Communications* **16**:10177 (2025)，2025-11-19，DOI 10.1038/s41467-025-65175-3。
代码 Zenodo 10.5281/zenodo.17089342，**仅限非商业用途**，权重是否包含 NOT STATED。

模型是**序列空间的离散扩散（DDPM）**，不是自回归；每步同时解码残基类型**和侧链构象**；
"多模态"= 主链 + **配体原子** + **多个构象态** + **MSA**（后两个是另外微调的变体）。

#### 我写错的那条（最严重的一次）

我写："重设计 xylanase 和 TEM β-内酰胺酶，**活性保持或超过野生型**，设计**全部 +10 °C 以上**"。

**正文的真实数字：**

| | 测到有活性 | **活性 ≥ 野生型** | 测过 Tm 的 |
|---|---|---|---|
| Xylanase（12 条） | 12/12 | **2/12** | 2 条（+19、+17 °C） |
| TEM-1（11 条） | 10/11 | **4/11** | 3 条（+4、+12、**+23** °C） |

**整体诚实的成功率：23 条全序列酶重设计里，6 条达到或超过野生型活性（约 26%），
其中只有 2 条同时拿到明确的 Tm 提升和保住的活性。** 而且约 66 条设计里**只有 10 条测了 Tm**，
作者明说是"to balance cost"。**全文没有任何 kcat / KM / kcat·KM⁻¹**——所有"活性"都是单点相对速率。

还有一条**只在 Source Data 里、正文没写**的：`txyl1-4`（纯 ABACUS-T 不带 MSA 的那四条）
"used about 10 times the amount of natural protein due to its low activity"——
也就是**比野生型低一到两个数量级的比活**。

#### 三条对我们 15b/15c 是直接证据，而且是湿实验级的

**① 反向折叠模型单独上，确实会在提稳定性的同时毁掉功能。作者自己的话：**

> "for the enzymes tested here, improvements in inverse folding achieved by ABACUS-T **in the absence of
> evolutionary information are insufficient for designing extensive stabilizing mutations without
> compromising catalytic activity**."

数字对得上：纯 ABACUS-T（不带 MSA）在 xylanase 上 **0/4** 条活性增强、在 TEM 上 **0/3**。
它们的解法是加 **MSA 进化信息**，不是加一个功能代价模型。

**② "按注释把活性位点固定住"这件事被湿实验证伪了。**
ProteinMPNN 对照组**固定了活性位点残基**（xylanase 固定 12 个、TEM 固定 7 个、ABP 固定 7 个），
结果：

> "Purified proteins of these two designs **did not show detectable catalytic activity**."

xylanase 0/2、TEM 0/2、ABP 结合 0/4。**这比我之前引的 EditorForge 那条循环论证强得多**——
这是真实验、真阴性。

**③ 作者把我们打算做的事明确写成了未来工作：**

> "Possible improvements on this aspect include ... **using activity-aware scoring functions trained on
> experimental data** to prioritize designs with enhanced activity alongside stability."

这正是我们那个两能量 oracle。**发在 Nature Communications 上，写成了下一步。**

#### 它还给了一条干净的活性-稳定性权衡前沿

`prior weight α` 这个旋钮直接控制权衡，作者原话："**implicitly controlled the tradeoff between
retaining catalytic activity and enhancing structural stability**"。TEM 上看得最清楚：

| 设计 | 突变数 | 相对活性 | Tm 变化 |
|---|---|---|---|
| plact2-0.5（α=0.5，重 MSA） | **12** | **2.60×** | 仅 +4 °C |
| plact2-0.2（α=0.2） | 37 | 1.44× | +12 °C |
| tlact2（纯 ABACUS-T，无 MSA） | 56 | 0.77× | **+23 °C** |

**突变数越多、越不管进化信息 → 越稳定、活性越差。** 这条前沿可以直接当我们 15c 的参照。

#### 它的切分和两个短板

**切分：MMseqs2 30% 同一性 + PDB 发布日期截断，没有家族级或 fold 级留出。**
而且——**那四个湿实验靶点（1gud/1rpj、4pn2、1jvj、5wi3）全部早于它 2022-04-30 的训练截断**，
按它自己描述的建库规则，这些蛋白（或其 30% 同一性簇）**就在训练池里**。论文对此一字未提。

**单突变 ΔΔG 上它输给物理方法**（补充表 5）：FoldX **0.597**、Rosetta **0.553** vs ABACUS-T **0.540**。
作者承认："deep learning models, including ABACUS-T, **do not surpass traditional models in predicting
the effects of single mutations**."

#### 一个对 15a 有用的数字（非抗体体系）

ProteinGym 零样本（21 个蛋白、334,215 个变体）平均 Spearman：
ABACUS-T_prior **0.48** > MSA Transformer 0.47 > ABACUS-T 0.45 > ESM-IF1 0.42 > ESM2-650M 0.39 >
**ProteinMPNN 0.23（垫底）**。

**这是一个正经的、非抗体的证据，说明 ProteinMPNN 在排序突变效应上很弱。**
我之前从抗体那篇得出同样结论是引错了来源，但结论本身在这里有更合适的出处。

### (12) Poley-Gil et al. — Galpern 那条 [67]，**而且方向和我以为的相反**

"Adaptive and Spandrel-like Constraints at Functional Sites in Protein Folds"，
Poley-Gil, Fernández-Martín, Banka, Heinzinger, Rost, Valencia, Parra（巴塞罗那超算中心 + TUM），
bioRxiv 2026-02-11，DOI 10.64898/2026.02.09.704872。
代码 https://github.com/miriampol2c/architectural-constraints （Apache-2.0，已公开），
结果数据 Zenodo 10.5281/zenodo.18922047，重训练权重 10.5281/zenodo.18957473。

#### 它报的是**两个相反的行为**，不是一个

| 体系 | ProteinMPNN 的行为 |
|---|---|
| **β-内酰胺酶** | **抗拒**优化掉功能位点的 frustration——K73、K234 以及非催化的 D131 在**所有采样温度、5 个重训练模型、两套参数、以及换用 Caliby**之下都被反复放回原生（且失稳）的身份 |
| **α-珠蛋白** | **确实抹掉**功能位点特征——12 个保守且高 frustration 的位置，T=0.1 时只剩 2 个，**T=1.0 时一个不剩**，换成了降低 frustration 的残基 |

**而 Galpern 引它引的是 β-内酰胺酶那一半**（原话："functional residues in the β-lactamase fold are
known to **resist** stability optimization by IF models"）。所以我上一轮说"Galpern 指向我们 15c 的前提"
**理解偏了**——它指的是"抗拒"，不是"毁掉"。

#### 对 15c 的前提，它只证明了一半，而且不是用实测单位

**证明了的**：α-珠蛋白上，两个独立的 IF 模型（ProteinMPNN 各温度 + Caliby）在无约束设计下
**抹掉了 12 个功能位点里的 10–12 个**（PPI 位点、盐桥、AHSP 伴侣结合位点），同时把残余 frustration
从天然的约 10% 压到 **1–4%**，而且设计的 pLDDT 很高。这是"无约束 IF 设计会抹掉功能位点特征、
同时改善稳定性代理指标"的干净、且跨模型一致的证据。

**没证明的，三条：**

1. **功能从未被测量。** "毁掉功能"指的是**丢掉一个数据库/前人文献的注释**。约 25,000 条设计里
   **没有一条做过活性、结合或热稳定实验**。作者自己说："Further studies are required to assess the
   behavior of these designs as functional β-lactamases."
2. **稳定性也从未被测量。** 没有 ΔΔG、没有 kcal/mol。读数是 **FrustratometeR 的局部 frustration
   Z 分数** + pLDDT。frustration 是局部能量冲突的代理，**不是折叠自由能**。
3. **它完全没有检验"解药"。** 没有功能代价信号、没有约束设计的自身对照，
   **所以"需要一个功能代价信号"这半句它一个字都没证。**

另外：中心观察本身**没有任何留出验证**（4 个家族、描述性分析），作者称之为 "initial evidence"。

#### 它的消融做得很扎实，值得学

温度扫 0.1–1.0；**把酶类（EC 3，也就是 β-内酰胺酶所属类）从训练集里删掉重训 4 个模型**（这实际上是
一次功能类别留出，作者没这么框）；换一个目标不同的模型（Caliby）；把 frustration 最小化的突变写进
目标主链再重设计；用实验解出的突变体结构当目标排除建模假象；两套 frustration 参数复现全部主结论。

#### 一条新的线索，而且比这篇更对题

它引了一篇我们没查过的：

> "This is consistent with prior reports showing that **ProteinMPNN can substantially remodel functional
> sites, often impairing activity unless key residues are explicitly fixed**, as observed for the TEV
> protease (**Sumida et al., 2024**)."

**这才是 15c 前提最直接的出处**，而且听起来有活性实测。已派去读正文，读完补进来。
