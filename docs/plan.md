# QAFI ML Platform：最终方案与学习计划

> **一句话目标**：把 QAFI 从"一堆 Python 脚本"重建成一个**按公司标准搭建、全部跑在 AWS 上的机器学习平台**。它覆盖数据接入、湖仓、蛋白组规模的特征计算、训练、模型管理、批量打分、API 服务、RAG 和 agent，以及监控和治理。你一边搭建，一边学完 Data Engineer、ML Engineer、AI Engineer 和云架构需要的全部核心技能。

---

## 0 · 一条规则

> **只学公司现在实际在用的工具和做法，只学数据 / ML / AI 工程师岗位要做的那部分。** 不求全。每加一个工具之前先问：主流公司在用吗？是工程师的活吗？两个答案有一个是"否"，就不学，最多一句话带过。

公司里是分工的：**平台 / 云团队**开账号、搭平台、管安全基线；**数据 / ML 工程师**在搭好的平台上接数据、做转换、训练、部署、监控。00、01 号是平台团队的活，做一遍懂原理就够，不再深挖。

## 1 · 定稿：两个平台，各学重点

> **2026-10-07 定稿。之后除非你要求，不再改架构。**

| | **AWS** | **Databricks** |
|---|---|---|
| 在公司里是什么 | 云：数据实际存放的地方、权限、对外服务、GenAI | 数据平台：工程师处理数据、训练模型的地方 |
| 学的重点 | IAM、S3、KMS、Terraform、OIDC、ECR、Lambda、S3 事件、API Gateway、Glue Catalog + Athena、Secrets Manager、CloudWatch、Bedrock | Unity Catalog、Delta、Spark、SQL warehouse、dbt、Lakeflow Jobs、MLflow |
| 章节 | 00–03、10–13 | 04–09 |

**数据怎么流动**：

```
数据源 ──03──▶ AWS S3 raw ──04──▶ Databricks: bronze → silver → gold ──07–09──▶ MLflow 训练 / 注册 / 打分
  （Lambda 自动校验、登记）        （05 dbt，06 Jobs + Airflow 编排）                  │
                                                                                      ▼
                       AWS: S3 lake ◀──10──  gold 预测表导出
                         ├─ Athena 用 SQL 查（10）
                         ├─ Lambda + API Gateway 对外提供查询（10）
                         └─ Bedrock：RAG + Agent（11、12）
```

**不学的**（公司里有，但不是这个岗位的重点，或和 Databricks 重复）：SageMaker（机器学习放在 Databricks 做；概念一样，MLflow 在 SageMaker 里也有）、EMR、Redshift、Kinesis、Step Functions、Lake Formation、Organizations。

## 2 · 关键决定

| # | 决定 | 为什么 |
|---|---|---|
| 1 | AWS 区域 eu-central-1（法兰克福） | 离巴塞罗那近、服务全 |
| 2 | AWS 资源用 Terraform 建，CI 用 OIDC 访问 AWS | 业界标准，不存长期密钥 |
| 3 | 原始文件先进 AWS S3 raw，再进 Databricks | 公司里的"进货口"就在云存储；原始数据归自己所有 |
| 4 | 数据处理和机器学习在 Databricks Free Edition | 公司用得最多的数据平台；免费版功能够用 |
| 5 | 预测结果批量算好，API 只查表 | 输入空间可枚举（约 2 亿个替换），特征计算重 |
| 6 | 对外 API 和 GenAI 放在 AWS | 公司里对外服务一般不放在数据平台里；Bedrock 是 AWS 上 GenAI 的标准入口 |
| 7 | **最终目标：用新流程重新预测全蛋白组**；训练数据和全蛋白组的特征都从 raw 重算 | 现有全蛋白组预测是旧方法算的，不一定对，只作对照（09 号新旧对比）。**2026-10-08 你决定：特征按公司标准重写，加入深度学习，不搬旧代码**（旧代码依赖 PSI-BLAST、Arpeggio 和复现不了的统计表，还有泄漏问题）。新特征：ESM-2 语言模型 + AlphaFold 结构 + 理化，全蛋白组也能算 |
| 8 | 所有原始数据只从官方来源进 raw，版本写死 | 可复现、可追溯；以前自己整理的文件只当参照答案，不当数据源 |

## 3 · 文件夹结构和三条工作规则

```
qafi-ml-platform/
├── README.md                 项目总览（英文，给招聘方看）
├── docs/plan.md              ← 本文件；docs/adr/ 架构决策记录
├── notebooks/                学习主线，按层分组
│   ├── A_foundation/         00–02（本机）
│   ├── B_data_platform/      03–06
│   ├── C_ml/                 07–09（Databricks）
│   ├── D_serving_ai/         10–12（AWS）
│   └── E_wrap_up/            13
├── src/qafi_platform/        工程代码：接入、指标、训练、推理、服务
├── src/qafi_core/            科学代码：07 号起从 QAFI_CODE_NEW 搬入，不改算法
├── databricks/               在 Databricks 上运行的 notebook 和 Job 定义
├── dbt/                      dbt-databricks 项目
├── airflow/                  触发 Databricks Job 的 DAG
├── containers/               Dockerfile
├── infra/dev/                Terraform：所有 AWS 资源
├── tests/                    unit · golden（论文数字）
└── .github/workflows/        CI/CD
```

**三条规则**：notebook 负责讲解和执行；稳定的代码搬进 `src/`；AWS 资源只由 Terraform 创建。

## 4 · 学习路线：15 个 notebook（03 分成 03 和 03b）

| # | 平台 | 内容 | 重点技能 |
|---|---|---|---|
| 00 ✅ | AWS | 账号安全基线 | IAM、MFA、Budgets、CloudTrail |
| 01 ✅ | AWS | 用 Terraform 建 4 个桶 + 加密密钥 + CI 的身份 | Terraform、S3、KMS |
| 02 | AWS + GitHub | 工程化代码库 + CI/CD；论文数字写成黄金测试（0.480 / 0.914） | GitHub Actions、OIDC、IAM 角色、Docker、pytest |
| 03 | AWS | 数据接入（大门）：86 GB 压缩包分块上传到 S3 raw；每个新文件到达时，S3 事件自动触发 Lambda 校验并登记 | S3、Lambda、S3 事件、CloudWatch Logs |
| 03b | AWS | 公开数据进货：按数据源清单（`config/sources.toml`）由送货员 Lambda 从官方直接搬进 raw：ClinVar、UniProt、ProteinGym、AlphaFold v4/v6；版本写死、幂等 | Lambda、HTTP Range、S3 分块上传 |
| 04 | Databricks | 湖仓：raw 文件进 Unity Catalog → bronze Delta 表 → silver。主线是**训练集**（ProteinGym DMS：一个实验里的一个单点突变一行，核对序列）和参考数据（UniProt、ClinVar）；旧的全蛋白组预测做成单独的任务，09 号再跑。一切从 raw 出发，以前自己整理的文件只当参照答案 | Unity Catalog、Delta、Spark、Asset Bundles |
| 05 | Databricks | dbt：gold 层星型模型（`dim_protein`、`dim_dms_assay`、`fct_dms_measurement`）；DMS 位置换算成 UniProt 编号（AlphaFold 用的编号）；单元测试 + 数据测试；说明写进 Unity Catalog；CI 跑 `dbt parse` | dbt-databricks |
| 06 | Databricks + AWS | 编排：Lakeflow Jobs 串起 04–05；本机 Airflow 定时触发这个 Job，并按数据源清单的 `refresh` 定时进新版本；Databricks 令牌存在 AWS Secrets Manager | Lakeflow Jobs、Airflow、Secrets Manager |
| 07 | Databricks | 特征和基线，一次一小步：**07a** 设计（特征清单、考试规则，锁住测试蛋白）→ **07b** 新数据 MaveDB（整库打包进 raw 和 bronze）→ **07c** 标签（同义突变 = 1、无义突变 = 0 两把刻度）→ **07d** AlphaFold 结构特征和图 → **07e** ESM-2 语言模型特征 → **07f** 特征表 + 树模型基线 + MLflow，和旧论文对照（含加不加 Domainome 的对照实验）。新代码写在 `src/qafi_core` | 特征工程、分组交叉验证、dbt seed、数据接入、MLflow |
| 08 | Databricks | 深度学习：**08a** PyTorch 入门（ESM 向量 + 小网络）→ **08b** 图神经网络（AlphaFold 结构图）→ **08c** 微调 ESM-2（LoRA，GPU）；同一套考试规则，必须比 07e 的基线好 | PyTorch、PyTorch Geometric、Hugging Face、LoRA、MLflow |
| 09 | Databricks | 选模型 → 质量闸门（测试蛋白上的成绩）→ Unity Catalog 模型注册 → **用新模型给全蛋白组重新预测**（批量打分），和旧的全蛋白组预测逐个对照（04 号准备好的 old_predictions 任务）；漂移检查和告警 | MLflow、Unity Catalog 模型、Spark |
| 10 | AWS | 服务：gold 预测表导出到 S3 lake；Glue Catalog + Athena 用 SQL 查；FastAPI 容器部署到 Lambda + API Gateway | Athena、Glue、ECR、Lambda、API Gateway |
| 11 | AWS | RAG：Bedrock Knowledge Base 收录 UniProt 注释和你的论文 | Bedrock |
| 12 | AWS | Agent：调用预测 API、知识库等工具；Guardrails；评估 | Bedrock Agents、Guardrails |
| 13 | 两边 | 收尾：CloudWatch 告警和成本；架构图、系统设计讲解稿、简历条目 | CloudWatch |

## 5 · 时间表

按顺序做完为止。参考节奏（每周 15–20 小时）：A 两周，B 三周，C 三周，D 两周，E 一周，约 **11 周**。**做完 10 号就开始投简历**，边面试边做。

## 6 · 成本与安全

**额度**：2025 年 7 月以后注册的新账号，注册时送 100 美元额度，完成几个指定任务最多再送 100 美元。选 **Free plan** 的话，6 个月内额度用完或到期，账户会自动关闭，**不会扣你信用卡的钱**。建议先用 Free plan；如果某个服务在 Free plan 下用不了，再升级。

**四条规则**：
1. **费用报警先于一切**（00 号的第一件事）：10 / 25 / 50 美元各设一档。
2. **只用按量计费的服务**：没有常开的数据库、集群或 NAT。
3. **计算资源用完就删**：`terraform destroy`，数据桶保留。
4. **大计算先试点**：08、09 号先跑 100 个蛋白，按实际花费估算全量成本，你确认后才全量运行。

**预计花费**：存储加查询每月几美元；SageMaker 训练和批量打分按小时计费，每次几美元；Bedrock 按调用量计费。全蛋白组特征不重算，所以没有大额计算费。

### Free plan 的限制和对策

| 限制 | 影响 | 对策 |
|---|---|---|
| 不能用 AWS Organizations / IAM Identity Center，一旦加入额度立即作废 | 00 号的登录方式；没法做多账号 | IAM 用户 + MFA + `aws login` 临时凭证；一个账号按名字分环境 |
| Bedrock 不支持跨区域推理 | 11、12 号的模型选择 | 11 号开始前确认法兰克福本地可用的模型 |

---

## 7 · 每部分对应什么岗位

| 岗位 | 主要证据 |
|---|---|
| Data Engineer | 03 S3 + Lambda 接入；04–06 Databricks 湖仓、dbt、Jobs + Airflow |
| ML Engineer / MLOps | 07–09 MLflow 实验、注册、批量打分；10 号 API |
| Data Scientist | 07 号防泄漏验证 + 论文本身 |
| AI Engineer | 11 号 RAG + 12 号 Agent（Bedrock） |
| Cloud / AI Architect | 00–02 IAM、Terraform、CI/CD + 10 号 AWS 服务化 + 13 号架构材料 |

**这套架构本身和生物无关**：接入 → 湖仓 → 训练 → 批量打分 → 服务 → 监控。把 QAFI 换成股票数据，就是一个量化因子平台。

## 8 · 进度

- [x] AWS 账号已注册（Free plan）
- [x] GitHub 已登录（`pipi0616`）
- [x] 全蛋白组预测：`QAFI_ALL/exhaustive_mutagenesis_data_v1.0.tar.gz`（20,521 个蛋白，86 GB 解压；**不要在本地解压**）
- [ ] 全蛋白组的 28 个特征：有没有现成的？（压缩包是在用户名为 `frog` 的机器上打包的，可能是实验室服务器）
- [x] 节奏：按顺序做完为止
- [x] **00 · 云账号与安全基线**（2026-10-07 完成：MFA、预算报警、账户级 S3 禁止公开）
- [x] **01 · Terraform**（2026-10-07 完成：21 个资源已 apply）
- [x] **02 · 工程化代码库 + CI/CD**（2026-10-08 完成：CI 5 项全绿，OIDC 登录 AWS 成功；修过两次红：`setup-uv@v10` 标签不存在、GitHub 改用带永久编号的 OIDC 写法）
- [ ] **09c · 全蛋白组打分**（2026-10-14 写好并试通，待用户运行并提交：`gold.dim_residue_proteome`（和 `dim_residue` 共用 dbt 宏 `residue_features`，20,652 个蛋白、11,456,418 个残基，一个不少）下载到 Mac（1.44 GB；`lakehouse.download` 断网会重试半小时、写完才改名）；突变特征在 Python 里重算，和 SQL 在 84 万个训练突变上逐个比对（最大差 3e-5，float32 舍入）；`python -m qafi_platform.scoring` 在后台按 `@champion` 从注册表取模型，ESM 向量边算边用、不存，GPU 形状固定，每 50 个蛋白存一份、可续跑，内存压力到"危急"就卸载模型让路（开着 VS Code 的 Mac 常年"警告"，所以不等"正常"）。实测 300 个蛋白：每秒 582 个残基，突变数 = 残基 × 19 一个不少，**全量约 5.5 小时**、结果约 3.4 GB；上传逐个文件、可续传（`scoring.upload`），装成 `ml.proteome_scores`（每行带版本和模型版本），产品视图 `gold.fct_variant_effect`（加基因名）。TP53 试跑：结构性热点 R175H 判有害，DNA 接触热点 R248Q、R273H 判不出来（模型主要学稳定性），写进讲义。外部验证 `gold.fct_clinvar_missense`：只用我们数据之外的蛋白，16.8 万个突变（4.9 万致病）；300 个蛋白的预览：功能分 AUROC 0.92、AlphaMissense 0.97。新旧对比预览：每个蛋白内和旧 QAFI2 相关 0.71）
- [ ] **09b · 模型注册**（2026-10-14 写好并试通，待用户运行并提交：Unity Catalog 注册表 `qafi_dev.ml.stability_head`、`qafi_dev.ml.function_trees`；只有 09a 的考试过了闸门才注册，版本带考试记录的标签，`@champion` 别名指向上线版本；换算表跟着运行走（`registry.distributions`）；回滚 = 挪别名。试通时注册的版本都已删除）
- [ ] **09a · 定版和考试**（2026-10-14 写好并试通，待用户运行并提交：Optuna 调稳定性网络（隐藏层、dropout、学习率、轮数、权重衰减，第 0 次试验用现有设置）；在全部训练蛋白上训练最终模型，分数换成百分位（换算表来自交叉验证预测，1,001 个分位点）；旧预测全量装表（约 1 小时，EVE/popEVE 冲突记为空）；**测试蛋白只打开一次**（MLflow 记 `test_set=opened`，再开会报错），三条上线规则（各自超过一半的实验更好且 p < 0.05）；同题对照 AlphaMissense、EVE、popEVE、CPT-1、旧 QAFI2。试跑（3 次试验）最好 0.631；功能集成交叉验证 0.432）
- [ ] **08c · 微调 ESM-2（LoRA）** ← 当前（2026-10-14 写好并试通，待用户运行并提交：用户选了小模型。实测 Mac 内存：35M 带 LoRA 训练 3.8 GB，150M 要 9.6 GB，所以用 ESM-2 35M（`facebook/esm2_t12_35M_UR50D`，版本钉死）。`qafi_core.finetune`：每组实验测过的一段加上下文切成 256 个记号的片段（570 段），8 段一批、形状固定（避开 08b 的 Apple GPU 内存问题），答案放在 256 × 20 的格子表里；LoRA 秩 8，贴在注意力和前馈矩阵上，只训练 83 万个参数（2.4%）；冻结的那个只读一次存起来。**结果**（同一个小模型、同样的折）：零样本 0.476，冻结 0.581，LoRA 0.599；LoRA 对冻结：稳定性 +0.014（p < 0.001），**功能 −0.014（p = 0.01）**；微调的小模型三类数据都不如冻结的 650M 大模型（0.599 对 0.627）；LoRA 慢 30 多倍。加进功能集成 +0.003（p = 0.03），显著但不值得，不加。**08 章结论：最终模型不变**——稳定性用 08a 的 `mlp_standardised`（0.65），功能用 08a 的 `ensemble`（0.43）；瓶颈是功能数据只有 29 个训练蛋白。下一步 09）
- [ ] **08b · 图神经网络：AlphaFold 结构图**（2026-10-13 写好并试通，待用户运行并提交：PyTorch Geometric，GraphSAGE 消息传递（`qafi_core.graph`），在 08a 的网络中间插图层，0 层即退回 08a；图只留离有标签残基两步以内的部分（29.3 万个节点裁到 5 万，预测不变，有测试）；在 CPU 上训练，因为 Apple GPU 对大小不一的批次会无限缓存、内存涨到 30 GB。**2 × 2 对照**：有 ESM 向量时看邻居反而变差（0.626 → 0.615，功能 −0.033）；没有 ESM、只有结构时看邻居有用（0.529 → 0.558，Domainome +0.032，81% 的实验更好）。**探针实验**：一条直线就能从 ESM 向量里读出 rsa 0.87、接触数 0.90、pLDDT 0.91，所以图层没带来新信息。结构图网络加进集成，功能不变（0.437 对 0.432，p = 0.25）。**结论：最终模型不变**（稳定性用 08a 的网络，功能用 08a 的集成）；要再提高，得改进向量本身或找更多功能数据。同时把 08a 讲义里第 1、3 步改成了更直白的版本。下一步 08c 微调 ESM-2）
- [x] **08a · PyTorch 入门：ESM 向量 + 小网络**（2026-10-13 完成，用户提交 dbb190b，CI 全绿：冻结的 ESM-2 后面接一个 73 万参数的小网络（`qafi_core.neural`：LayerNorm → 512 → 128 → 20 个输出，每种氨基酸一个），在 Mac 的 GPU 上训练，5 折不到一分钟；训练数据用 `lakehouse.download` 从平台取（Arrow，只取 `split = 'train'`），向量拼成 29 万 × 1,280 的 float16 矩阵；本机训练、平台记账（`qafi_platform.tracking`：MLflow、`ml.cv_scores`、`ml.cv_predictions`，带代码版本和 `-dirty` 标记）；新增存储区 `ml.files`。学习曲线：训练损失一直降，没见过的蛋白上的成绩第 4 轮见顶后下滑，轮数定在 4。**结果**（同一批折、和基线配对比较）：只用向量 0.585，加 16 个特征 0.593，每组实验的标签各自标准化 0.627；Domainome 0.652 对基线 0.572，显著更好；**功能 0.378，比基线和零样本都差一点**。三个预测（零样本、树、网络）按排名平均：功能 **0.432**，比零样本高 0.024，67% 的功能实验更好，p = 0.001，07f 定的 0.40 过线。结论：两个问题两个答案，稳定性用网络，功能用集成。下一步 08b 图神经网络）
- [x] **07f · 特征表 + 基线模型 + MLflow**（2026-10-13 完成，用户提交 ac5cd53，CI 全绿：特征表 `gold.fct_variant_features`（107 万个突变、16 个特征、4 组，无缺失）和 `gold.dim_residue`，dbt 建表 31 项检查、含 2 个 SQL 单元测试，整个项目 PASS=84；两张查找表 `amino_acid`、`blosum62` 由 `qafi_core.features` 生成并有测试对账；基线 XGBoost 3.4.1（`qafi_core.baseline`）：分组 5 折、每组实验一票、每组实验的 Spearman 取中位数、配对 Wilcoxon 比较；新流水线 `qafi baseline model`（features → train_baseline）、新的一层 `ml`（`cv_scores`、`cv_predictions`）、MLflow 实验 `qafi-variant-effect`，一次运行记 8 页（零样本、基线、6 个对照），每页带代码版本、特征表版本、参数。**结果**：整体 0.550（零样本 0.455）；Domainome 0.572 对 0.461、ProteinGym 稳定性 0.583 对 0.468，都显著更好；**功能数据 0.409 对 0.398，和零样本没有差别（p = 0.42）**。对照：三组特征单用都弱、合起来才强；Domainome 留在训练里（不拖累功能，稳定性离不开它）；周围环境那一组没有证明自己有用。和旧论文同题对照（14 个蛋白、8.9 万个突变）：QAFI2 0.461、新基线 0.454、ESM-2 零样本 0.502，统计上打平。07a 清单里的二级结构、Δ接触能量没做。08 的目标：功能数据超过 0.40）
- [x] **07e · 语言模型特征：ESM-2**（2026-10-13 完成，用户提交 77aa33d，CI 全绿；2026-10-10 写好、全蛋白组算完并装进表：ESM-2 650M 零样本分数 + 残基向量；模型版本写死进 raw（验货员认 .safetensors）；Databricks 免费版 GPU 配额 = 0（gpu-quota-denied），改在 Mac 的 GPU 上回填、CPU 任务装表（同代码/版本/全精度，装表幂等）；silver 三张表：全蛋白组 20,652 个蛋白 1,145.6 万残基的概率、457 个有标签蛋白的向量；和 ProteinGym 公布的 ESM2 650M 成绩差 0.02 以内；和标签相关性中位数 0.40–0.47）
- [x] **07d · 结构特征和图：AlphaFold**（2026-10-09 完成，用户提交 94541b9，CI 6 项全绿；全蛋白组名单核对过：UniProt 2026_03 参考蛋白组 20,652 个 = 官网基因数，AlphaFold v6 整包 23,586 个结构 = 官网数字，两边都有 20,511 个；缺的 141 个都是 AlphaFold 本来没有的（不到 30 个氨基酸的免疫基因片段、硒蛋白、v6 之后新增或改序列的）。用户决定不重跑。内容：新流水线 `structures`（copy_raw → bronze_alphafold → silver_structure），AlphaFold v6 23,586 个模型 / 20,550 个蛋白进 bronze；每个残基的 pLDDT、RSA、接触数、半球暴露度、10 Å 邻居；长蛋白按序列拼接；增量、分批保存、可续跑。加速：先量（半球暴露度占 78%），表面积和半球暴露度改成 numpy/scipy 矩阵运算（和 Biopython 逐个残基核对一致），每个蛋白 1.39 秒 → 0.20 秒；按蛋白切成小任务；全蛋白组约 35 分钟，原来要 5 个多小时。TTN 166 段撑爆内存（OOM），改成先定位、每段只留自己负责的残基，内存减半、结果不变。结果：20,467 个蛋白 ok（99.8%）、1,141 万个残基；有标签的 107 万个突变 100% 有结构特征）
- [x] **07c · 标签**（2026-10-08 完成，用户提交 d45cab3：同义突变 = 1、无义突变 = 0（中位数）；ΔΔG 稳定性实验野生型 = 1；silver_mavedb 226.6 万个蛋白突变，编号吻合 99.5%；gold `dim_label_dataset`（每组实验的刻度和 status）、`fct_variant_label`：457 个蛋白、107 万个标签；dbt PASS=53；和旧标签对照 Pearson 中位数 0.90。**新蛋白的分边 2026-10-10 补做**：新增 414 个有标签蛋白进 `protein_split`（共 494 个蛋白、294 个相似组），锁定的 80 行一行没改；分组改用代表法（CD-HIT/MMseqs2 的贪心算法，成员只和代表比），因为原来的连锁分法会顺着多结构域蛋白把锌指、同源框、激酶连成一个 200 个蛋白的大组，整组挂在已锁定的 CBX4 下面，测试集直接涨到 52%；改完测试集按蛋白 23.5%、按标签条数 21.3%，Domainome、其他 MaveDB、ProteinGym 三类各占两成上下；检查 `assert_every_dms_protein_has_a_split` 改名 `assert_every_labelled_protein_has_a_split`，现在管 DMS 和标签两边的蛋白；dbt PASS=53）
- [x] **07b · 新数据 MaveDB**（2026-10-08 完成，用户提交 cf08b66：Zenodo 整库打包 2026-06-25 版 1.91 GB 进 raw；bronze 三张表：2,777 组实验、820 万分数、321 万条 VEP 注释；人类当前蛋白实验 1,119 组，其中 Domainome 503 组/388 个蛋白，其他 616 组/74 个蛋白）
- [x] **07a · 设计**（2026-10-08 完成，用户亲手运行并提交 1c790b9：旧 28 个特征逐个评估；考试规则；测试蛋白 15 个锁成 dbt seed `protein_split`）
- [x] **06 · 编排**（2026-10-08 完成，用户亲手运行并提交 3b8a255，CI 6 项全绿含新的 `airflow` 检查；第一次由机器身份 `qafi-airflow` 跑通整条：进新版本 → 放提货单 → Databricks 6 步）
  - Lakeflow Job `training_data` 末尾加了 dbt 任务 `gold`，在 Job 里 PASS=25。Job 带版本参数，`copy_raw` 失败重试，超时 1 小时，失败发邮件
  - Airflow 3.3 在单独的环境里（`airflow/`），DAG `qafi_refresh` 每月 10 号运行：进新版本 → 放提货单 → 启动 Job。本机试跑 3 个任务全过，Job 6 步全过
  - Airflow 用机器身份 `qafi-airflow`（OAuth M2M）进 Databricks，密钥在 AWS Secrets Manager（`airflow/connections/databricks_default`），90 天轮换；只能启动这一个 Job、往 raw 文件区放提货单
  - `config/sources.toml`：`refresh = "monthly"`（ClinVar，自动）或 `"manual"`（改版本号要审核）
- [x] **05 · dbt gold**（2026-10-08 完成，用户亲手运行并提交 3774598，CI 全绿：`dbt build` PASS=25（4 个 staging 视图、3 张 gold 表、1 个单元测试、17 个数据测试）；16% 的 DMS 位置编号和 UniProt 不一致，97.9% 换算成功、0 错，2.1% 留空；dbt 用 `databricks auth token` 的一小时令牌）
- [x] **04 · Databricks 湖仓**（2026-10-08 完成，用户亲手运行并提交 0714e8e，CI 全绿：`databricks/` bundle 两个任务：`training_data`（ProteinGym + UniProt + ClinVar，约 4 分钟，6 条质检全过：69.6 万个单点突变、217 个实验、80 个人类蛋白）和 `old_predictions`（旧全蛋白组预测，20 个蛋白试通，09 号再跑）；由用户在 04 号亲手运行 training_data）
- [x] **03b · 公开数据进货**（已写好：`config/sources.toml` 7 个文件，送货员 Lambda `qafi-dev-raw-fetcher`，`infra/dev/fetch.tf`）
- [x] **03 · 数据接入**（已写好：S3 事件 → Lambda `qafi-dev-raw-validator` 验货、`_registry/` 收货单、分块上传 + CRC64NVME 指纹核对；`infra/dev/ingest.tf` 新增 8 个资源）
- 2026-10-07：计划按「只学公司标准做法」的规则从 22 个 notebook 缩到 14 个
- 2026-10-07：**定稿**：AWS（00–03、10–13）+ Databricks（04–09），各学重点；01 号已 apply（21 个资源）
