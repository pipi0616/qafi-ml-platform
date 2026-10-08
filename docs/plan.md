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
| 7 | **最终目标：用新流程重新预测全蛋白组**；训练数据和全蛋白组的特征都从 raw 重算 | 现有全蛋白组预测是旧方法算的，不一定对，只作对照（09 号新旧对比）。2 万个蛋白的结构、MSA、ESM 特征计算量很大，07 号先评估免费额度能撑住哪些、怎么分批 |
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
| 07 | Databricks | 特征和实验：从 raw 算特征（AlphaFold 结构、序列；先训练集的蛋白，再评估全蛋白组怎么算），和论文用的版本对照；科学代码搬进 `src/qafi_core`（黄金测试保证没搬坏）；防泄漏交叉验证；每次实验记进 MLflow | MLflow |
| 08 | Databricks | 训练 → 质量闸门（复现不了论文数字就不注册）→ 模型注册 | MLflow、Unity Catalog 模型 |
| 09 | Databricks | **用新模型给全蛋白组重新预测**（批量打分），和旧的全蛋白组预测逐个对照（04 号准备好的 old_predictions 任务）；漂移检查和告警 | MLflow、Spark |
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
- [ ] **06 · 编排** ← 当前（2026-10-08 写好并试通，由用户在 06 号运行）
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
