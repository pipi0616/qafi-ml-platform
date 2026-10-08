# QAFI ML Platform

A cloud-native machine learning platform on AWS for **proteome-scale variant effect prediction**. It re-engineers [QAFI](https://doi.org/10.1007/s00439-024-02692-z) (ranked first in the CAGI7 LPL and ATP7B challenges) from research scripts into a production system:

- ingestion of public sources into a governed lakehouse
- feature computation for ~20,000 human proteins as massively parallel batch jobs
- leakage-proof training and a gated ML pipeline with model registry
- batch scoring of **~216 million** possible amino-acid substitutions
- a serverless prediction API
- a retrieval-augmented, tool-using agent on top

Everything runs in AWS and is defined in Terraform.

## Architecture

```
Sources ──▶ AWS S3 raw ──▶ Databricks: bronze → silver → gold ──▶ MLflow: train · register · batch score
           (Lambda validates                (dbt, Lakeflow Jobs,                          │
            and registers each file)         triggered by Airflow)                        ▼
                         AWS S3 lake ◀── gold predictions exported
                           ├─ Athena (SQL)
                           ├─ FastAPI on Lambda + API Gateway
                           └─ Bedrock: RAG + agent
```

## Stack

| Platform | What it does here | Services |
|---|---|---|
| **AWS** | storage of record, security, serving, GenAI | IAM, S3, KMS, Lambda, S3 events, ECR, API Gateway, Glue Catalog, Athena, Secrets Manager, CloudWatch, Bedrock |
| **Databricks** | data platform and ML | Unity Catalog, Delta Lake, Spark, SQL warehouse, dbt, Lakeflow Jobs, MLflow |
| Tooling | infrastructure and delivery | Terraform, GitHub Actions (OIDC), Docker, Airflow |

## Key design decisions

Full records are in [`docs/adr/`](docs/adr/).

| Decision | Rationale |
|---|---|
| Precompute all substitutions instead of real-time inference | The input space is enumerable (~216M) and features are expensive, so serving becomes a lookup |
| Databricks for data and ML, AWS for storage, serving and GenAI | The split most companies run: the data platform sits on top of the cloud account |
| Raw files land in S3 first, then Databricks | The cloud bucket is the landing zone and the system of record |
| A reproduction gate before model registration | A model is registered only if it reproduces the published benchmarks |

## Repository layout

```
docs/           plan, ADRs, architecture diagrams
notebooks/      guided build, grouped by layer: A_foundation · B_data_platform · C_ml · D_serving_ai · E_wrap_up
src/            qafi_core (science) · qafi_platform (engineering)
airflow/        DAG that triggers the Databricks job
dbt/            dbt-databricks project
databricks/     notebooks (.py source format) and job definitions run on Databricks
containers/     Dockerfiles per workload
infra/          Terraform modules and environments
tests/          unit · data · golden (published numbers)
```

## Status

In progress; see [`docs/plan.md`](docs/plan.md). Done: 00 account baseline, 01 Terraform, 02 engineering + CI/CD. Current: 04 Databricks lakehouse (Unity Catalog, Asset Bundle job: raw → bronze → silver with quality checks). Done before: 03 raw ingestion (S3 event → Lambda validator with a receipt registry) and 03b config-driven ingestion of public datasets (ClinVar, UniProt, ProteinGym, AlphaFold) straight from the publishers.
