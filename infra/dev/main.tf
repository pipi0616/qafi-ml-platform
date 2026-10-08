# The platform's AWS storage: four S3 buckets and the KMS key that encrypts them.
# The data platform itself runs on Databricks (Unity Catalog, Delta tables); these
# buckets hold what lives outside it.
#
#   raw        original files exactly as received, the system of record
#              (the proteome tarball, UniProt dumps, ClinVar)
#   lake       tables exported from Databricks for the prediction API
#   artifacts  model exports and the documents behind the GenAI knowledge base
#   athena     scratch space for SQL over the exported tables; results expire after 7 days
#
# Run from this folder:  terraform init  ->  terraform plan  ->  terraform apply

terraform {
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
    archive = {
      source  = "hashicorp/archive" # zips the Lambda code in ingest.tf
      version = "~> 2.8"
    }
  }
}

provider "aws" {
  region = "eu-central-1"

  # Every resource gets these tags: the cost report can be filtered by them.
  default_tags {
    tags = {
      project     = "qafi-ml-platform"
      environment = "dev"
      managed_by  = "terraform"
    }
  }
}

data "aws_caller_identity" "current" {}

locals {
  # Bucket names are global across all AWS accounts, so the account ID makes them unique.
  prefix  = "qafi-${data.aws_caller_identity.current.account_id}-dev"
  buckets = toset(["raw", "lake", "artifacts", "athena"])
}

# ---------------------------------------------------------------------------
# Encryption key shared by the four buckets (1 USD / month)
# ---------------------------------------------------------------------------
resource "aws_kms_key" "lake" {
  description         = "qafi-ml-platform dev: data lake encryption key"
  enable_key_rotation = true
}

resource "aws_kms_alias" "lake" {
  name          = "alias/qafi-dev-lake"
  target_key_id = aws_kms_key.lake.key_id
}

# ---------------------------------------------------------------------------
# Buckets: for_each creates one of each resource below per bucket name
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "this" {
  for_each = local.buckets

  bucket = "${local.prefix}-${each.key}"
}

# Nobody on the internet can read these buckets, whatever else is configured.
resource "aws_s3_bucket_public_access_block" "this" {
  for_each = aws_s3_bucket.this

  bucket                  = each.value.id
  block_public_acls       = true
  ignore_public_acls      = true
  block_public_policy     = true
  restrict_public_buckets = true
}

# Everything written is encrypted with the key above.
resource "aws_s3_bucket_server_side_encryption_configuration" "this" {
  for_each = aws_s3_bucket.this

  bucket = each.value.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.lake.arn
    }
    bucket_key_enabled = true # keeps the per-request KMS cost near zero
  }
}

# Overwritten or deleted files can be recovered (the athena bucket is scratch, so not there).
resource "aws_s3_bucket_versioning" "this" {
  for_each = setsubtract(local.buckets, ["athena"])

  bucket = aws_s3_bucket.this[each.key].id
  versioning_configuration {
    status = "Enabled"
  }
}

# Athena query results are deleted after 7 days.
resource "aws_s3_bucket_lifecycle_configuration" "athena" {
  bucket = aws_s3_bucket.this["athena"].id

  rule {
    id     = "expire-query-results"
    status = "Enabled"
    filter {}
    expiration {
      days = 7
    }
  }
}

output "bucket_names" {
  description = "Bucket name per purpose; later notebooks read these instead of hard-coding names"
  value       = { for name, b in aws_s3_bucket.this : name => b.bucket }
}

output "kms_key_arn" {
  value = aws_kms_key.lake.arn
}
