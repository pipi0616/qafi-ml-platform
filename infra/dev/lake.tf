# The lake behind the API (notebook 10a): the released scores as parquet files in the lake bucket, described
# in the Glue Data Catalog so Athena can run SQL over them.
#
#   s3://<lake>/variant_effect/release=<release>/part-NN.parquet   ──┐
#   s3://<lake>/proteins/release=<release>/proteins.parquet         ──┼──▶ Glue: database qafi_dev_lake
#                                                                      │      tables variant_effect, proteins
#                                                                      ▼      (one partition per release)
#                                         Athena workgroup qafi-dev: SQL, results in the athena bucket
#
# The tables are only descriptions (columns, file format, where the files are); the data stays in S3
# ("schema on read"). Each release is added as a partition by notebook 10a, which also creates the views
# variant_effect_current and proteins_current that point at the release the API serves.

locals {
  lake_bucket = aws_s3_bucket.this["lake"]

  # Must match the parquet files written by the job "qafi lake export" (databricks/src/notebooks/export_lake.py).
  variant_effect_columns = [
    { name = "uniprot_id", type = "string", comment = "UniProt accession, e.g. P04637" },
    { name = "gene_name", type = "string", comment = "Gene symbol, e.g. TP53" },
    { name = "entry_name", type = "string", comment = "UniProt entry name, e.g. P53_HUMAN" },
    { name = "position", type = "int", comment = "Residue number in the UniProt sequence" },
    { name = "ref_aa", type = "string", comment = "Amino acid in the reference sequence" },
    { name = "alt_aa", type = "string", comment = "Amino acid after the substitution" },
    { name = "variant", type = "string", comment = "The substitution in one-letter code, e.g. R175H" },
    { name = "stability_score", type = "float", comment = "0 to 1, low = likely destabilising" },
    { name = "function_score", type = "float", comment = "0 to 1, low = likely loss of function" },
    { name = "zero_shot", type = "float", comment = "ESM-2 log-probability of the new amino acid minus the old one" },
    { name = "stability_head_version", type = "string", comment = "Registry version of the stability network" },
    { name = "function_trees_version", type = "string", comment = "Registry version of the function trees" },
  ]
  protein_columns = [
    { name = "uniprot_id", type = "string", comment = "UniProt accession" },
    { name = "gene_name", type = "string", comment = "Gene symbol" },
    { name = "entry_name", type = "string", comment = "UniProt entry name" },
    { name = "protein_name", type = "string", comment = "Recommended protein name" },
    { name = "length", type = "int", comment = "Residues in the UniProt sequence" },
    { name = "has_dms", type = "boolean", comment = "Has deep mutational scanning data in the platform" },
  ]
  lake_tables = {
    variant_effect = { columns = local.variant_effect_columns, description = "Released scores of every missense variant of the human proteome" }
    proteins       = { columns = local.protein_columns, description = "The scored proteins, one row each" }
  }
}

resource "aws_glue_catalog_database" "lake" {
  name        = "qafi_dev_lake"
  description = "qafi-ml-platform dev: tables over the lake bucket (notebook 10a)"
}

resource "aws_glue_catalog_table" "lake" {
  for_each = local.lake_tables

  name          = each.key
  database_name = aws_glue_catalog_database.lake.name
  description   = each.value.description
  table_type    = "EXTERNAL_TABLE" # dropping the table never deletes the files

  parameters = {
    EXTERNAL       = "TRUE"
    classification = "parquet"
  }

  partition_keys {
    name    = "release"
    type    = "string"
    comment = "Release of the scores: one folder per release, written once"
  }

  storage_descriptor {
    location      = "s3://${local.lake_bucket.bucket}/${each.key}/"
    input_format  = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"

    ser_de_info {
      serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
    }

    dynamic "columns" {
      for_each = each.value.columns
      content {
        name    = columns.value.name
        type    = columns.value.type
        comment = columns.value.comment
      }
    }
  }
}

# Every query runs in this workgroup: results go to the athena bucket (deleted after 7 days), encrypted with
# the platform key, and a query that would scan more than 5 GB is stopped, so one query costs at most about
# 2.5 cents (Athena bills 5 USD per TB scanned).
resource "aws_athena_workgroup" "lake" {
  name          = "qafi-dev"
  description   = "qafi-ml-platform dev: SQL over the lake (notebook 10a) and the API's queries (10b)"
  force_destroy = true # dev: destroy also removes the saved query history

  configuration {
    enforce_workgroup_configuration    = true # nobody can send results elsewhere or lift the limit
    publish_cloudwatch_metrics_enabled = true
    bytes_scanned_cutoff_per_query     = 5 * 1024 * 1024 * 1024

    engine_version {
      selected_engine_version = "Athena engine version 3"
    }

    result_configuration {
      output_location = "s3://${aws_s3_bucket.this["athena"].bucket}/results/"
      encryption_configuration {
        encryption_option = "SSE_KMS"
        kms_key_arn       = aws_kms_key.lake.arn
      }
    }
  }
}

output "lake" {
  description = "Where the released scores live and how to query them (notebooks 10a and 10b)"
  value = {
    bucket    = local.lake_bucket.bucket
    database  = aws_glue_catalog_database.lake.name
    workgroup = aws_athena_workgroup.lake.name
  }
}
