# The knowledge base for question answering (notebook 11), in Ireland: the free plan cannot use cross-Region
# inference, and Claude runs inside one Region in Ireland but not in Frankfurt (decision 9, docs/plan.md).
# Every resource here says region = local.knowledge_region; everything else stays in Frankfurt.
#
#   documents ──▶ s3://<prefix>-knowledge/uniprot/<release>/<accession>.md   (+ .metadata.json)
#                 s3://<prefix>-knowledge/papers/...
#                      │  ingestion: each document cut into passages, each passage turned into
#                      │  1,024 numbers by Titan Text Embeddings V2
#                      ▼
#                 S3 Vectors: vector bucket qafi-dev-knowledge, index passages
#                      │  retrieve: the passages closest in meaning to a question
#                      ▼
#                 notebook 11 ──▶ Claude (bedrock-mantle endpoint) answers from them, citing them

locals {
  knowledge_region    = "eu-west-1"
  embedding_model_arn = "arn:aws:bedrock:${local.knowledge_region}::foundation-model/amazon.titan-embed-text-v2:0"
}

# ---------------------------------------------------------------------------
# The documents
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "knowledge" {
  region        = local.knowledge_region
  bucket        = "${local.prefix}-knowledge"
  force_destroy = true # dev: every document is rebuilt from raw by notebook 11
}

resource "aws_s3_bucket_public_access_block" "knowledge" {
  region                  = local.knowledge_region
  bucket                  = aws_s3_bucket.knowledge.id
  block_public_acls       = true
  ignore_public_acls      = true
  block_public_policy     = true
  restrict_public_buckets = true
}

# Public sources only (UniProt, your published paper): S3's own encryption is enough here. The platform key
# (main.tf) is for our own data, and a key is regional: a second one would cost another 1 USD a month.
resource "aws_s3_bucket_server_side_encryption_configuration" "knowledge" {
  region = local.knowledge_region
  bucket = aws_s3_bucket.knowledge.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# ---------------------------------------------------------------------------
# Where the passages and their vectors are kept: S3 Vectors, paid per use, nothing running
# ---------------------------------------------------------------------------
resource "aws_s3vectors_vector_bucket" "knowledge" {
  region             = local.knowledge_region
  vector_bucket_name = "qafi-dev-knowledge"
  force_destroy      = true
}

resource "aws_s3vectors_index" "passages" {
  region             = local.knowledge_region
  vector_bucket_name = aws_s3vectors_vector_bucket.knowledge.vector_bucket_name
  index_name         = "passages"
  data_type          = "float32"
  dimension          = 1024     # what Titan Text Embeddings V2 produces
  distance_metric    = "cosine" # similar meaning = vectors pointing the same way

  # Bedrock stores each passage's text and its document's metadata next to the vector; too large to filter on.
  metadata_configuration {
    non_filterable_metadata_keys = ["AMAZON_BEDROCK_TEXT", "AMAZON_BEDROCK_METADATA"]
  }
}

# ---------------------------------------------------------------------------
# The knowledge base's identity: read the documents, embed with Titan, write and search the index
# ---------------------------------------------------------------------------
resource "aws_iam_role" "knowledge" {
  name = "qafi-dev-knowledge"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "bedrock.amazonaws.com" }
      Action    = "sts:AssumeRole"
      Condition = { # only knowledge bases of this account may use the role
        StringEquals = { "aws:SourceAccount" = data.aws_caller_identity.current.account_id }
        ArnLike      = { "aws:SourceArn" = "arn:aws:bedrock:${local.knowledge_region}:${data.aws_caller_identity.current.account_id}:knowledge-base/*" }
      }
    }]
  })
}

resource "aws_iam_role_policy" "knowledge" {
  name = "read-documents-embed-index"
  role = aws_iam_role.knowledge.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "EmbedWithTitan"
        Effect   = "Allow"
        Action   = "bedrock:InvokeModel"
        Resource = local.embedding_model_arn
      },
      {
        Sid      = "ReadTheDocuments"
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:ListBucket"]
        Resource = [aws_s3_bucket.knowledge.arn, "${aws_s3_bucket.knowledge.arn}/*"]
      },
      {
        Sid      = "WriteAndSearchTheIndex"
        Effect   = "Allow"
        Action   = ["s3vectors:PutVectors", "s3vectors:GetVectors", "s3vectors:DeleteVectors", "s3vectors:QueryVectors", "s3vectors:GetIndex"]
        Resource = aws_s3vectors_index.passages.index_arn
      },
    ]
  })
}

# ---------------------------------------------------------------------------
# The knowledge base and its two sources
# ---------------------------------------------------------------------------
resource "aws_bedrockagent_knowledge_base" "proteins" {
  region      = local.knowledge_region
  name        = "qafi-dev-knowledge"
  description = "UniProt annotations of the reviewed human proteins and the QAFI paper (notebook 11)"
  role_arn    = aws_iam_role.knowledge.arn

  knowledge_base_configuration {
    type = "VECTOR"
    vector_knowledge_base_configuration {
      embedding_model_arn = local.embedding_model_arn
      embedding_model_configuration {
        bedrock_embedding_model_configuration {
          dimensions          = 1024
          embedding_data_type = "FLOAT32"
        }
      }
    }
  }

  storage_configuration {
    type = "S3_VECTORS"
    s3_vectors_configuration {
      index_arn = aws_s3vectors_index.passages.index_arn
    }
  }

  depends_on = [aws_iam_role_policy.knowledge]
}

# One document per protein, all alike: cut every 300 tokens, 20% overlap so no sentence is lost at a cut.
resource "aws_bedrockagent_data_source" "uniprot" {
  region               = local.knowledge_region
  knowledge_base_id    = aws_bedrockagent_knowledge_base.proteins.id
  name                 = "uniprot"
  description          = "One Markdown document per reviewed human protein (UniProt Swiss-Prot)"
  data_deletion_policy = "DELETE" # removing the source also removes its passages from the index

  data_source_configuration {
    type = "S3"
    s3_configuration {
      bucket_arn         = aws_s3_bucket.knowledge.arn
      inclusion_prefixes = ["uniprot/"]
    }
  }

  vector_ingestion_configuration {
    chunking_configuration {
      chunking_strategy = "FIXED_SIZE"
      fixed_size_chunking_configuration {
        max_tokens         = 300
        overlap_percentage = 20
      }
    }
  }
}

# A paper reads differently: cut where the topic changes (semantic chunking), not every N tokens.
resource "aws_bedrockagent_data_source" "papers" {
  region               = local.knowledge_region
  knowledge_base_id    = aws_bedrockagent_knowledge_base.proteins.id
  name                 = "papers"
  description          = "The QAFI paper"
  data_deletion_policy = "DELETE"

  data_source_configuration {
    type = "S3"
    s3_configuration {
      bucket_arn         = aws_s3_bucket.knowledge.arn
      inclusion_prefixes = ["papers/"]
    }
  }

  vector_ingestion_configuration {
    chunking_configuration {
      chunking_strategy = "SEMANTIC"
      semantic_chunking_configuration {
        max_token                       = 300
        buffer_size                     = 1  # compare each sentence together with its neighbours
        breakpoint_percentile_threshold = 95 # cut at the 5% largest changes in meaning
      }
    }
  }
}

output "knowledge" {
  description = "The knowledge base for notebook 11"
  value = {
    region         = local.knowledge_region
    bucket         = aws_s3_bucket.knowledge.bucket
    knowledge_base = aws_bedrockagent_knowledge_base.proteins.id
    data_sources   = { uniprot = aws_bedrockagent_data_source.uniprot.data_source_id, papers = aws_bedrockagent_data_source.papers.data_source_id }
  }
}
