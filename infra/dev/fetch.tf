# Public-data ingestion (notebook 03b): a Lambda that copies one file from its publisher's
# URL straight into datasets/ in the raw bucket. The upload then triggers the validator in
# ingest.tf, exactly as for a file uploaded by hand.
#
#   config/sources.toml ──▶ invoke qafi-dev-raw-fetcher {url, key}   (one call per file)
#                                │  HTTP range requests, 8 parts at a time
#                                ▼
#                           s3://<raw>/datasets/<dataset>/<version>/<file> ──▶ validator ──▶ _registry/

data "archive_file" "raw_fetcher" {
  type        = "zip"
  source_file = "${path.module}/../../src/qafi_platform/raw_fetcher.py"
  output_path = "${path.module}/build/raw_fetcher.zip"
}

# ---------------------------------------------------------------------------
# The function's identity: may write new files under datasets/, nothing else
# ---------------------------------------------------------------------------
resource "aws_iam_role" "raw_fetcher" {
  name = "qafi-dev-raw-fetcher"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "raw_fetcher" {
  name = "write-datasets"
  role = aws_iam_role.raw_fetcher.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "WriteNewFiles"
        Effect   = "Allow"
        Action   = ["s3:PutObject", "s3:AbortMultipartUpload"]
        Resource = "${local.raw_bucket.arn}/datasets/*"
      },
      {
        Sid      = "CheckWhatWasWritten"
        Effect   = "Allow"
        Action   = "s3:GetObject" # HeadObject: compare the stored size with the publisher's
        Resource = "${local.raw_bucket.arn}/datasets/*"
      },
      {
        Sid      = "UseTheBucketKey"
        Effect   = "Allow"
        Action   = ["kms:Decrypt", "kms:GenerateDataKey"] # multipart uploads to an SSE-KMS bucket need both
        Resource = aws_kms_key.lake.arn
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "raw_fetcher_logs" {
  role       = aws_iam_role.raw_fetcher.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# ---------------------------------------------------------------------------
# The function itself
# ---------------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "raw_fetcher" {
  name              = "/aws/lambda/qafi-dev-raw-fetcher"
  retention_in_days = 30
}

resource "aws_lambda_function" "raw_fetcher" {
  function_name    = "qafi-dev-raw-fetcher"
  role             = aws_iam_role.raw_fetcher.arn
  runtime          = "python3.12"
  architectures    = ["arm64"]
  handler          = "raw_fetcher.lambda_handler"
  filename         = data.archive_file.raw_fetcher.output_path
  source_code_hash = data.archive_file.raw_fetcher.output_base64sha256
  memory_size      = 2048 # 8 parts of 64 MB in flight; more memory also means more network bandwidth
  timeout          = 900  # Lambda's maximum, 15 minutes; a 5 GB file needs a few

  environment {
    variables = { RAW_BUCKET = local.raw_bucket.bucket }
  }

  logging_config {
    log_format            = "JSON"
    application_log_level = "INFO"
    system_log_level      = "INFO"
    log_group             = aws_cloudwatch_log_group.raw_fetcher.name
  }

  depends_on = [aws_iam_role_policy_attachment.raw_fetcher_logs]
}

# A failed copy is not retried automatically: it shows up as a missing receipt and an error
# in the logs, and is re-run on purpose rather than silently downloading 5 GB three times.
resource "aws_lambda_function_event_invoke_config" "raw_fetcher" {
  function_name          = aws_lambda_function.raw_fetcher.function_name
  maximum_retry_attempts = 0
}

output "raw_fetcher_name" {
  value = aws_lambda_function.raw_fetcher.function_name
}
