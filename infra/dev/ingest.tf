# Raw-data ingestion (notebook 03): every file that arrives under datasets/ in the raw
# bucket triggers a Lambda that checks it and writes a receipt to _registry/.
#
#   upload ──▶ s3://<raw>/datasets/<dataset>/<version>/<file>
#                 │  S3 event (ObjectCreated, prefix datasets/)
#                 ▼
#              Lambda qafi-dev-raw-validator ──▶ s3://<raw>/_registry/<key>.json
#                 │
#                 ▼
#              CloudWatch Logs /aws/lambda/qafi-dev-raw-validator
#
# The receipts live under a different prefix from the trigger, so writing them can never
# trigger the function again.

locals {
  raw_bucket = aws_s3_bucket.this["raw"]
}

# ---------------------------------------------------------------------------
# The function's code: one Python file, zipped
# ---------------------------------------------------------------------------
data "archive_file" "raw_validator" {
  type        = "zip"
  source_file = "${path.module}/../../src/qafi_platform/raw_validator.py"
  output_path = "${path.module}/build/raw_validator.zip"
}

# ---------------------------------------------------------------------------
# The function's identity: may read new files, write receipts, use the key, write logs
# ---------------------------------------------------------------------------
resource "aws_iam_role" "raw_validator" {
  name = "qafi-dev-raw-validator"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "raw_validator" {
  name = "read-datasets-write-registry"
  role = aws_iam_role.raw_validator.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ReadNewFiles"
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:GetObjectVersion"]
        Resource = "${local.raw_bucket.arn}/datasets/*"
      },
      {
        Sid      = "WriteReceipts"
        Effect   = "Allow"
        Action   = "s3:PutObject"
        Resource = "${local.raw_bucket.arn}/_registry/*"
      },
      {
        Sid      = "UseTheBucketKey"
        Effect   = "Allow"
        Action   = ["kms:Decrypt", "kms:GenerateDataKey"]
        Resource = aws_kms_key.lake.arn
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "raw_validator_logs" {
  role       = aws_iam_role.raw_validator.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# ---------------------------------------------------------------------------
# The function itself and where its logs go
# ---------------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "raw_validator" {
  name              = "/aws/lambda/qafi-dev-raw-validator"
  retention_in_days = 30 # logs older than a month are deleted, so they never pile up
}

resource "aws_lambda_function" "raw_validator" {
  function_name    = "qafi-dev-raw-validator"
  role             = aws_iam_role.raw_validator.arn
  runtime          = "python3.12"
  architectures    = ["arm64"] # Graviton: about 20% cheaper than x86 for the same work
  handler          = "raw_validator.lambda_handler"
  filename         = data.archive_file.raw_validator.output_path
  source_code_hash = data.archive_file.raw_validator.output_base64sha256
  memory_size      = 256
  timeout          = 30

  environment {
    variables = { REGISTRY_PREFIX = "_registry/" }
  }

  logging_config {
    log_format            = "JSON" # one JSON object per line: searchable in CloudWatch
    application_log_level = "INFO"
    system_log_level      = "INFO"
    log_group             = aws_cloudwatch_log_group.raw_validator.name
  }

  depends_on = [aws_iam_role_policy_attachment.raw_validator_logs]
}

# ---------------------------------------------------------------------------
# The trigger: S3 may invoke the function, and does so for every new file under datasets/
# ---------------------------------------------------------------------------
resource "aws_lambda_permission" "raw_bucket" {
  statement_id   = "AllowRawBucketToInvoke"
  action         = "lambda:InvokeFunction"
  function_name  = aws_lambda_function.raw_validator.function_name
  principal      = "s3.amazonaws.com"
  source_arn     = local.raw_bucket.arn
  source_account = data.aws_caller_identity.current.account_id
}

resource "aws_s3_bucket_notification" "raw" {
  bucket = local.raw_bucket.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.raw_validator.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "datasets/"
  }

  depends_on = [aws_lambda_permission.raw_bucket]
}

# Large files go up in parts. If an upload is interrupted, the parts already sent are
# invisible but still billed; this deletes them after 7 days.
resource "aws_s3_bucket_lifecycle_configuration" "raw" {
  bucket = local.raw_bucket.id

  rule {
    id     = "abort-incomplete-uploads"
    status = "Enabled"
    filter {}
    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

output "raw_validator_log_group" {
  value = aws_cloudwatch_log_group.raw_validator.name
}
