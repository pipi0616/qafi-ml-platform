# The prediction API (notebook 10b): the container image in ECR, run by Lambda, behind API Gateway.
#
#   caller, signed with its AWS credentials
#        │  GET /v1/proteins/TP53/variants/R175H
#        ▼
#   API Gateway qafi-dev-api (HTTP API: checks the signature, at most 2 requests a second)
#        ▼
#   Lambda qafi-dev-api (the image from ECR: FastAPI + the Lambda Web Adapter)
#        │  SQL with parameters
#        ▼
#   Athena workgroup qafi-dev ──▶ views variant_effect_current, proteins_current ──▶ lake bucket (lake.tf)
#
# Deploying: notebook 10b builds and pushes the image and writes its exact name, with its digest, into
# build/api_image.txt; terraform apply then points the function at it. Before the first push the file does
# not exist and only the repository is created, because Lambda needs an image to start from.

data "aws_region" "current" {}

locals {
  api_image_file = "${path.module}/build/api_image.txt"
  api_image      = fileexists(local.api_image_file) ? trimspace(file(local.api_image_file)) : ""
  api_count      = local.api_image == "" ? 0 : 1
  glue_arn       = "arn:aws:glue:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}"
}

# ---------------------------------------------------------------------------
# Where the images live
# ---------------------------------------------------------------------------
resource "aws_ecr_repository" "api" {
  name                 = "qafi-dev-api"
  image_tag_mutability = "IMMUTABLE" # a tag (the git commit) always means the same image
  force_delete         = true        # dev: destroy also removes the images

  image_scanning_configuration {
    scan_on_push = true # every pushed image is checked for known vulnerabilities in its packages
  }
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the 5 newest images: enough to roll back, never a growing bill"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 5 }
      action       = { type = "expire" }
    }]
  })
}

# ---------------------------------------------------------------------------
# The function's identity: run queries in the workgroup, read the lake and the table definitions
# ---------------------------------------------------------------------------
resource "aws_iam_role" "api" {
  name = "qafi-dev-api"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "api" {
  name = "query-the-lake"
  role = aws_iam_role.api.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "RunQueriesInTheWorkgroup"
        Effect   = "Allow"
        Action   = ["athena:StartQueryExecution", "athena:GetQueryExecution", "athena:GetQueryResults", "athena:StopQueryExecution"]
        Resource = aws_athena_workgroup.lake.arn
      },
      {
        Sid      = "ReadTheTableDefinitions"
        Effect   = "Allow"
        Action   = ["glue:GetDatabase", "glue:GetTable", "glue:GetPartition", "glue:GetPartitions"]
        Resource = ["${local.glue_arn}:catalog", "${local.glue_arn}:database/${aws_glue_catalog_database.lake.name}", "${local.glue_arn}:table/${aws_glue_catalog_database.lake.name}/*"]
      },
      {
        Sid      = "ReadTheLake"
        Effect   = "Allow"
        Action   = "s3:GetObject"
        Resource = [for table in keys(local.lake_tables) : "${local.lake_bucket.arn}/${table}/*"]
      },
      {
        Sid      = "ListTheBuckets"
        Effect   = "Allow"
        Action   = ["s3:ListBucket", "s3:GetBucketLocation"]
        Resource = [local.lake_bucket.arn, aws_s3_bucket.this["athena"].arn]
      },
      {
        Sid      = "WriteAndReadQueryResults"
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:AbortMultipartUpload"]
        Resource = "${aws_s3_bucket.this["athena"].arn}/results/*"
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

resource "aws_iam_role_policy_attachment" "api_logs" {
  role       = aws_iam_role.api.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# ---------------------------------------------------------------------------
# The function
# ---------------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/lambda/qafi-dev-api"
  retention_in_days = 30
}

resource "aws_lambda_function" "api" {
  count = local.api_count

  function_name = "qafi-dev-api"
  role          = aws_iam_role.api.arn
  package_type  = "Image"
  image_uri     = local.api_image
  architectures = ["arm64"] # Graviton, like the other functions; the image is built for it
  memory_size   = 512       # also buys CPU: a cold start (image, Python, FastAPI) takes about a second
  timeout       = 29        # API Gateway waits at most 30 s

  environment {
    variables = {
      ATHENA_WORKGROUP = aws_athena_workgroup.lake.name
      ATHENA_DATABASE  = aws_glue_catalog_database.lake.name
    }
  }

  logging_config {
    log_format = "Text" # the app writes one JSON line per request itself
    log_group  = aws_cloudwatch_log_group.api.name
  }

  depends_on = [aws_iam_role_policy_attachment.api_logs]
}

# ---------------------------------------------------------------------------
# The front door: an HTTP API that only signed callers get through, at a limited rate
# ---------------------------------------------------------------------------
resource "aws_apigatewayv2_api" "api" {
  count = local.api_count

  name          = "qafi-dev-api"
  protocol_type = "HTTP"
  description   = "Released variant effect scores (notebook 10b)"
}

resource "aws_apigatewayv2_integration" "api" {
  count = local.api_count

  api_id                 = aws_apigatewayv2_api.api[0].id
  integration_type       = "AWS_PROXY" # the whole request goes to the function, which answers it whole
  integration_uri        = aws_lambda_function.api[0].invoke_arn
  payload_format_version = "2.0"
}

# Every GET path goes to the function; AWS_IAM: the caller must sign the request with AWS credentials allowed
# to call this API (your user is), so nobody on the internet can run queries on your bill.
resource "aws_apigatewayv2_route" "api" {
  count = local.api_count

  api_id             = aws_apigatewayv2_api.api[0].id
  route_key          = "GET /{proxy+}"
  target             = "integrations/${aws_apigatewayv2_integration.api[0].id}"
  authorization_type = "AWS_IAM"
}

resource "aws_cloudwatch_log_group" "api_access" {
  name              = "/aws/apigateway/qafi-dev-api"
  retention_in_days = 30
}

resource "aws_apigatewayv2_stage" "api" {
  count = local.api_count

  api_id      = aws_apigatewayv2_api.api[0].id
  name        = "$default"
  auto_deploy = true

  default_route_settings {
    throttling_rate_limit  = 2 # requests per second, on average
    throttling_burst_limit = 5 # requests at the same moment
  }

  # One JSON line per request: who called, what, the answer and how long it took.
  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_access.arn
    format = jsonencode({
      requestId = "$context.requestId"
      caller    = "$context.identity.userArn"
      method    = "$context.httpMethod"
      path      = "$context.path"
      status    = "$context.status"
      latencyMs = "$context.responseLatency"
      lambdaMs  = "$context.integrationLatency"
    })
  }
}

resource "aws_lambda_permission" "api" {
  count = local.api_count

  statement_id  = "AllowApiGatewayToInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api[0].function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.api[0].execution_arn}/*/*"
}

output "api" {
  description = "The image repository, and the API's address once an image is deployed (notebook 10b)"
  value = {
    repository = aws_ecr_repository.api.repository_url
    url        = local.api_count == 1 ? aws_apigatewayv2_api.api[0].api_endpoint : null
    function   = local.api_count == 1 ? aws_lambda_function.api[0].function_name : null
  }
}
