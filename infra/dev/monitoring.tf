# Operations (notebook 13): alarms on what a user or the bill would notice first, and one dashboard.
#
#   API Gateway 5xx, Lambda errors or throttles, slow answers ──┐
#   Athena scanning more than 10 GB in a day ────────────────────┼──▶ SNS topic qafi-dev-alerts ──▶ e-mail
#                                                                │
#   dashboard qafi-dev: requests, latency, errors, Athena bytes, Claude tokens (Ireland), on one page
#
# The e-mail address is a variable, not written in git: terraform apply -var alert_email=you@example.org,
# then confirm the subscription from the e-mail AWS sends. The bill itself is watched by the budget of
# notebook 00 (alerts at 10, 25 and 50 USD).

variable "alert_email" {
  description = "Where alarms are sent; empty means no e-mail (the alarms still show in the console)"
  type        = string
  default     = ""
}

# Not encrypted with the platform key: CloudWatch would need a grant in its key policy, and an alarm message
# holds no data, only the alarm's name and state.
resource "aws_sns_topic" "alerts" {
  name = "qafi-dev-alerts"
}

resource "aws_sns_topic_subscription" "alert_email" {
  count = var.alert_email == "" ? 0 : 1

  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

locals {
  alarm_actions = [aws_sns_topic.alerts.arn]
  api_id        = try(aws_apigatewayv2_api.api[0].id, "") # empty until the API exists (notebook 10b)
  api_function  = try(aws_lambda_function.api[0].function_name, "")
  api_alarm_specs = {
    api_5xx = {
      description = "The API answered with a server error (5xx): something behind it failed"
      namespace   = "AWS/ApiGateway", metric = "5xx", statistic = "Sum", threshold = 0, periods = 1
      dimensions  = { ApiId = local.api_id }
    }
    api_slow = {
      description = "Answers got slow: the 95th percentile above 5 seconds for 15 minutes"
      namespace   = "AWS/ApiGateway", metric = "Latency", statistic = null, threshold = 5000, periods = 3
      dimensions  = { ApiId = local.api_id }
    }
    lambda_errors = {
      description = "The API function raised an error"
      namespace   = "AWS/Lambda", metric = "Errors", statistic = "Sum", threshold = 0, periods = 1
      dimensions  = { FunctionName = local.api_function }
    }
    lambda_throttles = {
      description = "Requests were turned away because too many ran at once"
      namespace   = "AWS/Lambda", metric = "Throttles", statistic = "Sum", threshold = 0, periods = 1
      dimensions  = { FunctionName = local.api_function }
    }
  }
  api_alarms = { for name, spec in local.api_alarm_specs : name => spec if local.api_count == 1 }
}

resource "aws_cloudwatch_metric_alarm" "api" {
  for_each = local.api_alarms

  alarm_name          = "qafi-dev-${replace(each.key, "_", "-")}"
  alarm_description   = each.value.description
  namespace           = each.value.namespace
  metric_name         = each.value.metric
  dimensions          = each.value.dimensions
  statistic           = each.value.statistic
  extended_statistic  = each.value.statistic == null ? "p95" : null
  period              = 300
  evaluation_periods  = each.value.periods
  threshold           = each.value.threshold
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching" # no traffic is not a problem
  alarm_actions       = local.alarm_actions
  ok_actions          = local.alarm_actions
}

# The workgroup's own limit stops one query at 5 GB; this catches many queries adding up.
resource "aws_cloudwatch_metric_alarm" "athena_bytes" {
  alarm_name          = "qafi-dev-athena-bytes"
  alarm_description   = "Athena scanned more than 10 GB in a day (about 5 cents): something queries far more than usual"
  namespace           = "AWS/Athena"
  metric_name         = "ProcessedBytes"
  dimensions          = { WorkGroup = aws_athena_workgroup.lake.name, QueryState = "SUCCEEDED", QueryType = "DML" }
  statistic           = "Sum"
  period              = 86400
  evaluation_periods  = 1
  threshold           = 10 * 1000 * 1000 * 1000
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
}

resource "aws_cloudwatch_dashboard" "platform" {
  dashboard_name = "qafi-dev"
  dashboard_body = jsonencode({
    widgets = concat(
      [for widget in [
        {
          type = "metric", x = 0, y = 0, width = 12, height = 6
          properties = {
            title = "API: requests and errors", region = data.aws_region.current.region, stat = "Sum", period = 300
            metrics = [
              ["AWS/ApiGateway", "Count", "ApiId", local.api_id],
              [".", "4xx", ".", "."],
              [".", "5xx", ".", "."],
            ]
          }
        },
        {
          type = "metric", x = 12, y = 0, width = 12, height = 6
          properties = {
            title = "API: latency (ms)", region = data.aws_region.current.region, period = 300
            metrics = [
              ["AWS/ApiGateway", "Latency", "ApiId", local.api_id, { stat = "p50" }],
              ["...", { stat = "p95" }],
            ]
          }
        },
        {
          type = "metric", x = 0, y = 6, width = 12, height = 6
          properties = {
            title = "Lambda: invocations, errors, throttles", region = data.aws_region.current.region, stat = "Sum", period = 300
            metrics = [
              ["AWS/Lambda", "Invocations", "FunctionName", local.api_function],
              [".", "Errors", ".", "."],
              [".", "Throttles", ".", "."],
            ]
          }
        },
      ] : widget if local.api_count == 1],
      [
        {
          type = "metric", x = 12, y = 6, width = 12, height = 6
          properties = {
            title   = "Athena: bytes scanned per hour", region = data.aws_region.current.region, period = 3600
            metrics = [["AWS/Athena", "ProcessedBytes", "QueryState", "SUCCEEDED", "QueryType", "DML", "WorkGroup", aws_athena_workgroup.lake.name, { stat = "Sum" }]]
          }
        },
        {
          type = "metric", x = 0, y = 12, width = 24, height = 6
          properties = {
            title = "Claude on Bedrock (Ireland): tokens per hour", region = local.knowledge_region, period = 3600, stat = "Sum"
            metrics = [
              [{ expression = "SEARCH('{AWS/Bedrock,ModelId} MetricName=\"InputTokenCount\"', 'Sum', 3600)", id = "input", label = "input tokens" }],
              [{ expression = "SEARCH('{AWS/Bedrock,ModelId} MetricName=\"OutputTokenCount\"', 'Sum', 3600)", id = "output", label = "output tokens" }],
            ]
          }
        },
      ]
    )
  })
}

output "monitoring" {
  description = "Where the alarms go and the dashboard's name (notebook 13)"
  value = {
    alerts_topic = aws_sns_topic.alerts.arn
    dashboard    = aws_cloudwatch_dashboard.platform.dashboard_name
    alarms       = concat([for a in aws_cloudwatch_metric_alarm.api : a.alarm_name], [aws_cloudwatch_metric_alarm.athena_bytes.alarm_name])
  }
}
