# The agent's guardrail (notebook 12), in Ireland next to Claude (decision 9, docs/plan.md).
#
# The agent applies it twice through the ApplyGuardrail API, which works independently of the model:
#   the question, before the agent starts        (source INPUT)
#   the answer, before the user sees it           (source OUTPUT, with the tool results as grounding sources)
#
# The "classic" tier is used: the newer "standard" tier routes across Regions, which the free plan cannot do.

resource "aws_bedrock_guardrail" "agent" {
  region      = local.knowledge_region
  name        = "qafi-dev-agent"
  description = "Research use only: no personal medical advice, no personal data, answers grounded in tool results"

  blocked_input_messaging   = "I can answer research questions about proteins and variants, but I cannot give medical advice about a specific person. Please ask a clinical geneticist."
  blocked_outputs_messaging = "I could not produce an answer that is supported by the scores and documents I found."

  # A research tool must not act as a doctor for a particular patient.
  topic_policy_config {
    topics_config {
      name       = "Personal medical advice"
      type       = "DENY"
      definition = "Diagnosing, treating or advising a specific person or patient about their own health, genetic test result, disease risk or treatment decision."
      examples = [
        "My father carries TP53 R175H, should he start chemotherapy?",
        "My genetic test shows a BRCA1 variant, do I need surgery?",
        "Will my baby be sick if she has this variant?",
      ]
    }
  }

  # Personal contact details are masked in questions and answers (each side must be switched on).
  sensitive_information_policy_config {
    dynamic "pii_entities_config" {
      for_each = ["EMAIL", "PHONE"]
      content {
        type           = pii_entities_config.value
        action         = "ANONYMIZE"
        input_enabled  = true
        input_action   = "ANONYMIZE"
        output_enabled = true
        output_action  = "ANONYMIZE"
      }
    }
  }

  content_policy_config {
    filters_config { # attempts to override the instructions ("ignore all previous instructions ...")
      type            = "PROMPT_ATTACK"
      input_strength  = "HIGH"
      output_strength = "NONE"
    }
    filters_config {
      type            = "HATE"
      input_strength  = "MEDIUM"
      output_strength = "MEDIUM"
    }
    filters_config {
      type            = "INSULTS"
      input_strength  = "MEDIUM"
      output_strength = "MEDIUM"
    }
  }

  # An answer must rest on what the tools returned. Thresholds measured in notebook 12: a made-up answer scores
  # 0.00 and a correct summary of the results 0.77 to 0.97, but correct answers that compare numbers (0.31) or
  # say that a variant does not exist (0.12) score low too. So only near-certain fabrication is blocked; the
  # relevance filter is not used: it judged "that variant does not exist" as off topic (0.06).
  contextual_grounding_policy_config {
    filters_config {
      type      = "GROUNDING"
      threshold = 0.1
    }
  }
}

# A numbered, unchangeable snapshot: the agent uses a version, so editing the draft cannot change it silently.
resource "aws_bedrock_guardrail_version" "agent" {
  region        = local.knowledge_region
  guardrail_arn = aws_bedrock_guardrail.agent.guardrail_arn
  description   = "Released by Terraform (notebook 12)"

  lifecycle {
    replace_triggered_by = [aws_bedrock_guardrail.agent] # any change to the policies publishes a new version
  }
}

output "guardrail" {
  description = "The agent's guardrail for notebook 12"
  value = {
    region  = local.knowledge_region
    id      = aws_bedrock_guardrail.agent.guardrail_id
    version = aws_bedrock_guardrail_version.agent.version
  }
}
