# Lets GitHub Actions act in this account without any stored AWS key.
#
# How it works: GitHub signs a short-lived token for each workflow run. AWS trusts
# GitHub's token service (the OIDC provider below) and lets a run from THIS repository
# assume the role. The role's permissions are read-only for now; later notebooks add
# what the pipelines need.

variable "github_sub_prefix" {
  description = "Subject prefix of this repository's GitHub OIDC tokens; look it up with: gh api repos/OWNER/REPO/actions/oidc/customization/sub"
  type        = string
  # GitHub's immutable subject format: owner and repository carry their numeric IDs, so a
  # deleted and re-created repository with the same name cannot assume the role.
  default = "repo:pipi0616@147825483/qafi-ml-platform@1409257726"
}

resource "aws_iam_openid_connect_provider" "github" {
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
}

data "aws_iam_policy_document" "github_trust" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    # Only workflows of this repository running on main: the repository is public, so a branch
    # or pull request (anyone's code) never gets AWS credentials. The aws job runs only on main.
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["${var.github_sub_prefix}:ref:refs/heads/main"]
    }
  }
}

resource "aws_iam_role" "github_actions" {
  name               = "qafi-dev-github-actions"
  assume_role_policy = data.aws_iam_policy_document.github_trust.json
}

resource "aws_iam_role_policy_attachment" "github_actions_readonly" {
  role       = aws_iam_role.github_actions.name
  policy_arn = "arn:aws:iam::aws:policy/ReadOnlyAccess"
}

output "github_actions_role_arn" {
  description = "Set this as the repository variable AWS_ROLE_ARN in GitHub"
  value       = aws_iam_role.github_actions.arn
}
