# Pipeline compute (notebook 14b): AWS Batch for the Nextflow reads-to-fitness pipeline.
#
# Nextflow runs on the laptop and only orchestrates: every step becomes a Batch job, Batch starts EC2
# machines when jobs are waiting and stops them when the queue is empty, so nothing costs anything idle.
# Inputs come from the raw bucket, intermediate files go to a work folder in the artifacts bucket (deleted
# after 7 days), results go to the artifacts bucket next to it.
#
#   laptop: nextflow run -profile awsbatch ──▶ Batch job queue ──▶ up to two Spot EC2 machines (x86)
#                                                   │  each job: one container, AWS CLI from the host
#                                                   ▼
#                     s3://<raw>/datasets/...  ──▶  s3://<artifacts>/nextflow/work  ──▶  s3://<artifacts>/pipelines/...

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

# The machines take no incoming traffic; they reach S3, the container registry and AWS APIs.
resource "aws_security_group" "batch" {
  name        = "qafi-dev-batch"
  description = "Batch pipeline machines: outbound only"
  vpc_id      = data.aws_vpc.default.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ---------------------------------------------------------------------------
# The machines' identity: run ECS tasks, read raw data, write the work folder and results
# ---------------------------------------------------------------------------
resource "aws_iam_role" "batch_instance" {
  name = "qafi-dev-batch-instance"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "batch_instance_ecs" {
  role       = aws_iam_role.batch_instance.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_role_policy" "batch_instance_data" {
  name = "pipeline-data"
  role = aws_iam_role.batch_instance.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ReadRawDatasets"
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = "${aws_s3_bucket.this["raw"].arn}/datasets/*"
      },
      {
        Sid      = "ReadWriteWorkAndResults"
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
        Resource = ["${aws_s3_bucket.this["artifacts"].arn}/nextflow/*", "${aws_s3_bucket.this["artifacts"].arn}/pipelines/*"]
      },
      {
        Sid      = "ListThoseBuckets"
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = [aws_s3_bucket.this["raw"].arn, aws_s3_bucket.this["artifacts"].arn]
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

resource "aws_iam_instance_profile" "batch_instance" {
  name = "qafi-dev-batch-instance"
  role = aws_iam_role.batch_instance.name
}

# ---------------------------------------------------------------------------
# The machines: 100 GB disk (DiMSum keeps three copies of the reads while it works), and the AWS CLI on
# the host (in /opt/awscli), which Nextflow mounts into every container to move files to and from S3
# ---------------------------------------------------------------------------
resource "aws_launch_template" "batch" {
  name = "qafi-dev-batch"

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = 100
      volume_type           = "gp3"
      encrypted             = true
      delete_on_termination = true
    }
  }

  metadata_options {
    http_tokens                 = "required" # IMDSv2 only
    http_put_response_hop_limit = 2          # containers on the host can still reach it
  }

  # Batch requires MIME multi-part user data. This part installs the AWS CLI into its own conda environment:
  # Nextflow mounts that folder into every container, and BioContainers images are too minimal for the
  # official AWS CLI v2 bundle (it needs the host's libz), while a conda environment carries every library
  # it needs inside the folder. This is the setup the Nextflow documentation recommends.
  user_data = base64encode(<<-EOT
    MIME-Version: 1.0
    Content-Type: multipart/mixed; boundary="==BOUNDARY=="

    --==BOUNDARY==
    Content-Type: text/x-shellscript; charset="us-ascii"

    #!/bin/bash
    set -euo pipefail
    dnf install -y bzip2
    curl -sSL https://micro.mamba.pm/api/micromamba/linux-64/2.3.3 | tar -xj -C /tmp bin/micromamba
    /tmp/bin/micromamba create -y -p /opt/awscli -c conda-forge awscli
    --==BOUNDARY==--
  EOT
  )
}

# ---------------------------------------------------------------------------
# Compute environment and queue: 0 to 2 Spot machines of 2 vCPUs and 8 GB.
# Two limits of this account decide the size, and Batch reports neither, it just leaves jobs waiting:
#   the Free plan launches only Free Tier instance types (CloudTrail: "not eligible for Free Tier");
#     the largest is m7i-flex.large, 2 vCPU / 8 GB, enough for DiMSum with 2 workers;
#   a new account may run 5 vCPUs of EC2 at once (service quota "All Standard Spot Instance Requests"),
#     so two machines at most. On a paid plan with a raised quota, list bigger types and raise max_vcpus.
# ---------------------------------------------------------------------------
resource "aws_batch_compute_environment" "pipelines" {
  name = "qafi-dev-pipelines"
  type = "MANAGED"

  compute_resources {
    type                = "SPOT"
    allocation_strategy = "SPOT_PRICE_CAPACITY_OPTIMIZED"
    min_vcpus           = 0
    max_vcpus           = 4
    instance_type       = ["m7i-flex.large"] # 2 vCPU / 8 GB, the largest Free Tier type
    instance_role       = aws_iam_instance_profile.batch_instance.arn
    subnets             = data.aws_subnets.default.ids
    security_group_ids  = [aws_security_group.batch.id]

    ec2_configuration {
      image_type = "ECS_AL2023"
    }

    launch_template {
      launch_template_id = aws_launch_template.batch.id
      version            = aws_launch_template.batch.latest_version
    }
  }
}

resource "aws_batch_job_queue" "pipelines" {
  name     = "qafi-dev-pipelines"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.pipelines.arn
  }
}

# Nextflow's intermediate files are only needed for -resume; they expire after 7 days.
resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.this["artifacts"].id

  rule {
    id     = "expire-nextflow-work"
    status = "Enabled"
    filter {
      prefix = "nextflow/work/"
    }
    expiration {
      days = 7
    }
    noncurrent_version_expiration { # the bucket is versioned: also drop the old versions
      noncurrent_days = 1
    }
    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
}

output "batch_job_queue" {
  description = "AWS Batch queue for the Nextflow pipelines (notebook 14b)"
  value       = aws_batch_job_queue.pipelines.name
}
