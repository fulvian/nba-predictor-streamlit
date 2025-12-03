# Terraform Backend Configuration for NBA Predictor
# Context7-Compliant State Management

# S3 Backend for Remote State
terraform {
  backend "s3" {
    bucket         = "nba-predictor-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "nba-predictor-terraform-locks"

    # Context7-compliant state management
    context7_compliance = "enabled"
    state_encryption    = "AES256"
    locking_mechanism   = "dynamodb"

    # State versioning and backups
    state_versioning = "enabled"
    state_backups    = "enabled"
  }
}

# S3 Bucket for Terraform State
resource "aws_s3_bucket" "terraform_state" {
  bucket = "nba-predictor-terraform-state"

  tags = merge(local.context7_labels, {
    Name        = "nba-predictor-terraform-state"
    Purpose     = "terraform-state"
    Environment = var.environment
  })
}

# S3 Bucket Versioning
resource "aws_s3_bucket_versioning" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id
  versioning_configuration {
    status = "Enabled"
  }
}

# S3 Bucket Encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# S3 Bucket Public Access Block
resource "aws_s3_bucket_public_access_block" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# DynamoDB Table for State Locking
resource "aws_dynamodb_table" "terraform_locks" {
  name           = "nba-predictor-terraform-locks"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }

  tags = merge(local.context7_labels, {
    Name        = "nba-predictor-terraform-locks"
    Purpose     = "terraform-locking"
    Environment = var.environment
  })
}

# KMS Key for Additional Encryption (Context7 Compliance)
resource "aws_kms_key" "terraform_state" {
  description             = "KMS key for Terraform state encryption"
  deletion_window_in_days = 10
  enable_key_rotation     = true

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow Terraform Access"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = "*"
      }
    ]
  })

  tags = merge(local.context7_labels, {
    Name        = "nba-predictor-terraform-kms"
    Purpose     = "terraform-encryption"
    Environment = var.environment
  })
}

# KMS Key Alias
resource "aws_kms_alias" "terraform_state" {
  name          = "alias/nba-predictor-terraform-state"
  target_key_id = aws_kms_key.terraform_state.key_id
}

# Data source for current AWS account
data "aws_caller_identity" "current" {}

# S3 Bucket Lifecycle Configuration
resource "aws_s3_bucket_lifecycle_configuration" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id

  rule {
    id     = "cleanup_old_versions"
    status = "Enabled"

    noncurrent_version_expiration {
      noncurrent_days = 30
    }

    noncurrent_version_transition {
      noncurrent_days = 7
      storage_class   = "STANDARD_IA"
    }

    noncurrent_version_transition {
      noncurrent_days = 14
      storage_class   = "GLACIER"
    }
  }
}

# CloudWatch Log Group for Terraform Operations
resource "aws_cloudwatch_log_group" "terraform" {
  name              = "/aws/terraform/nba-predictor"
  retention_in_days = 30

  tags = merge(local.context7_labels, {
    Name        = "nba-predictor-terraform-logs"
    Purpose     = "terraform-auditing"
    Environment = var.environment
  })
}

# Context7 Compliance Monitoring
resource "aws_cloudwatch_metric_alarm" "terraform_state_access" {
  alarm_name          = "nba-predictor-terraform-state-access"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "GetObjects"
  namespace           = "AWS/S3"
  period              = "300"
  statistic           = "Sum"
  threshold           = "10"
  alarm_description   = "This metric monitors terraform state access patterns"
  treat_missing_data  = "notBreaching"

  dimensions = {
    BucketName = aws_s3_bucket.terraform_state.id
  }

  tags = merge(local.context7_labels, {
    Name        = "nba-predictor-terraform-state-alarm"
    Purpose     = "terraform-monitoring"
    Environment = var.environment
  })
}