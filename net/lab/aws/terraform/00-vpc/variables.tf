variable "name" {
  description = "Name prefix for all resources"
  type        = string
  default     = "net-lab"
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "eu-central-1"
}

variable "owner" {
  description = "Tag value for Owner"
  type        = string
  default     = "you"
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.50.0.0/16"
}

variable "enable_nat_gw" {
  description = "Create a NAT Gateway. Costs ~0.045 USD/hr."
  type        = bool
  default     = true
}

variable "enable_flow_logs" {
  description = "Enable VPC flow logs to CloudWatch."
  type        = bool
  default     = true
}
