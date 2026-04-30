# 02-eks - small EKS cluster (capstone)

This is a stub. You build it up during W16 as part of the capstone.

## Target architecture

- EKS 1.30+ control plane.
- 2 `t3.medium` (or `t4g.medium`) managed node group nodes in private subnets.
- Cilium as CNI (replaces aws-vpc-cni and kube-proxy).
- ALB controller for ingress (`eks.amazonaws.com/role-arn` on the SA).
- VPC interface endpoints for `ecr.api`, `ecr.dkr`, `sts`, `logs`. Gateway endpoint for S3.
- A 3-tier sample app (web -> api -> RDS).
- NetworkPolicy default-deny + explicit allows.

## Suggested files when you flesh this out

- `main.tf` - module call to `terraform-aws-modules/eks/aws` or hand-rolled.
- `cilium.tf` - `helm_release` for Cilium.
- `alb-controller.tf` - IRSA + helm.
- `app.tf` - `kubernetes_*` resources for the demo app.
- `rds.tf` - small Postgres in private subnet.

## Cost reminder

EKS control plane is $0.10/hr. Tear it down between sessions.
