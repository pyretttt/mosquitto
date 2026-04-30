# AWS networking lab

Terraform skeletons for the AWS modules in W15 and W16. **Nothing is applied automatically.** Each module is independent and meant to be `terraform apply`'d only when you reach that week.

## Layout

- `terraform/00-vpc/` - W15 base: VPC, public + private subnets across 2 AZs, IGW, NAT GW, route tables, flow logs.
- `terraform/01-bastion/` - W15: SSM-managed bastion in the private subnet. No public SSH.
- `terraform/02-eks/` - W16: small EKS cluster + Cilium + 3-tier app.
- `terraform/03-edge/` - W16: ALB, Route53 zone (you bring the domain), ACM cert.
- `diagrams/` - PlantUML / mermaid diagrams of each module.

## Cost discipline

- AWS Free Tier covers `t3.micro`/`t4g.micro`, ~750 hrs of EC2, 1 ALB, 5 GB CloudWatch logs.
- NAT Gateway costs ~$0.045/hr + data. Do not leave it on overnight if you're cost-sensitive.
- EKS control plane is $0.10/hr (~$72/mo if left on). Tear down between sessions.
- VPC Endpoints (interface) cost per-AZ-per-hour. Use Gateway endpoints (S3, DynamoDB) when possible.

## Pre-flight

```bash
aws configure --profile netlab
export AWS_PROFILE=netlab
aws sts get-caller-identity

cd lab/aws/terraform/00-vpc
cp terraform.tfvars.example terraform.tfvars
terraform init
terraform plan
```

Use a billing alarm before you `apply` anything.

## Tear-down

Each module has a `terraform destroy` target. `02-eks` should be destroyed before `00-vpc`.
