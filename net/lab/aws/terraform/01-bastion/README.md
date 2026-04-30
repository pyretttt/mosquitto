# 01-bastion - SSM-managed bastion

Tiny `t4g.micro` in a private subnet with the SSM agent. No SSH ports, no key pair. You connect via `aws ssm start-session`.

This module is a stub. Fill it in during W15 once you have hands-on with the base VPC.

```bash
aws ssm start-session --target <instance-id>

aws ssm start-session --target <instance-id> \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["80"],"localPortNumber":["8080"]}'
```

## Skeleton todo

- IAM role with `AmazonSSMManagedInstanceCore`.
- Latest Amazon Linux 2023 AMI via `data "aws_ami"`.
- Reads `vpc_id`, `private_subnet_ids` from `00-vpc` outputs.
- Security group: egress 0.0.0.0/0, no ingress.
- VPC interface endpoints for `com.amazonaws.<region>.ssm`, `ssmmessages`, `ec2messages` so the bastion can reach SSM without a NAT GW (or keep the NAT GW for simplicity).
