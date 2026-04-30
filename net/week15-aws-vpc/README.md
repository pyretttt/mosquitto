# Week 15 - AWS networking

You take everything you've built locally and learn the AWS abstractions that wrap it. By the end of this week the base VPC for the W16 capstone is up.

## Reading

- AWS VPC user guide (skim, then deep on Subnets, Route Tables, IGW, NAT GW, SG vs NACL): https://docs.aws.amazon.com/vpc/latest/userguide/
- "AWS Networking Fundamentals" whitepaper: https://docs.aws.amazon.com/whitepapers/latest/aws-vpc-connectivity-options/welcome.html
- VPC Reachability Analyzer docs: https://docs.aws.amazon.com/vpc/latest/reachability/
- VPC Flow Logs guide: https://docs.aws.amazon.com/vpc/latest/userguide/flow-logs.html
- re:Invent (latest year) NET-tagged talks. The classic "Advanced VPC Design" (NET401/NET402) is required watching - search YouTube.
- (optional) "A Day in the Life of a Billion Packets" - re:Invent classic.

## Learning goals

- Explain how a VPC differs from a "real" network: it's a software-defined L3 fabric. There's no L2 broadcast across subnets. ARP is mostly synthesized.
- Reason about: VPC, subnet (per-AZ), route table (per-subnet or main), IGW, NAT GW, ENI, EIP, SG (stateful, per-ENI), NACL (stateless, per-subnet).
- Set up a 2-AZ VPC with public + private subnets, IGW, NAT GW, route tables.
- Use Reachability Analyzer + Flow Logs to diagnose connectivity.
- Know where VPC Endpoints (Gateway: S3/DynamoDB; Interface: everything else) save you NAT GW bandwidth.

## Lab

**Goal:** stand up the base VPC with Terraform, deliberately misconfigure it, diagnose with AWS-native tools.

1. Pre-flight (see `lab/aws/README.md`):
   ```bash
   aws configure --profile netlab
   export AWS_PROFILE=netlab
   aws sts get-caller-identity
   ```
2. Apply the base VPC:
   ```bash
   cd lab/aws/terraform/00-vpc
   cp terraform.tfvars.example terraform.tfvars
   terraform init
   terraform plan
   terraform apply
   ```
3. Quick connectivity tests using a t4g.micro test instance. Spin one up via Console or extend the terraform with a `aws_instance`. Connect via SSM.
4. Run **Reachability Analyzer**:
   - Console -> VPC -> Reachability Analyzer -> Create path.
   - Source: instance ENI. Destination: `0.0.0.0/0` (or a specific public IP). Protocol TCP/443.
   - Inspect the explanation graph. It will tell you if any RT/SG/NACL hop blocks you.
5. Look at **VPC Flow Logs** in CloudWatch. Filter by your instance's ENI.
6. **Break things on purpose.** For each, predict the failure mode, run Reachability Analyzer, then look at Flow Logs:
   - Remove the `0.0.0.0/0 -> NAT GW` route from the private route table. Try to `curl https://aws.amazon.com/`.
   - Add a NACL rule denying outbound port 443 on the private subnet. (Stateless! Allow the return port range.)
   - Tighten the SG to deny egress 443. Compare error to the NACL case.
   - Disassociate the public route table from a public subnet.

## Exercises

1. **Conceptual.** In `notes.md`:
   - Write a comparison table: SG vs NACL (stateful?, per-ENI vs per-subnet, allow-only vs allow+deny+ordered, default behavior).
   - Why is "no L2" a feature, not a bug, of AWS VPC? What does it free AWS to do?
   - When does a VPC Interface Endpoint pay for itself vs sending traffic via NAT GW?
2. **Practical - reproduce all 4 failures above.** For each, write down: predicted failure, actual error message at the client, what Reachability Analyzer said, what Flow Logs showed (`ACCEPT` vs `REJECT`, `srcaddr`, `dstaddr`, port). Save as `exercises/15-failures.md`.
3. **Practical - VPC Endpoint.** Add a Gateway endpoint for S3 and an Interface endpoint for STS. Show with Flow Logs that traffic now goes through the endpoint, not through the NAT GW.
4. **Stretch - Transit Gateway.** Stand up two VPCs in two regions and TGW-peer them. Reproduce a route-table misconfiguration that prevents one direction.

## Self-check

- Without notes: which AWS network construct is *stateful*? *Stateless*? *Per-ENI*? *Per-subnet*?
- What does "main route table" mean in a VPC?
- Where do the IPs of the AWS-managed DNS resolver come from? (Hint: VPC base + 2.)
- Why does enabling DNS hostnames + DNS support matter for Private Hosted Zones?
- What goes in a Flow Logs record? Why is `srcaddr` ambiguous when SNAT happens before the ENI?

## Useful one-liners

```bash
aws ec2 describe-vpcs --filters "Name=tag:Project,Values=net-lab"
aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID"
aws ec2 describe-route-tables --filters "Name=vpc-id,Values=$VPC_ID"
aws ec2 describe-security-groups --filters "Name=vpc-id,Values=$VPC_ID"
aws ec2 describe-network-acls --filters "Name=vpc-id,Values=$VPC_ID"

aws ec2 describe-flow-logs
aws logs tail /aws/vpc/net-lab/flow-logs --follow

aws ssm start-session --target i-...
```
