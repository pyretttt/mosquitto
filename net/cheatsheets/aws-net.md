# AWS networking cheatsheet (`aws ec2`, Reachability Analyzer, Flow Logs)

## VPC inventory

```bash
aws ec2 describe-vpcs --filters "Name=tag:Project,Values=net-lab"
aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" \
  --query 'Subnets[].[SubnetId,AvailabilityZone,CidrBlock,MapPublicIpOnLaunch]' --output table

aws ec2 describe-route-tables --filters "Name=vpc-id,Values=$VPC_ID"
aws ec2 describe-internet-gateways --filters "Name=attachment.vpc-id,Values=$VPC_ID"
aws ec2 describe-nat-gateways --filter "Name=vpc-id,Values=$VPC_ID"
aws ec2 describe-network-acls --filters "Name=vpc-id,Values=$VPC_ID"
aws ec2 describe-security-groups --filters "Name=vpc-id,Values=$VPC_ID"
aws ec2 describe-vpc-endpoints --filters "Name=vpc-id,Values=$VPC_ID"
```

## Security Group vs NACL one-screen

| Aspect            | SG                     | NACL                       |
|-------------------|------------------------|----------------------------|
| Layer attached    | per-ENI                | per-subnet                 |
| Stateful?         | yes                    | no (must allow return)     |
| Default in        | deny all               | allow all                  |
| Default out       | allow all              | allow all                  |
| Rule semantics    | allow only             | allow + deny, ordered      |
| Evaluation        | union of attached SGs  | first match wins           |
| Counts toward     | "5 SGs per ENI" limit  | rule count limit per NACL  |

Mental model: SG is your *application-level* firewall (per-workload). NACL is your *subnet-level* firewall (rare; use as a coarse failsafe).

## ENIs and instance interfaces

```bash
aws ec2 describe-network-interfaces --filters "Name=vpc-id,Values=$VPC_ID"

aws ec2 describe-instances --filters "Name=vpc-id,Values=$VPC_ID" \
  --query 'Reservations[].Instances[].[InstanceId,PrivateIpAddress,PublicIpAddress,SubnetId]' \
  --output table
```

## Reachability Analyzer

```bash
aws ec2 create-network-insights-path \
  --source $SOURCE_ENI \
  --destination $DEST_ENI \
  --destination-port 443 \
  --protocol TCP

aws ec2 start-network-insights-analysis \
  --network-insights-path-id $PATH_ID

aws ec2 describe-network-insights-analyses \
  --network-insights-analysis-ids $ANALYSIS_ID
```

The analysis result tells you exactly which RT/SG/NACL/IGW/NAT GW component blocks the path.

## Flow Logs

Enable on a VPC, subnet, or ENI:

```bash
aws ec2 create-flow-logs \
  --resource-type VPC \
  --resource-ids $VPC_ID \
  --traffic-type ALL \
  --log-destination-type cloud-watch-logs \
  --log-destination $LOG_GROUP_ARN \
  --deliver-logs-permission-arn $ROLE_ARN
```

Default log fields:

```
version account-id interface-id srcaddr dstaddr srcport dstport
protocol packets bytes start end action log-status
```

Custom format example:

```
${version} ${vpc-id} ${subnet-id} ${interface-id} ${srcaddr} ${dstaddr}
${srcport} ${dstport} ${protocol} ${packets} ${bytes} ${action}
${tcp-flags} ${pkt-srcaddr} ${pkt-dstaddr} ${flow-direction}
```

`pkt-srcaddr` vs `srcaddr` matters when SNAT happens at the ENI: `pkt-srcaddr` is the post-NAT address, `srcaddr` is what the ENI saw.

Tail in CLI:

```bash
aws logs tail /aws/vpc/net-lab/flow-logs --follow
aws logs tail /aws/vpc/net-lab/flow-logs --since 5m \
  --filter-pattern '?REJECT'
aws logs tail /aws/vpc/net-lab/flow-logs --since 1h \
  --filter-pattern '"443"'
```

## ALB / NLB

```bash
aws elbv2 describe-load-balancers
aws elbv2 describe-target-groups
aws elbv2 describe-target-health --target-group-arn $TG_ARN
aws elbv2 describe-listeners --load-balancer-arn $ALB_ARN
aws elbv2 describe-rules --listener-arn $L_ARN
```

Common gotchas:

- ALB target group target type `instance` -> ALB sends to NodePort on instance ENI; client IP not preserved unless you enable `Proxy Protocol v2` (NLB only).
- ALB target type `ip` -> direct to pod IP (when using AWS LB controller in EKS).
- ALB has its own SG. Backend SG must allow from ALB SG (not from "anywhere").
- NLB does not have an SG (was a long-standing missing feature; now supported optionally).

## Route53

```bash
aws route53 list-hosted-zones
aws route53 list-resource-record-sets --hosted-zone-id $ZONE_ID
aws route53 test-dns-answer --hosted-zone-id $ZONE_ID --record-name app.example.com --record-type A
```

## VPC Endpoints

Gateway endpoints (free): S3, DynamoDB. Just add a route in the route table.

Interface endpoints (cost per AZ-hour): everything else (ECR, STS, SSM, CloudWatch Logs, etc.). Show up as ENIs in your subnet with private IPs and a private DNS name.

```bash
aws ec2 describe-vpc-endpoint-services --query 'ServiceNames' | grep ecr
```

## SSM Session Manager (no SSH needed)

```bash
aws ssm start-session --target i-...

aws ssm start-session --target i-... \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["80"],"localPortNumber":["8080"]}'

aws ssm start-session --target i-... \
  --document-name AWS-StartPortForwardingSessionToRemoteHost \
  --parameters '{"host":["db.internal"],"portNumber":["5432"],"localPortNumber":["5432"]}'
```

## Common debug recipe

When "the EC2 in private subnet can't reach the internet":

1. Route table on its subnet has `0.0.0.0/0 -> NAT GW`?
2. NAT GW has a route through IGW (it should automatically)?
3. SG egress allows the destination port?
4. NACL allows both directions (stateless!) on ephemeral port range?
5. Run Reachability Analyzer with the instance ENI as source.

When "ALB returns 502/504":

1. Target health: `aws elbv2 describe-target-health`. Likely "unhealthy".
2. Health check path on the target group matches the app? Returns 200?
3. SG on backend allows ALB SG on the health check port?
4. Pod actually listening on `0.0.0.0` and not just `127.0.0.1`?
5. ALB idle timeout < backend keep-alive timeout? -> 504.
