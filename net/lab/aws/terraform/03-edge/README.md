# 03-edge - public edge: ALB + Route53 + ACM

Stub. Build out during W16.

## Target

- Public ALB in `public_subnet_ids`.
- ACM cert (DNS-validated against your Route53 zone).
- Route53 alias record `app.<your-domain>` -> ALB.
- ALB listener 443 forwarding to the EKS Ingress (or directly to a target group).
- HTTP -> HTTPS redirect.

## Notes

You need a real domain in Route53 for ACM DNS validation. If you don't have one, use `nip.io` for the hostname and skip ACM (test with HTTP only).
