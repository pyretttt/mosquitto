# TLS material (local only)

Generate a self-signed cert for Ingress with:

```bash
mise run gen-certs
```

Outputs (gitignored):

- `tls.crt` / `tls.key` — mounted as Secret `records-tls`
- `ca.crt` — trust this with `curl --cacert` during verify steps

Never commit private keys.
