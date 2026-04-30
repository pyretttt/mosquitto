# Docker labs

Three small stacks for the application-layer / TLS / proxy weeks. Run them from inside the Linux VM.

## What is here

- `dns-lab/` - bind9 authoritative server + a client. Used in W4.
- `tls-lab/` - nginx with self-signed cert. Used in W12.
- `proxy-lab/` - nginx as a forward proxy in front of a tiny upstream. Used in W3 and reused in W12 for TLS termination.
- `docker-compose.yml` - generic "client + server + netshoot sidecar" environment. Useful any week.

## Why netshoot

`nicolaka/netshoot` ships every common networking tool (tcpdump, dig, mtr, iperf3, hping3, nmap, drill, ngrep, openssl, curl with HTTP/3, ...). Drop it in any compose stack as a sidecar:

```yaml
services:
  debug:
    image: nicolaka/netshoot
    network_mode: "service:server"
    command: sleep infinity
```

Then `docker compose exec debug tcpdump -ni eth0 -w /tmp/cap.pcap` captures from inside server's network namespace.

## Conventions

- Each lab folder has its own `docker-compose.yml`.
- Captures go to `../../../weekNN/captures/` (mounted via volume) so they end up next to your notes.
- `docker compose down -v` between sessions to keep state clean.
