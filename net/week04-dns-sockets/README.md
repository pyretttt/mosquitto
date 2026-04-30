# Week 4 - Ch2: DNS + Socket programming + P2P

"It's always DNS." This week you internalize how DNS *actually* works (recursion, caching, record types), then write your first socket programs.

## Reading

- Kurose & Ross 8e, 2.4-2.7 (~30 pages):
  - 2.4 DNS - the Internet's directory service
  - 2.5 Peer-to-peer applications (skim)
  - 2.6 Video streaming and CDNs
  - 2.7 Socket programming
- RFC 1034 (DNS concepts) and RFC 1035 (DNS specification) - skim, especially 1035 sections 3-4 (message format).
- Julia Evans, "Implement DNS in a weekend": https://implement-dns.wizardzines.com/
- Beej's Guide to Network Programming, sections 5-7 (sockets API).

## Learning goals

- Trace a recursive DNS lookup from your laptop -> resolver -> root -> TLD -> authoritative.
- Read a DNS query/response in Wireshark byte-by-byte (header, question, answer sections).
- Know the common record types (A, AAAA, CNAME, MX, TXT, NS, SOA, PTR, SRV, CAA) and when to use each.
- Understand caching: `TTL`, negative caching, why DNS changes "take time".
- Write a TCP echo server + client and a DNS query builder over UDP.

## Lab

**Goal:** stand up your own authoritative + recursive DNS pair and watch every step on the wire.

1. Generate certs (not needed here, but the lab dir uses other utilities):
   ```bash
   cd lab/docker/dns-lab
   docker compose up -d
   ```
2. Smoke test:
   ```bash
   docker compose exec client dig @10.20.0.20 www.lab.local
   docker compose exec client dig @10.20.0.20 mail.lab.local MX
   docker compose exec client dig @10.20.0.20 short.lab.local      # CNAME chase
   docker compose exec client dig @10.20.0.10 www.lab.local +norec # ask authoritative directly
   ```
3. Capture queries:
   ```bash
   docker compose exec debug tcpdump -ni eth0 -w /tmp/dns.pcap port 53 &
   docker compose exec client dig @10.20.0.20 www.lab.local
   docker compose exec client dig @10.20.0.20 www.lab.local        # cached - watch TTL drop
   docker compose exec client dig +trace @1.1.1.1 cursor.com
   docker compose exec debug pkill tcpdump
   ```
4. Open the pcap in Wireshark. Find the QNAME, QTYPE, ANSWERS in the binary.

## Exercises

1. **Conceptual.** In `notes.md`:
   - When you `dig +trace`, in what order are queries made and what's the role of each server? (root -> TLD -> authoritative)
   - What's the difference between iterative and recursive resolution? Who does which?
   - Why is `ANY` queries discouraged? (RFC 8482)
2. **Practical - TCP echo.** In `exercises/04-echo-server.py` build:
   - A concurrent server (using `asyncio` or threads) on `0.0.0.0:9000` that echoes any line back.
   - A client that sends 100 lines and verifies the responses.
   - Capture the TCP handshake on the wire.
3. **Practical - hand-rolled DNS query.** In `exercises/04-dns-query.py`, using `socket` + `struct`, build a DNS query for `A` of `cursor.com`, send it as UDP to `1.1.1.1:53`, parse the response. Don't use `dnspython` for the wire format - read RFC 1035 section 4.1 and write the bytes yourself. (You may use `dnslib` for unit tests / verification.)

## Self-check

- Why does DNS use UDP by default but fall back to TCP?
- What's a CNAME chain and why is it dangerous to put one on a zone apex?
- What's the difference between a public resolver (1.1.1.1, 8.8.8.8) and an authoritative server?
- What's a glue record and when do you need one?
- What does `EDNS0` give us?

## Common gotchas

- `bind9` is picky about zone serial numbers and trailing dots. If your zone won't load, check journalctl for the line number.
- macOS caches DNS aggressively. Use `dscacheutil -flushcache; sudo killall -HUP mDNSResponder` if needed (or just test from inside the Linux VM/container).
