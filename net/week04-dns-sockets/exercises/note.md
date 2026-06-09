## Exercises

1. **Conceptual.** In `notes.md`:
   - When you `dig +trace`, in what order are queries made and what's the role of each server? (root -> TLD -> authoritative)

    `dig +trace` traces from LDNS to root to RTLD to authoritative

   - What's the difference between iterative and recursive resolution? Who does which?

    Iterative required round trip from client to each consequent dns. Recursive instructs first DNS to resolve whole path by itself.

   - Why is `ANY` queries discouraged? (RFC 8482)

    Mostly because they cause severe traffic amplification, and made DDOS over DNS easier.

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
