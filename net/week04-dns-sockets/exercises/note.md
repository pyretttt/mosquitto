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

   Done in `echo_server.py` server requests concurrently

   - A client that sends 100 lines and verifies the responses.

   Did just simple echo client similar to server, core mechanics are the same.

   - Capture the TCP handshake on the wire.

   Done in wireshark. Threestep handshake captured: SYN -> SYN ACK -> ACK

3. **Practical - hand-rolled DNS query.** In `exercises/04-dns-query.py`, using `socket` + `struct`, build a DNS query for `A` of `cursor.com`, send it as UDP to `1.1.1.1:53`, parse the response. Don't use `dnspython` for the wire format - read RFC 1035 section 4.1 and write the bytes yourself. (You may use `dnslib` for unit tests / verification.)

## Self-check

- Why does DNS use UDP by default but fall back to TCP?

UDP is faster, as soon as dns is required for almost all applications it's crucial to have low latency. For bit queries it's better to use TCP, because it guarantees order delivery.

- What's a CNAME chain and why is it dangerous to put one on a zone apex?

CNAME chain is linked list of name to canonical name mappings. The Zone apex is root of zone itself. A CNAME at a name cannot coexist with any other record at that name. The apex must also host NS, SOA, and often MX, TXT (SPF/DKIM), DNSKEY, etc. A CNAME would block or break those.


- What's the difference between a public resolver (1.1.1.1, 8.8.8.8) and an authoritative server?

Public resolver may not contain records for queried domain, while authoritative server name is responsible for this domain, may update records and always return update to date info

- What's a glue record and when do you need one?

GLUE record is AAAA or A recorded addded in addition to NS records returned from DNS, so that extra query is not recuired.

- What does `EDNS0` give us?


EDNS0 (Extension Mechanisms for DNS, version 0) adds an OPT pseudo-record to DNS messages. EDNS0 gives bigger UDP DNS messages, richer error codes, and a standard way to request extensions (especially DNSSEC and larger responses), which cuts TCP fallback and enables modern DNS features. Without it, you are stuck with 512-byte UDP and the original limited error set.

