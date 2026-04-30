# Week 12 - Ch8 part 1: Crypto basics + TLS

The most important security protocol in production. By the end of this week, you should be able to debug a "TLS handshake failed" error from a packet capture in under 5 minutes.

## Reading

- Kurose & Ross 8e, 8.1-8.5 (~50 pages):
  - 8.1 What is network security?
  - 8.2 Principles of cryptography (symmetric, public-key, hashes, MACs, signatures)
  - 8.3 Message integrity and digital signatures
  - 8.4 Endpoint authentication
  - 8.5 Securing email - skim
  - 8.6 Securing TCP connections: TLS (read carefully)
- RFC 8446 (TLS 1.3) - read sections 1-2 + 4.1.
- "The Illustrated TLS 1.3 Connection": https://tls13.xargs.org/ - actually go through this byte-by-byte.
- (optional) RFC 5246 (TLS 1.2) for handshake comparison.

## Learning goals

- Distinguish symmetric (AES, ChaCha20) vs asymmetric (RSA, ECDSA, Ed25519) crypto and what each is used for.
- Understand certificates: subject, SAN, issuer, validity, signature, public key. Read one with `openssl x509`.
- Walk through TLS 1.3's 1-RTT handshake: ClientHello (with key share + SNI), ServerHello (key share, EncryptedExtensions, Certificate, CertificateVerify, Finished), Client Finished. Compare to TLS 1.2's 2-RTT.
- Know how SNI works (and why ECH was invented).
- Know what a self-signed cert is, what a root CA is, what intermediates are, and why "trust" is hierarchical.

## Lab

**Goal:** stand up nginx with TLS, capture a TLS 1.3 handshake, and decrypt your own traffic in Wireshark.

1. Generate certs and bring up the lab:
   ```bash
   cd lab/docker/tls-lab
   ./gen-certs.sh
   docker compose up -d
   ```
2. Test:
   ```bash
   docker compose exec client \
     curl --cacert /certs/ca.crt --resolve server.lab.local:443:10.30.0.10 \
          https://server.lab.local/
   ```
3. Inspect the cert:
   ```bash
   docker compose exec client \
     openssl s_client -connect 10.30.0.10:443 -servername server.lab.local \
                      -showcerts -tls1_3 < /dev/null
   ```
4. Capture the handshake:
   ```bash
   docker compose exec debug tcpdump -ni eth0 -w /tmp/tls.pcap port 443 &
   docker compose exec client \
     curl --cacert /certs/ca.crt --resolve server.lab.local:443:10.30.0.10 \
          https://server.lab.local/
   docker compose exec debug pkill tcpdump
   ```
5. **Decrypt your own traffic.** Set `SSLKEYLOGFILE` for `curl`:
   ```bash
   docker compose exec client bash -c '
     export SSLKEYLOGFILE=/tmp/keylog.txt
     curl --cacert /certs/ca.crt --resolve server.lab.local:443:10.30.0.10 \
          https://server.lab.local/'
   ```
   In Wireshark: Edit -> Preferences -> Protocols -> TLS -> "(Pre)-Master-Secret log filename" -> point at `keylog.txt`. The encrypted records become readable.

## Exercises

1. **Conceptual.** In `notes.md`:
   - Explain when symmetric vs asymmetric crypto is used during a TLS handshake (asymmetric for key exchange + signature, symmetric for bulk).
   - What's the difference between a MAC and a signature?
   - Walk through the TLS 1.3 handshake byte-by-byte using the captured `tls.pcap`. Diagram it.
2. **Practical - probe everything with `openssl s_client`.** Pick 5 real sites; for each, record:
   - TLS versions supported (`-tls1_2`, `-tls1_3`, `-no_tls1_3`).
   - Cipher chosen (AES-256-GCM, ChaCha20-Poly1305, ...).
   - Cert chain length and issuer.
   - SNI behavior (with vs without `-servername`).
   Save as `exercises/12-tls-survey.md`.
3. **Practical - break things.** Configure nginx to require client certificate auth (mTLS). Show what happens when the client doesn't present one. Compare `curl --cert` working vs not.
4. **Stretch.** Run https://testssl.sh/ against your nginx and against a real production site you control. Triage the findings.

## Self-check

- Without notes: list what's in a TLS 1.3 ClientHello.
- What's the difference between TLS 1.2 RSA key exchange and TLS 1.3 ECDHE? Why is the latter "PFS"?
- Why is "self-signed cert" usually rejected by browsers? When is it OK?
- What does `SAN` mean in a cert and why has it replaced `CN` for hostname verification?
- What's a certificate transparency (CT) log and why is it useful for security?

## Useful one-liners

```bash
openssl x509 -in cert.pem -text -noout
openssl x509 -in cert.pem -noout -subject -issuer -dates -ext subjectAltName
openssl s_client -connect example.com:443 -servername example.com -tls1_3 -showcerts
openssl s_client -connect example.com:443 -tlsextdebug -msg
curl -v --tlsv1.3 https://example.com/
testssl.sh https://example.com/
```
