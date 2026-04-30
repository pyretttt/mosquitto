# TLS tools cheatsheet (`openssl`, `curl`, `testssl.sh`)

## openssl s_client (the workhorse)

```bash
openssl s_client -connect example.com:443 -servername example.com
openssl s_client -connect example.com:443 -servername example.com -showcerts
openssl s_client -connect example.com:443 -servername example.com -tls1_3
openssl s_client -connect example.com:443 -servername example.com -tls1_2
openssl s_client -connect example.com:443 -servername example.com -no_tls1_3
openssl s_client -connect example.com:443 -servername example.com -tlsextdebug -msg

openssl s_client -connect example.com:443 -CAfile ca.pem -verify_return_error
openssl s_client -connect example.com:443 -cert client.pem -key client.key   # mTLS

openssl s_client -starttls smtp -connect mail.example.com:25
openssl s_client -starttls imap -connect mail.example.com:143
openssl s_client -starttls postgres -connect db.example.com:5432
```

The `-msg` flag dumps every record header in human-readable form. `-tlsextdebug` decodes extensions (SNI, ALPN, supported groups, key share). Use them.

## openssl x509 (read certs)

```bash
openssl x509 -in cert.pem -text -noout
openssl x509 -in cert.pem -noout -subject -issuer -dates
openssl x509 -in cert.pem -noout -ext subjectAltName
openssl x509 -in cert.pem -noout -fingerprint -sha256
openssl x509 -in cert.pem -noout -pubkey | openssl pkey -pubin -text -noout

openssl crl2pkcs7 -nocrl -certfile chain.pem | openssl pkcs7 -print_certs -noout
```

## openssl req / genrsa / pkey (make certs)

```bash
openssl genrsa -out ca.key 4096
openssl req -x509 -new -key ca.key -sha256 -days 3650 -subj "/CN=test-CA" -out ca.crt

openssl genrsa -out server.key 2048
openssl req -new -key server.key -subj "/CN=server.local" -out server.csr
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
    -out server.crt -days 825 -sha256 \
    -extfile <(printf "subjectAltName=DNS:server.local,IP:10.0.0.1")

openssl pkcs12 -export -out bundle.p12 -inkey server.key -in server.crt -certfile ca.crt
```

## curl

```bash
curl -v https://example.com/                       # show TLS handshake summary
curl --tlsv1.3 https://example.com/
curl --http1.1 https://example.com/
curl --http2 https://example.com/
curl --http3 https://example.com/                  # needs curl built with QUIC

curl --cacert ca.crt https://server.local/
curl --resolve server.local:443:10.0.0.1 https://server.local/
curl --cert client.pem --key client.key https://api.example.com/
curl -k https://self-signed.local/                 # accept anything (debug only)

curl -w "namelookup=%{time_namelookup}s connect=%{time_connect}s appconnect=%{time_appconnect}s starttransfer=%{time_starttransfer}s total=%{time_total}s\n" -o /dev/null -s https://example.com/
```

`SSLKEYLOGFILE` for decrypting your own traffic in Wireshark:

```bash
SSLKEYLOGFILE=/tmp/keylog.txt curl https://example.com/
```

## testssl.sh

Comprehensive TLS audit. Slow but thorough:

```bash
testssl.sh https://example.com/
testssl.sh -p https://example.com/                 # protocols only (fast)
testssl.sh -e https://example.com/                 # ciphers only
testssl.sh -V                                      # tested versions
```

## sslyze (alternative)

```bash
sslyze --regular example.com:443
```

## Wireshark

- Filter `tls.handshake.type == 1` -> ClientHello.
- Look at "Server Name Indication" for SNI.
- Look at "Supported Groups" / "Key Share" extensions.
- For TLS 1.3, ServerHello shows "Cipher Suite". The application data after that is encrypted.
- Use `SSLKEYLOGFILE` to decrypt your own session.

## Common debug recipes

```bash
echo "QUIT" | openssl s_client -connect example.com:443 -servername example.com 2>&1 | grep -E 'subject=|issuer=|Verify return|Protocol|Cipher'

openssl s_client -connect example.com:443 -servername example.com 2>/dev/null </dev/null | openssl x509 -noout -dates

openssl s_client -connect example.com:443 -servername example.com -reconnect </dev/null 2>&1 | grep "Reused"   # check session reuse

curl -v https://example.com/ 2>&1 | grep -E '^[*<>]'   # full TLS + HTTP details
```

## Common errors decoded

- `unable to get local issuer certificate` - chain doesn't include intermediates, or your CA bundle is wrong.
- `certificate verify failed` - cert is for a different name (check SAN), or expired, or self-signed.
- `tlsv1 alert handshake failure` - cipher mismatch, or SNI required and missing.
- `wrong version number` - you're talking TLS to a non-TLS service (or HTTP to HTTPS).
- `bad record mac` - usually MTU / middlebox issue, or wrong keylog file.
