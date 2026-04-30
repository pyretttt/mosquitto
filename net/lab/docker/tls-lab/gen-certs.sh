#!/usr/bin/env bash
# Generate a tiny CA and a server cert for the TLS lab.
# Run on your Linux VM. Outputs to ./certs/.

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p certs
cd certs

openssl genrsa -out ca.key 4096 2>/dev/null
openssl req -x509 -new -key ca.key -sha256 -days 3650 \
  -subj "/CN=lab-CA" \
  -out ca.crt

openssl genrsa -out server.key 2048 2>/dev/null
openssl req -new -key server.key \
  -subj "/CN=server.lab.local" \
  -out server.csr

cat > server.ext <<EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = server.lab.local
DNS.2 = localhost
IP.1  = 10.30.0.10
EOF

openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
  -out server.crt -days 825 -sha256 -extfile server.ext

echo "wrote certs/{ca.crt,ca.key,server.crt,server.key}"
echo "verify:"
openssl x509 -in server.crt -noout -subject -issuer -dates -ext subjectAltName
