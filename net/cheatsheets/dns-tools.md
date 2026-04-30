# DNS tools cheatsheet (`dig`, `host`, `kdig`, `drill`)

`dig` is the standard. `kdig` (from knot-dnsutils) is friendlier for DoT/DoH.

## dig basics

```bash
dig example.com                              # A record, default resolver
dig example.com AAAA                         # IPv6
dig example.com MX
dig example.com NS
dig example.com TXT
dig example.com SOA
dig example.com ANY                          # most servers refuse this now (RFC 8482)
dig example.com CAA
dig example.com SRV _http._tcp

dig @1.1.1.1 example.com                     # specific resolver
dig +short example.com                       # just the answer
dig +noall +answer example.com               # no other sections
dig +tcp example.com                         # force TCP
dig +trace example.com                       # iteratively walk root -> TLD -> auth
dig +norec example.com @<authoritative>      # no recursion
dig +bufsize=512 example.com                 # disable EDNS0 large
dig +dnssec example.com                      # request DNSSEC RRSIGs
```

## kdig (DoT, DoH)

```bash
kdig @1.1.1.1 +tls example.com               # DNS over TLS
kdig @cloudflare-dns.com +tls-ca +tls-host=cloudflare-dns.com example.com
kdig @https://1.1.1.1/dns-query +https example.com
```

## host (simple)

```bash
host example.com
host -t MX example.com
host -a example.com
```

## drill (ldns-utils)

```bash
drill example.com
drill -D example.com                          # DNSSEC chain
drill -T example.com                          # trace
```

## Reverse DNS

```bash
dig -x 8.8.8.8
host 8.8.8.8
```

## Useful debugging patterns

```bash
dig +trace cursor.com                         # which auth servers, what TTLs
dig +nssearch cursor.com                      # ask all NSes, compare answers
dig @<resolver> example.com +tcp +bufsize=4096 +dnssec   # full DNSSEC probe

dig @<resolver> example.com | grep -E 'flags|status'      # status: NOERROR/NXDOMAIN/SERVFAIL
```

`status` codes you'll see:

- `NOERROR` - normal answer (may be empty if no records of that type).
- `NXDOMAIN` - the name doesn't exist anywhere.
- `SERVFAIL` - resolver could not get a definitive answer.
- `REFUSED` - server refuses to answer (often because it's not authoritative and recursion is off).

## Local resolver behavior

```bash
cat /etc/resolv.conf
resolvectl status                             # systemd-resolved
resolvectl query example.com
sudo systemd-resolve --flush-caches

# macOS:
scutil --dns
sudo dscacheutil -flushcache
sudo killall -HUP mDNSResponder
```

## Capture DNS

```bash
sudo tcpdump -ni any -w /tmp/dns.pcap port 53
sudo tcpdump -ni any -A 'port 53'             # ASCII (works for plain UDP DNS)
```

## Common gotchas

- `dig` defaults to A. To check IPv6, you must say `AAAA` explicitly.
- `dig @resolver name +short` is great for scripting.
- A CNAME on the *apex* of a zone is forbidden by RFC 1034 - use ALIAS / ANAME if your DNS provider supports it.
- `TXT` records are quoted strings up to 255 chars each; longer ones are split (multi-string).
- DNSSEC failures often surface as `SERVFAIL`. Use `dig +cd` (checking-disabled) to bypass and see if data exists at all.
