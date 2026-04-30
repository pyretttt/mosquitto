# Week 3 - Ch2: HTTP and the Web

HTTP is the protocol you'll see most often in production. By the end of this week you should be able to read raw HTTP/1.1 off a wire and explain why HTTP/2 and HTTP/3 exist.

## Reading

- Kurose & Ross 8e, 2.1-2.3 (~25 pages):
  - 2.1 Principles of network applications
  - 2.2 The Web and HTTP
  - 2.3 SMTP / mail (skim)
- RFC 9110 (HTTP semantics) - read sections 1-3 + 9 (status codes)
- RFC 9112 (HTTP/1.1) - skim
- RFC 9113 (HTTP/2) - read section 1-2 (binary framing, streams)
- (optional) "HTTP/3 explained" book by Daniel Stenberg: https://http3-explained.haxx.se/

## Learning goals

- Trace an HTTP/1.1 request/response on the wire (request line, headers, blank line, body).
- Explain *persistent connections* (keep-alive) and why HTTP/1.1 has head-of-line blocking *per connection*.
- Explain HTTP/2's binary framing, streams, HPACK, and what HoL blocking it solves vs. what it doesn't.
- Know where caching sits (browser, CDN, reverse proxy) and what `Cache-Control`, `ETag`, `Vary` do.
- Understand cookies and how `SameSite` / `Secure` / `HttpOnly` interact with browser security.

## Lab

**Goal:** capture HTTP/1.1, HTTP/2, and HTTP/3 traffic against the same nginx upstream and compare.

1. Bring up the proxy lab:
   ```bash
   cd lab/docker/proxy-lab
   docker compose up -d
   ```
2. Capture three flavors:
   ```bash
   docker compose exec debug tcpdump -ni eth0 -w /tmp/h1.pcap tcp port 80 &
   docker compose exec client curl -v --http1.1 http://proxy/
   docker compose exec debug pkill tcpdump

   docker compose exec debug tcpdump -ni eth0 -w /tmp/h2.pcap tcp port 80 &
   docker compose exec client curl -v --http2-prior-knowledge http://proxy/
   docker compose exec debug pkill tcpdump
   ```
3. Compare the two pcaps in Wireshark. Note:
   - HTTP/1.1: headers in cleartext, request lines obvious.
   - HTTP/2: binary frames (HEADERS, DATA, SETTINGS), HPACK-compressed headers.
4. For HTTP/3, point `curl` (with HTTP/3 support, available in netshoot) at a real site:
   ```bash
   docker compose exec client curl --http3 -v https://cloudflare-quic.com/
   ```

## Exercises

1. **Conceptual.** Answer in `notes.md`:
   - Why does HTTP/1.1 with pipelining never really worked in practice?
   - In HTTP/2, what does *stream priority* do? What does it *not* solve?
   - For a small static site, would you bother enabling HTTP/3? Why or why not?
2. **Practical - write your own HTTP/1.1 client.** In `exercises/03-raw-http.py`, build a Python client over a raw `socket` that:
   - Opens a TCP connection.
   - Sends a properly-formed `GET / HTTP/1.1\r\nHost: ...\r\n\r\n` request.
   - Reads the response, parses the status line and headers.
   - Handles `Transfer-Encoding: chunked` correctly (read chunk-size hex, then chunk-data, terminate on `0\r\n\r\n`).
   - Handles `Content-Length:` if present.
   - Compare the bytes-on-the-wire against `curl --trace-ascii -` for the same URL. They should be byte-identical.
3. **Stretch.** Configure caching on the proxy (`proxy_cache_path` + `proxy_cache`) and capture a request that produces `X-Cache-Status: MISS` then one that produces `HIT`. Note RTT and bytes saved.

## Self-check

- Without notes: list 5 HTTP status codes you should know cold and what they mean.
- What's the wire difference between `Connection: close` and `Connection: keep-alive`?
- Why does HTTP/2 exist if HTTP/1.1 supports keep-alive and pipelining?
- Where does TLS sit relative to HTTP/2 in the stack? Where does it sit for HTTP/3?

## Useful one-liners

```bash
curl -v http://example.com/                   # full request/response

curl --trace-ascii - http://example.com/      # see exact bytes

curl -I https://example.com/                  # HEAD only

curl --resolve foo.com:443:1.2.3.4 ...        # override DNS

curl -w '%{time_namelookup} %{time_connect} %{time_appconnect} %{time_starttransfer} %{time_total}\n' -o /dev/null -s https://example.com/
```
