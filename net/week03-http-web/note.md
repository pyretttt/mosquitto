## Learning goals

- Explain *persistent connections* (keep-alive) and why HTTP/1.1 has head-of-line blocking *per connection*.

Persistent connection is about using the same TCP connection through socket, which is not closed after request-response pair. Allowing to send multiple requests/responses through single TCP connection.

Head-of-line blocking arises because a single connection now serves multiple requests. Each response is sent in request order, so a slow or large response blocks everything behind it on that connection.

- Explain HTTP/2's binary framing, streams, HPACK, and what HoL blocking it solves vs. what it doesn't.

HTTP/2 uses a binary framing sublayer: each request/response is split into frames (HEADERS, DATA, etc.) that can be **interleaved** across streams on one TCP connection. That removes HTTP/1.1 application-layer HoL—you are not stuck waiting for one full response before another stream's bytes can flow.

Streams are uniquely identified channels for one request/response exchange. Many streams share a single TCP connection; the peer multiplexes by stream id. A problem on one stream does not force you to drain another stream's whole response first (unlike pipelining).

Note: all streams still share the **same TCP connection**, so TCP-level loss still stalls every stream until the lost packet is retransmitted.

HPACK is the header compression format used by HTTP/2. HPACK compresses headers using:

1. Static table - just index inside built-in table of common headers.
2. Dynamic table - A TCP connection-specific table dynamically, shared for each stream over connection.
3. Huffman encoding

Without compression, header overhead would explode.

HTTP/2 solves **HTTP-layer** HoL via multiplexed streams and interleaved frames. It does **not** solve **TCP** HoL blocking: when a packet is lost, all later bytes on that connection (including frames from other streams) wait for the retransmit.


- Know where caching sits (browser, CDN, reverse proxy) and what Cache-Control, ETag, Vary do.

Cache can reside in browser if `Cache Control:max-age=3600` is used.
CDN is global distributed cache system, used to deliver content faster. If content is already cached at the CDN edge: it's returned immediately, otherwise origin server is contacted. Commonly cached:
* images
* CSS
* JavaScript
* videos
* fonts
* static HTML
* API responses (sometimes)

A CDN is essentially a **distributed reverse proxy at the edge** (close to users). An origin-side reverse proxy sits in front of your app servers; caching semantics are similar (`Cache-Control`, etc.), but a CDN's main win is geographic proximity and scale at the edge.


Cache-Control controls cache in browser and shared caches _(Cache that exists between the origin server and clients: Proxies, CDNs)_.

Etag - entity tag is used as identifier of resources in response message. It lets caches be more efficient and save bandwidth, as a web server does not need to resend a full response if the content has not changed. If the resource at a given URL changes, a new ETag value must be generated. A comparison of them can determine whether two representations of a resource are the same.

With the help of the `ETag` and the `If-Match` headers, you can detect mid-air edit collisions (conflicts). When saving changes to a wiki page (posting data), the POST request will contain the If-Match header containing the ETag values to check freshness against.

It can also be used with header `If-None-Match` with value of previously obtained `ETag`. The server compares the client's ETag (sent with If-None-Match) with the ETag for its current version of the resource, and if both values match (that is, the resource has not changed), the server sends back a 304 Not Modified status, without a body, which tells the client that the cached version of the response is still good to use (fresh).

The HTTP `Vary` response header. describes the parts of the request message (aside from the method and URL) that influenced the content of the response it occurs in. Including a Vary header ensures that responses are separately cached based on the headers listed in the Vary field.

- Understand cookies and how `SameSite` / `Secure` / `HttpOnly` interact with browser security.

`SameSite` - Controls whether or not a cookie is sent with cross-site requests: that is, requests originating from a different site, including the scheme, from the site that set the cookie.

`Secure` - requires to use cookie only with https

`HttpOnly` - Forbids JavaScript from accessing the cookie, for example, through the Document.cookie property.


## Exercises

- Why does HTTP/1.1 with pipelining never really worked in practice?

It failed to adopt, because HOL problem arised that was strenghten by TCP failures _(retransmission of failed packets by tcp)_.

```
REQ A -> slow
REQ B -> fast

must return:
RESP A first
RESP B second
```

One slow request blocks all later responses.

- In HTTP/2, what does *stream priority* do? What does it *not* solve?

**Does:** Priority is a hint from the client about which streams should get bandwidth first when the connection is constrained (weighted dependency tree). Each request/response already has its own stream; priority ranks streams relative to each other.

**Does not solve:** TCP-level HoL (packet loss still blocks all streams on that connection). It does not guarantee faster page load—many servers ignore or simplify priorities. It does not remove the need to complete a large response on a stream; it only influences scheduling among competing streams.

- For a small static site, would you bother enabling HTTP/3? Why or why not?

Usually **not** worth the effort if HTTP/2 plus CDN or browser caching already covers you. A tiny static site has few parallel assets, so multiplexing wins are small. HTTP/3 adds QUIC/UDP deployment and ops complexity (and UDP is sometimes blocked) for modest gain on good networks. HTTP/3 shines more on lossy mobile/Wi‑Fi paths; for a simple site, cache hits and TLS termination at the edge often matter more than the HTTP version.


## Self-check

- Without notes: list 5 HTTP status codes you should know cold and what they mean.

200 - OK, 204 - No Content, 302 - redirect, 304 - Not Modified (cache still valid), 404 - not found, 500 - server error

- What's the wire difference between `Connection: close` and `Connection: keep-alive`?

`Connection: close` — the peer should close the TCP socket after this response (no further requests on that connection). `Connection: keep-alive` — leave the TCP connection open for another request/response round-trip (HTTP/1.1 defaults to persistent unless either side sends `close`). Idle connections are closed after a timeout.

- Why does HTTP/2 exist if HTTP/1.1 supports keep-alive and pipelining?

HTTP/1.1 pipelining performs poorly, because of HOL problem, which blocked small resources while big resource being transferred. Responses must be in order of requests.

- Where does TLS sit relative to HTTP/2 in the stack? Where does it sit for HTTP/3?

Stack (bottom to top):

- **HTTP/1.1:** TCP → TLS → HTTP text
- **HTTP/2:** TCP → TLS → HTTP/2 frames (TLS is **below** framing; frames are carried inside the encrypted byte stream, not "after" TLS in the stack)
- **HTTP/3:** UDP → QUIC (integrates TLS 1.3 in the handshake) → HTTP/3 frames on QUIC streams (no TCP; encryption is part of QUIC, not a separate layer above it like TCP+TLS)