## Learning goals

- Explain *persistent connections* (keep-alive) and why HTTP/1.1 has head-of-line blocking *per connection*.

Persistent connection is about using the same TCP connection through socket, which is not closed after request-response pair. Allowing to send multiple requests/responses through single TCP connection.

Head-of-line blocking arised because, single connection now serves multiple connection. Each request/response served back-to-back, so transmitting huge resources could block transmitting other resources.

- Explain HTTP/2's binary framing, streams, HPACK, and what HoL blocking it solves vs. what it doesn't.

HTTP/2 uses Framing sublayer, so that each response/request is splitted to HTTP frame. It's used to avoid HOL problem, because frames are usually lower in size, so that we can transmit frames concurrently.

Streams are uniquely identified transport channels, that used to concurrently transmit request/response data. So that frames of request transmitted in channel, if transmit fails, it doesn't affect other channels. Usually multuple streams lives in single TCP connection.

HPACK is the header compression format used by HTTP/2. HPACK compresses headers using:

1. Static table - just index inside built-in table of common headers.
2. Dynamic table - A TCP connection-specific table dynamically, shared for each stream over connection.
3. Huffman encoding

Without compression, header overhead would explode.

HTTP/2 solved HTTP HOL blocking by introducing frames, now multiple requests are send simulatenously. But HTTP/2 do not solve TCP HOL blocking, when packet is lost all other subsequent packets _(with frames of different streams)_ need to wait until this packet is retransmitted.


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

CDN and reverse proxy are almost the same thing, reverse proxy usually don't have to be local to user to be efficient. Besides caching it also used to hide and protect backend servers, but caching works in the same terms.


Cache-Control controls cache in browser and shared caches _(Cache that exists between the origin server and clients: Proxies, CDNs)_.

Etag - entity tag is used as identifier of resources in response message. It lets caches be more efficient and save bandwidth, as a web server does not need to resend a full response if the content has not changed. If the resource at a given URL changes, a new ETag value must be generated. A comparison of them can determine whether two representations of a resource are the same.

With the help of the `ETag` and the `If-Match` headers, you can detect mid-air edit collisions (conflicts). When saving changes to a wiki page (posting data), the POST request will contain the If-Match header containing the ETag values to check freshness against.

It can also be used with header `If-None-Match` with value of previosly obtained `Etag`. The server compares the client's ETag (sent with If-None-Match) with the ETag for its current version of the resource, and if both values match (that is, the resource has not changed), the server sends back a 304 Not Modified status, without a body, which tells the client that the cached version of the response is still good to use (fresh).

The HTTP `Vary` response header. describes the parts of the request message (aside from the method and URL) that influenced the content of the response it occurs in. Including a Vary header ensures that responses are separately cached based on the headers listed in the Vary field.

- Understand cookies and how `SameSite` / `Secure` / `HttpOnly` interact with browser security.

`SameSite` - Controls whether or not a cookie is sent with cross-site requests: that is, requests originating from a different site, including the scheme, from the site that set the cookie.

`Secure` - requires to use cookie only with https

`HttpOnly` - Forbids JavaScript from accessing the cookie, for example, through the Document.cookie property.


## Execises

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

Stream priority is used to give HTTP message to be sent over stream. Without priority all streams have the same priority. Priority allows to send frames of given message through more important _(higher priority)_ stream.


- For a small static site, would you bother enabling HTTP/3? Why or why not?

Small static site are perfect candidate to use multiplexing introduced in HTTP/2 and HTTP/3 to open less connections on the server. If HTTP/2 already used it's not that profitable to move to HTTP/3. Also DNS or reverse proxy caching may take the most workload - so that HTTP/3 is redundant.

