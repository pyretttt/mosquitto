# Exercices

## Conceptual

- You have a 100 Mbps link, RTT 20 ms, MSS 1460. What's the BDP in bytes? What window size do you need to fully utilize the link?

100 mb per sec is 1e+8 bits per sec. RTT is round trip time 20 milliseconds. MSS maximum segment size is 1460 bytes.

BDP is Badnwidth delay product is Bandwidth * RTT.

Bandwidth = 100mb per sec.
RTT = 0.02 sec

BDP = 100 * 1e6 / 8 * 0.02 = 250000 bytes

We need at least 250000 congestion window. If max size of segment is 1460, we need at least 172 segment (MSS - sized) congestion window.

- Same link, but RTT 200 ms (intercontinental). What changes?

if RTT = 0.2 sec then BDP = 2.500.000 bytes
MSS sized widnow is now 1713. Because now more data can be "in link" at any given moment, we need more buffer storage to be able to store it.

- Solve 2-3 of the book's end-of-chapter problems on delay/queuing.

Done


## Practical

- `ping -c 50 1.1.1.1` to get RTT distribution. Note p50, p99, jitter.

median, p99, jitter: 0.18884 0.19652319999999998 0.018142857142857145

## Stretch
Capture an HTTPS page load to a real site (e.g. `curl -v https://example.com`). In Wireshark, identify the TLS Client Hello and look up the SNI extension.

Transport Layer Security
    [Stream index: 10]
    TLSv1.3 Record Layer: Handshake Protocol: Client Hello
        Content Type: Handshake (22)
        Version: TLS 1.0 (0x0301)
        Length: 1557
        Handshake Protocol: Client Hello
            Handshake Type: Client Hello (1)
            Length: 1553
            Version: TLS 1.2 (0x0303)
            Random: f4f29ca2473dbfd17e562c467d24cc667c2366290be89f0aaf853b46b30b0439
            Session ID Length: 32
            Session ID: ff3faf4548ad1e8e7e2538381602a80f75328a42fc1ad796dd842c1f788216ab
            Cipher Suites Length: 60
            Cipher Suites (30 suites)
            Compression Methods Length: 1
            Compression Methods (1 method)
            Extensions Length: 1420
            Extension: renegotiation_info (len=1)
            Extension: server_name (len=16) name=example.com
            Extension: ec_point_formats (len=4)
            Extension: supported_groups (len=18)
            Extension: application_layer_protocol_negotiation (len=14)
            Extension: encrypt_then_mac (len=0)
            Extension: extended_master_secret (len=0)
            Extension: post_handshake_auth (len=0)
            Extension: signature_algorithms (len=54)
            Extension: supported_versions (len=5) TLS 1.3, TLS 1.2
            Extension: psk_key_exchange_modes (len=2)
            Extension: key_share (len=1258) X25519MLKEM768, x25519
            [JA4: t13d3012h2_1d37bd780c83_882d495ac381]
            [JA4_r […]: t13d3012h2_002f,0033,0035,0039,003c,003d,0067,006b,009c,009d,009e,009f,1301,1302,1303,c009,c00a,c013,c014,c023,c024,c027,c028,c02b,c02c,c02f,c030,cca8,cca9,ccaa_000a,000b,000d,0016,0017,002b,002d,0031,0033,ff01_0905,0906,0904,]
            [JA3 Fullstring: 771,4866-4867-4865-49196-49200-159-52393-52392-52394-49195-49199-158-49188-49192-107-49187-49191-103-49162-49172-57-49161-49171-51-157-156-61-60-53-47,65281-0-11-10-16-22-23-49-13-43-45-51,4588-29-23-30-24-25-256-257,0-1-2]
            [JA3: 446fb8bd613d17e5d44bd3b288ac94ca]

Couldn't be able to detect SNI extension

## Self-check

- Sketch the 5-layer stack and place HTTP, TCP, IP, Ethernet, fiber on it.

fiber -> Ethernet frame -> IP datagram -> TCP segment -> http message

- Without notes, write the formula for transmission delay and explain why it shrinks as link capacity grows.

d = L/R where R is link capacity, when it grows, less time is required to push L packets bits to link itself.

- What does Wireshark's "Follow TCP Stream" do, and when would you use "Follow HTTP Stream" instead?

Follow TCP stream shows all segments that belong to single session.

I would use HTTP stream when i want to trace request response pairs.
