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
