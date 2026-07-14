## Learning goals

- Walk through TCP 3-way handshake and 4-way close (or simultaneous close) on the wire.

Handshake:
SYN ----->
    <----- SYNACK
ACK ----->

4 way close
FIN ------>
    <------ ACK
    <------ FIN
ACK ------>

- Explain `seq`, `ack`, sliding window, fast retransmit, fast recovery, RTO.

* `seq` field of tcp segment is used to specialize sequence number of the first bit inside TCP segment _(and correspondingly IP datagram)_

* `ack` is used to inform other side of tcp connection, what bit is expected to be received next

* The TCP sliding window does not restrict a range of bits. It restricts the amount (range of bytes) of data that can be sent without receiving an acknowledgment.

* fast retransmit happens when three duplicate acks are received, it means that succeding acked one is lost.

* fast recovery is tcp congestion control mechanism when transmission rate is increased non-linearly after congestion occurred.

* RTO - retransmission timeout is the amount of time TCP waits for an acknowledgment (ACK) after sending a segment before assuming that the segment was lost and retransmitting it.

- Describe slow start, congestion avoidance, the role of `ssthresh`.

* Classic TCP congestion control's slow start is about starting with 1 segment per RTT, and increasing sending rate each RTT.

* Congestion avoidance happens when congestion was met, `cwnd` is now about half of `cwnd` when congestion was met. TCP now increases `cwnd` by a single `MTT` per `RTT`

* Since `sshtresh` is half of `cwnd` when congestion was last detected it might be a bit reckless to keep doubling `cwnd` when it reaches or surpasses the value of `sshtresh`. Thus when, value of `cwnd` equals `sshtresh`, slow start ends and TCP transitions into Congestion-Avoidanc mode.


- Compare CUBIC (default Linux) vs BBR. Know what each one's "mental model" is (loss-based vs model-based).

CUBIC uses cubic function to fastly recover to maximum sending rate _(when congestion was met)_, near the end it slowly tests whether is there a congestion.
Once it recovered to `W_max` it test new maximum aggresively. So that it loss based

BBR on the other hand is statistical model which computes statistics in window to compute `max delivery rate observed` and `Minimum RTT observed`. It doesn't define congestion as loss.

- Articulate what QUIC adds over TCP+TLS+HTTP/2: faster handshake, no head-of-line across streams, stream multiplexing native, connection migration.

* QUIC allows for 0 RTT before first bit of data is sent

* QUIC streams are independent, they do not block each other

* If user change connection type from cellular to wifi it can migrate session without handshake and tls negotiation

## Exercises

1. **Conceptual.** In `notes.md`:
   - Walk through the 3-way handshake byte-by-byte (SYN, SYN-ACK, ACK). What sequence/ack numbers do you expect?

   It start with randomly generated seq_number:
   1. `SYN` SEQ_NUM 603812734 (RANDOMLY GENERATED)
   2. `SYN` + `ACK` SEQ_NUM 3874629843 `ACK` 603812735 (+1 OF SEQ NUM)
   3. `ACK` SEQ_NUM 603812735 (ACK FROM 2.) `ACK` 3874629844 (+1 OF PREV SEQ NUM)

   - Solve 2 end-of-chapter problems on TCP throughput / RTT estimation. ✅

   - Why does QUIC need 1-RTT (or 0-RTT) for the first request, while TCP+TLS-1.3 needs 2-RTT? Sketch both timelines.

    QUIC combines connection handshake with TLS handshake thereof it requires 1RTT _(It may be 0RTT if the connection is returning)_

    TCP on the other hand requires 1 handshake to establish tcp connection. And separate handshake to establish TLS connection. One after another

2. **Practical - cwnd graph.** Run `iperf3 -c h2 -t 30 -i 1` under both CUBIC and BBR on the same lossy link. Capture cwnd via `ss -ti` sampled every 100ms. Plot the two curves (matplotlib or a quick gnuplot). Save as `exercises/06-cwnd.png`. Comment in `notes.md` on the difference.

Captured measures, plotted cwnd. Reno vs Cubic, cubic has higher cwnd on average

3. **Practical - retransmit forensics.** Force a high loss rate (`tc netem loss 10%`) and capture a flow. In Wireshark, find a *fast retransmit* (3 dup ACKs). Identify it by hand and write down the sequence numbers in `notes.md`.



4. **Stretch - QUIC inspection.** Use `curl --http3 -v https://cloudflare-quic.com/` and capture with `tcpdump -i any -w quic.pcap udp port 443`. Open in Wireshark - QUIC frames are visible up to the encrypted parts. Use `SSLKEYLOGFILE` + `--http3` to dump session keys, then point Wireshark at the keylog file to decrypt.

## Self-check

- Without notes: explain `seq` and `ack` with an example where the receiver has already ACKed bytes 0-99 and the next data segment is bytes 100-199.

`seq` represents sequence number of the first byte in TCP segments payload w.r.t. tcp connection. `ack` represents next expected sequence number in the next received segment.

If receiver already ACKed bytes 0-99, then last ack was 100, next seq number will be `100`. Sender sends bytes 100-100 with `seq=100` and receiver acks with `ack=200`

- What's the difference between RTO and RTT?

RTT - Round trip time, time required to send data across channel between sender and receiver and in reverse.

RTO - Retransmission timeout, it measured from `estimatedRTT` and `estimatedDev`, which are used to measure of expected RTT.

- What does `ss -ti` show that `tcpdump` doesn't, and vice versa?



- Why is BBR more aggressive than CUBIC on a lossy fiber link?

BBRv1 do not track loss, so it still sends packets at the same rate. It assumes that packet loss is due to queue overflow, instead of lossy channel.

CUBIC on the other hand takes packet loss into account, so it can adjust to it better.

- What are 3 reasons HTTP/3 (over QUIC) is faster than HTTP/2 (over TCP+TLS)?

1. 1RTT before data is sent.
2. Independent data streams, no Block of line problem, when data is lost.
3. Change of network type does not require new connection
