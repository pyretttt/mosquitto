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