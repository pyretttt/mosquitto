## Learning goals

- Explain what *socket multiplexing/demultiplexing* really means (kernel uses `(src ip, src port, dst ip, dst port, proto)` tuple).

Multiplexing/demultiplexing refers to transport layer service, that uses shared network layer _(which works with hosts)_, it allows application to application network.

- Explain UDP semantics: best-effort, connectionless, no congestion control, datagram-bounded.

Best-effort semantics means:

- No delivery guarantee: A UDP datagram may be lost and never reach the destination.
- No ordering guarantee: Datagrams can arrive out of order.
- No duplicate protection: A datagram may be delivered more than once.
- No error recovery: UDP does not retransmit lost or corrupted packets.
- No flow control: A fast sender can overwhelm a slow receiver.
- No congestion control: UDP does not reduce its sending rate when the network is congested.

connectionless - no handshake routine, 0RTT before data can be sent

no congestion control - application sends data at any rate

datagram-bounded - whole application message sent to socket, at the receiver the same whole message is ressembled from ip datagrams

- Reason about Stop-and-Wait, Go-Back-N, Selective Repeat. What window sizes and sequence-number widths do they need?

Stop-and-wait simple protocol, it requires 1 bit window width

Go-Back-N window width must be at most (Seq_Number - 1). It allows pipelining so outperforms StopAndWait

Selective repeat outperforms Stop-And-Wait and transfers less data through network than Go-Back-N. Window width must be atmost (Seq_Number/2)

- Reproduce loss in the lab and observe how a naive sender behaves.

Done!

## Exercises

1. **Conceptual.** In `notes.md`:
   - Why is UDP useful when TCP exists? Give 3 real-world examples (DNS, QUIC, RTP, gaming, ...).

It's useful when latency is dominant over reliability, for example streaming, dns queries (can be cached hard) and etc.
Games use UDP, because low latency is over reliability, video streaming is the same. QUIC uses udp, but provides reliable data transfer.

   - In Go-Back-N with sequence numbers in [0..N-1], what's the maximum window size? Why?

Maximum window size is `N - 1`, it's not possible to make it `N` because retransmission then could be though as next cycle. That unused sequence number acts as a “gap” between the current window and the next one after wraparound. So the receiver never has to guess whether a packet with sequence number 0 is old or new

   - In Selective Repeat, why must the window be at most N/2?

If W > N/2, then after sequence number wraparound, an old packet can fall into the receiver’s current receive window, making it impossible to determine whether it’s a retransmission or a new transmission.

Sequence numbers:
```
0 1 2 3 4 5 6 7
```

Sender sends

```
0 1 2 3 4
```

Receiver window is now

```
0 1 2 3 4
```

Now imagine all ACKs are lost. The sender times out and retransmits packet 0. The receiver now asks - Is this:
- the old packet 0 being retransmitted because its ACK was lost?
- or the new packet 0 after sequence numbers wrapped around?

It must be at most N/2

2. **Practical - implement Stop-and-Wait.** In `exercises/05-rdt-saw.py` build a sender + receiver over UDP that:
   - Numbers messages with a 1-bit sequence number.
   - Receiver ACKs each correctly-numbered message.
   - Sender waits for ACK with a timeout (200ms is fine), retransmits on timeout.
   - Survives 10% loss (use `tc netem loss 10%`).
3. **Practical - implement Selective Repeat.** In `exercises/05-rdt-sr.py`, build the SR variant with window size 4 and 3-bit sequence numbers. Buffer out-of-order packets at the receiver. Cumulative + selective ACKs.

Both launched with `docker run -u 0 --privileged  --volume ./lab:/lab --volume week05-udp-rdt:/udp -it jonlabelle/network-tools sh`

Then

```
lab/netns/two-hosts.sh
ip netns exec h1 tc qdisc add dev veth-h1 root netem loss 25%
ip netns exec h2 tc qdisc add dev veth-h2 root netem loss 25%
ip netns exec h2 python3 /udp/exercises/05-rdt-saw.py --receiver &
ip netns exec h1 python3 /udp/exercises/05-rdt-saw.py --sender
```

For the `Stop-And-Wait` I have got:
```
Timeout waiting for ACK
Received data: b'Hello, world 0'
Received data: b'Hello, world 1'
Received data: b'Hello, world 2'
Timeout waiting for ACK
Received data: b'Hello, world 3'
Received data: b'Hello, world 4'
Timeout waiting for ACK
Timeout waiting for ACK
Timeout waiting for ACK
Timeout waiting for ACK
Received data: b'Hello, world 5'
Received data: b'Hello, world 6'
Received data: b'Hello, world 7'
Received data: b'Hello, world 8'
Received data: b'Hello, world 9'
```

For the `Selective-Repeat` output of

```
lab/netns/two-hosts.sh
ip netns exec h1 tc qdisc add dev veth-h1 root netem loss 25%
ip netns exec h2 tc qdisc add dev veth-h2 root netem loss 25%
ip netns exec h2 python3 /udp/exercises/05-rdt-sr.py --receiver &
ip netns exec h1 python3 /udp/exercises/05-rdt-sr.py --sender
```

For the receiver is:


```
Received data segment: 0, data: Hello, world 0
Flushed segment to upper layer: 0
Received data segment: 1, data: Hello, world 1
Flushed segment to upper layer: 1
Received data segment: 2, data: Hello, world 2
Flushed segment to upper layer: 2
Received data segment: 3, data: Hello, world 3
Flushed segment to upper layer: 3
Received data segment: 5, data: Hello, world 5
Received data segment: 6, data: Hello, world 6
Received data segment: 7, data: Hello, world 7
Received data segment: 4, data: Hello, world 4
Flushed segment to upper layer: 4
Flushed segment to upper layer: 5
Flushed segment to upper layer: 6
Flushed segment to upper layer: 7
Received data segment: 0, data: Hello, world 8
Flushed segment to upper layer: 0
Received data segment: 1, data: Hello, world 9
Flushed segment to upper layer: 1
```

For the sender is:

```
Sent data segment: 0
Sent data segment: 1
Sent data segment: 2
Sent data segment: 3
Received ACK for segment: 0
Sent data segment: 4
Received ACK for segment: 2
Received ACK for segment: 3
Timeout waiting for ACK
Received ACK for segment: 1
Sent data segment: 5
Sent data segment: 6
Sent data segment: 7
Received ACK for segment: 5
Received ACK for segment: 6
Received ACK for segment: 7
Timeout waiting for ACK
Received ACK for segment: 4
Sent data segment: 0
Sent data segment: 1
Received ACK for segment: 1
Timeout waiting for ACK
Received ACK for segment: 0
```

4. **Stretch.** Add `tc netem reorder 25%` + `delay 50ms` and see what your protocols do. Where do they break?

I don't think they break, they correctly handle timeout and corrupted packages


## Self-check

- Without notes, draw a Stop-and-Wait timing diagram with one lost data packet and one lost ACK.
---------
         | \        |
         |   \ seq=1|
         |      \   |
timeout  |          |
_________|          |
         |  \seq=1  |
         |    \     |
         |       \  |
         |         \|
         | ack    / |
         |       /  |
timeout  |     /    |
_________|          |
         |  \seq=1   |
         |    \     |
         |       \  |
         |         \|
         |   ack  / |
         |     /    |
         |  /       |
         |/         |

- What does the kernel use to deliver a UDP datagram to the right socket on the destination host?
It looks at destination port inside udp segment, it also validates ip address from ip datagram.

- Why does Wireshark show "UDP checksum bad: maybe" sometimes?
Often shows outgoing UDP packets as having a "bad checksum" due to hardware checksum offloading. Your computer’s network card calculates the checksum after Wireshark intercepts the packet in the operating system. The packet leaves your machine perfectly intact, but Wireshark flags the early capture as incorrect.