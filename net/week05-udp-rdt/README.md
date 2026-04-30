# Week 5 - Ch3 part 1: Transport multiplexing + UDP + reliable transfer principles

This week is about *first principles* of transport: how do you build reliability on top of an unreliable channel? You'll implement Stop-and-Wait and Selective Repeat over UDP yourself.

## Reading

- Kurose & Ross 8e, 3.1-3.4 (~50 pages):
  - 3.1 Introduction and transport-layer services
  - 3.2 Multiplexing and demultiplexing
  - 3.3 Connectionless transport: UDP
  - 3.4 Principles of reliable data transfer
- RFC 768 (UDP) - 3 pages, read it.
- (skim) RFC 5681 (TCP Congestion Control) for the principles vocabulary - we use it next week.

## Learning goals

- Explain what *socket multiplexing/demultiplexing* really means (kernel uses `(src ip, src port, dst ip, dst port, proto)` tuple).
- Explain UDP semantics: best-effort, connectionless, no congestion control, datagram-bounded.
- Reason about Stop-and-Wait, Go-Back-N, Selective Repeat. What window sizes and sequence-number widths do they need?
- Reproduce loss in the lab and observe how a naive sender behaves.

## Lab

**Goal:** make a UDP link lossy and watch your reliable transfer protocol do its thing.

1. Bring up the two-host topology:
   ```bash
   cd lab/netns
   sudo bash two-hosts.sh
   ```
2. Add 10% packet loss on h2's interface:
   ```bash
   sudo ip netns exec h2 tc qdisc add dev veth-h2 root netem loss 10%
   ```
3. Run a UDP echo and observe:
   ```bash
   sudo ip netns exec h2 socat -v UDP4-LISTEN:9000,fork SYSTEM:'cat' &
   sudo ip netns exec h1 bash -c 'for i in $(seq 1 50); do echo "msg-$i" | nc -u -w1 10.0.0.2 9000; done'
   ```
4. Capture and look at how many got through:
   ```bash
   sudo ip netns exec h1 tcpdump -ni veth-h1 -w /tmp/udp-loss.pcap udp port 9000
   ```
5. Tear down with `sudo bash teardown.sh`.

## Exercises

1. **Conceptual.** In `notes.md`:
   - Why is UDP useful when TCP exists? Give 3 real-world examples (DNS, QUIC, RTP, gaming, ...).
   - In Go-Back-N with sequence numbers in [0..N-1], what's the maximum window size? Why?
   - In Selective Repeat, why must the window be at most N/2?
2. **Practical - implement Stop-and-Wait.** In `exercises/05-rdt-saw.py` build a sender + receiver over UDP that:
   - Numbers messages with a 1-bit sequence number.
   - Receiver ACKs each correctly-numbered message.
   - Sender waits for ACK with a timeout (200ms is fine), retransmits on timeout.
   - Survives 10% loss (use `tc netem loss 10%`).
3. **Practical - implement Selective Repeat.** In `exercises/05-rdt-sr.py`, build the SR variant with window size 4 and 3-bit sequence numbers. Buffer out-of-order packets at the receiver. Cumulative + selective ACKs.
4. **Stretch.** Add `tc netem reorder 25%` + `delay 50ms` and see what your protocols do. Where do they break?

## Self-check

- Without notes, draw a Stop-and-Wait timing diagram with one lost data packet and one lost ACK.
- What does the kernel use to deliver a UDP datagram to the right socket on the destination host?
- Why does Wireshark show "UDP checksum bad: maybe" sometimes?

## Useful one-liners

```bash
sudo tc qdisc add dev veth-h2 root netem loss 10%
sudo tc qdisc change dev veth-h2 root netem loss 10% delay 100ms
sudo tc qdisc del dev veth-h2 root

socat -v UDP4-LISTEN:9000,fork SYSTEM:'cat'   # one-line echo
nc -u -w1 10.0.0.2 9000                       # send a UDP line
```
