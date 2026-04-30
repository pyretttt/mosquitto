# Week 6 - Ch3 part 2: TCP + congestion control + QUIC

The single most important week of the course. By the end you should be able to look at a TCP capture and a `ss -ti` snapshot and tell what's happening.

## Reading

- Kurose & Ross 8e, 3.5-3.8 (~70 pages):
  - 3.5 Connection-oriented transport: TCP
  - 3.6 Principles of congestion control
  - 3.7 TCP congestion control
  - 3.8 Evolution of transport-layer functionality (QUIC)
- RFC 9293 (consolidated TCP, 2022) - skim sections 1-3.
- RFC 5681 (TCP Congestion Control)
- RFC 9000 (QUIC) - read sections 1-5
- "BBR: Congestion-Based Congestion Control" - Cardwell et al. - https://queue.acm.org/detail.cfm?id=3022184
- (optional) "The Illustrated TLS Connection" + "HTTP/3 explained" if you want to see QUIC end-to-end.

## Learning goals

- Walk through TCP 3-way handshake and 4-way close (or simultaneous close) on the wire.
- Explain `seq`, `ack`, sliding window, fast retransmit, fast recovery, RTO.
- Describe slow start, congestion avoidance, the role of `ssthresh`.
- Compare CUBIC (default Linux) vs BBR. Know what each one's "mental model" is (loss-based vs model-based).
- Articulate what QUIC adds over TCP+TLS+HTTP/2: faster handshake, no head-of-line across streams, stream multiplexing native, connection migration.

## Lab

**Goal:** capture a TCP flow's full lifecycle and graph cwnd over time across CC algorithms.

1. Bring up the router lab so we have a "link" to abuse:
   ```bash
   cd lab/netns
   sudo bash router-3-nodes.sh
   ```
2. Install `iperf3` in each netns or use the existing one.
3. Baseline:
   ```bash
   sudo ip netns exec h2 iperf3 -s &
   sudo ip netns exec h1 iperf3 -c 10.0.1.1 -t 10
   ```
4. Add 50ms delay + 1% loss on the router's outbound:
   ```bash
   sudo ip netns exec r tc qdisc add dev veth-rh2 root netem delay 50ms loss 1%
   ```
5. Capture cwnd evolution while iperf runs (in another shell):
   ```bash
   sudo ip netns exec h1 ss -ti dst 10.0.1.1 | head -50
   ```
6. Switch CC algorithm and rerun:
   ```bash
   sudo ip netns exec h1 sysctl -w net.ipv4.tcp_congestion_control=bbr
   sudo ip netns exec h1 iperf3 -c 10.0.1.1 -t 30 -i 1
   sudo ip netns exec h1 sysctl -w net.ipv4.tcp_congestion_control=cubic
   sudo ip netns exec h1 iperf3 -c 10.0.1.1 -t 30 -i 1
   ```
7. Sample cwnd every 100ms during the run with a small loop:
   ```bash
   while true; do sudo ip netns exec h1 ss -ti | grep cwnd; sleep 0.1; done > exercises/06-cwnd-cubic.log
   ```

## Exercises

1. **Conceptual.** In `notes.md`:
   - Walk through the 3-way handshake byte-by-byte (SYN, SYN-ACK, ACK). What sequence/ack numbers do you expect?
   - Solve 2 end-of-chapter problems on TCP throughput / RTT estimation.
   - Why does QUIC need 1-RTT (or 0-RTT) for the first request, while TCP+TLS-1.3 needs 2-RTT? Sketch both timelines.
2. **Practical - cwnd graph.** Run `iperf3 -c h2 -t 30 -i 1` under both CUBIC and BBR on the same lossy link. Capture cwnd via `ss -ti` sampled every 100ms. Plot the two curves (matplotlib or a quick gnuplot). Save as `exercises/06-cwnd.png`. Comment in `notes.md` on the difference.
3. **Practical - retransmit forensics.** Force a high loss rate (`tc netem loss 10%`) and capture a flow. In Wireshark, find a *fast retransmit* (3 dup ACKs). Identify it by hand and write down the sequence numbers in `notes.md`.
4. **Stretch - QUIC inspection.** Use `curl --http3 -v https://cloudflare-quic.com/` and capture with `tcpdump -i any -w quic.pcap udp port 443`. Open in Wireshark - QUIC frames are visible up to the encrypted parts. Use `SSLKEYLOGFILE` + `--http3` to dump session keys, then point Wireshark at the keylog file to decrypt.

## Self-check

- Without notes: explain `seq` and `ack` with an example where the receiver has already ACKed bytes 0-99 and the next data segment is bytes 100-199.
- What's the difference between RTO and RTT?
- What does `ss -ti` show that `tcpdump` doesn't, and vice versa?
- Why is BBR more aggressive than CUBIC on a lossy fiber link?
- What are 3 reasons HTTP/3 (over QUIC) is faster than HTTP/2 (over TCP+TLS)?

## Useful one-liners

```bash
sysctl net.ipv4.tcp_available_congestion_control
sysctl -w net.ipv4.tcp_congestion_control=bbr

ss -ti                     # in-kernel TCP info per socket: cwnd, rtt, retrans
ss -tn dst :443            # only sockets to port 443

tcpdump -ni any -w cap.pcap 'tcp port 80 and (tcp[tcpflags] & (tcp-syn|tcp-fin) != 0)'

tc qdisc add dev eth0 root netem delay 50ms loss 1%
tc qdisc del dev eth0 root
```
