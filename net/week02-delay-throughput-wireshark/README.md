# Week 2 - Ch1 part 2: Delay, loss, throughput, layering + Wireshark intro

The week you learn to *quantify* networks (delays, BDP) and learn the protocol-analysis tool (Wireshark) you'll lean on for the rest of the curriculum.

## Reading

- Kurose & Ross 8e, sections 1.4-1.7 (~30 pages):
  - 1.4 Delay, loss, and throughput in packet-switched networks
  - 1.5 Protocol layers and their service models
  - 1.6 Networks under attack (skim, we revisit in W12-W13)
  - 1.7 History of computer networking (read for fun)
- Wireshark Lab from the book: "Getting Started" - download the PDF + .pcap from http://gaia.cs.umass.edu/kurose_ross/wireshark.php
- (optional) "High Performance Browser Networking" Ch1 (Primer on Latency and Bandwidth): https://hpbn.co/primer-on-latency-and-bandwidth/

## Learning goals

- Decompose end-to-end delay into processing + queuing + transmission + propagation.
- Reason about *bandwidth-delay product (BDP)* and why it caps single-flow throughput.
- Explain what *encapsulation* means and which header lives at which layer.
- Capture, filter, and decode a packet in Wireshark.

## Lab

**Goal:** capture a real HTTP page load and identify each layer's headers.

1. Bring up the generic stack:
   ```bash
   cd lab/docker
   docker compose up -d
   ```
2. Capture from the server's namespace while making a request:
   ```bash
   docker compose exec debug tcpdump -ni eth0 -w /tmp/cap.pcap port 80 &
   docker compose exec client curl -v http://server/
   docker compose exec debug pkill tcpdump
   docker cp net-debug:/tmp/cap.pcap week02-delay-throughput-wireshark/captures/
   ```
3. Open `cap.pcap` in Wireshark on your Mac. Pick a SYN packet and identify, byte-by-byte:
   - Ethernet header (dst, src, ethertype)
   - IP header (version, TTL, protocol, src, dst)
   - TCP header (src port, dst port, seq, flags)

## Exercises

1. **Conceptual.** In `notes.md`, compute by hand:
   - You have a 100 Mbps link, RTT 20 ms, MSS 1460. What's the BDP in bytes? What window size do you need to fully utilize the link?
   - Same link, but RTT 200 ms (intercontinental). What changes?
   - Solve 2-3 of the book's end-of-chapter problems on delay/queuing.
2. **Practical.** Measure your own link:
   - `ping -c 50 1.1.1.1` to get RTT distribution. Note p50, p99, jitter.
   - `iperf3 -c iperf.he.net` (or any public iperf3 server) for throughput.
   - Compute observed BDP and compare to your ISP's advertised speed. Save as `exercises/02-bdp.md`.
3. **Stretch.** Capture an HTTPS page load to a real site (e.g. `curl -v https://example.com`). In Wireshark, identify the TLS Client Hello and look up the SNI extension. (We'll go deep in W12; this is just a teaser.)

## Self-check

- Sketch the 5-layer stack and place HTTP, TCP, IP, Ethernet, fiber on it.
- Without notes, write the formula for transmission delay and explain why it shrinks as link capacity grows.
- What does Wireshark's "Follow TCP Stream" do, and when would you use "Follow HTTP Stream" instead?

## Common gotchas

- A `.pcap` captured *inside* a container's netns won't see traffic the container forwards through the host bridge unless you capture on the right interface. Capture inside the netns (`network_mode: service:foo` sidecar) when in doubt.
- On macOS, Wireshark needs `ChmodBPF`. The installer prompts for it.
