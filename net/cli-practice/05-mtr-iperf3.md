# `mtr` and `iperf3`: path and performance

Reachability and performance are different questions. `ping` or `curl` can
succeed while users still report "the network is slow." These two tools
split that complaint into **where delay or loss appears** and **how many
bits the path can carry**.

## Where you are

You can prove that a packet exists at a capture point and that a socket is
established. This module assumes the path **works**. Do not start here when
`curl` times out; go back to `ip`, `ss`, and `tcpdump`.

## What you need, and what you do not

Need: TTL from [foundations](00-foundations.md); `ss -ti` from
[`ss`](03-ss.md); the routed lab with services.

Do not need: traffic-control theory, queuing disciplines as a design skill,
or application profiling. You will *meet* `tc qdisc` as a way Linux
attaches delay, loss, and rate limits to an interface. You do not need to
author a production QoS policy.

## Two questions, two tools

- `mtr`: how do latency and apparent loss change by hop?
- `iperf3`: what throughput, retransmission, UDP loss, and jitter can two
  endpoints produce?

Neither identifies application latency by itself. Measure DNS, TLS, server
work, and payload transfer separately when debugging an application. This
lab has no DNS or TLS, so a slow `curl` of a large file is mostly path plus
CPU.

## `mtr` mental model

IP packets carry a **TTL**. Each router subtracts 1. When TTL hits 0, the
router should send ICMP Time Exceeded to the sender and drop the packet.

`mtr` (and traceroute) repeatedly sends probes with increasing TTL:

```text
probe TTL=1  → first router expires it → ICMP time exceeded
probe TTL=2  → second hop expires it
…
probe TTL=N  → destination answers (ICMP echo reply, TCP response, …)
```

In this lab there is only one router, so a healthy report looks like:

```text
hop 1  10.20.1.1     the router
hop 2  10.20.2.10    the server
```

Report columns commonly include:

- `Loss%`: missing probe replies at this hop;
- `Snt`: probes sent;
- `Last`, `Avg`, `Best`, `Wrst`: latency samples;
- `StDev`: variability.

**The important lie:** intermediate routers often rate-limit or ignore
probes. Loss at one hop that does **not** continue at later hops is not
evidence of forwarding loss. Real path loss normally appears at a hop and
**persists to the destination**.

Probe protocol matters. ICMP may work while TCP/8080 is filtered. Test the
protocol closest to the symptom, but remember that an `mtr` TCP probe is
not a full application request.

## `mtr` anatomy

```bash
mtr -n -r -c 20 10.20.2.10
```

Read this as numeric output, report mode, 20 cycles, destination.

```bash
mtr -n 10.20.2.10              # interactive numeric view
mtr -n -r -w -c 30 10.20.2.10  # wide report for an incident
mtr -n -T -P 8080 10.20.2.10   # TCP probes to application port
mtr -n -u 10.20.2.10           # UDP probes
```

Run it in the client namespace:

```bash
ip netns exec clp-client mtr -n -r -c 10 10.20.2.10
```

## `iperf3` mental model

An `iperf3` client sends to an `iperf3` server, TCP/5201 by default. The
lab already starts that server with `setup-services.sh`. A test measures
the endpoints, their kernels, and the path together — not "the wire" in
isolation.

TCP output reports interval throughput and retransmissions. UDP output
reports the offered rate, received rate, jitter, and datagram loss. UDP
`-b` is a **requested sending rate**, not discovered capacity. If you offer
10 Mbit/s on a 5 Mbit/s path, UDP will lose; that is the test working.

Key distinctions:

- `-R` reverses **data** direction while the same endpoint remains the
  client. Control still starts from the client. Use this to detect
  one-way impairments.
- `-P N` uses parallel TCP streams; it can hide a single-flow window or
  congestion limitation.
- `-u -b RATE` tests UDP at a chosen offered load.
- `-O N` omits warm-up seconds from the final summary.
- `--json` creates machine-readable evidence.

One run is not a baseline. Record direction, protocol, stream count,
duration, packet size where relevant, and path conditions. Changing three
of those and calling the result "faster" is not a measurement.

## Bandwidth-delay product

To fill a path, roughly this much unacknowledged data must be in flight:

```text
BDP bytes = bandwidth bits/second × RTT seconds ÷ 8
```

Example: 5 Mbit/s with 120 ms RTT:

```text
5_000_000 × 0.120 / 8 = 75_000 bytes ≈ 50 packets of 1500 bytes
```

If the congestion window stays smaller than that, a single TCP stream
cannot fill the path. Parallel streams (`-P 4`) may then look "better"
even though you have not repaired the path. That is a diagnostic clue, not
a fix.

## Where impairments live in Linux

Linux shapes traffic with a **qdisc** (queuing discipline) on an
interface. A qdisc acts on **egress**: packets **leaving** that NIC. Delay
or loss attached to `r-right` therefore hits router → server traffic, not
the reverse, unless a matching qdisc exists on the other path.

You inspect this with:

```bash
ip netns exec clp-router tc qdisc show
ip -n clp-router -s link show
```

You do not need `tc` fluency to finish the track. You do need to remember
that "the router is slow" is not a complete sentence until you name
interface and direction.

### Possible gap: application time versus network time

A slow webpage can be DNS + TLS + server compute + transfer. iperf3
eliminates the application. If iperf3 is fast and HTTP is slow, stop
blaming the path. This lab's HTTP server is local and trivial, so that
split rarely appears here. In production it appears constantly.

## Muscle-memory drill

```bash
./scripts/setup-routed.sh
./scripts/setup-services.sh
```

Run from the client namespace:

```bash
ip netns exec clp-client mtr -n -r -c 10 10.20.2.10
ip netns exec clp-client iperf3 -c 10.20.2.10 -t 5
ip netns exec clp-client iperf3 -c 10.20.2.10 -R -t 5
ip netns exec clp-client iperf3 -c 10.20.2.10 -P 4 -t 5
ip netns exec clp-client iperf3 -c 10.20.2.10 -u -b 10M -t 5
```

For each test, predict what changes and record the result. While TCP runs,
use `ss -tin` in another shell to correlate RTT, congestion window, and
retransmissions.

Then launch:

```bash
./scripts/scenarios/slow-link.sh
```

You are not told the impairment. Determine:

1. whether latency, loss, or capacity changed;
2. which direction is affected;
3. whether one TCP stream and four streams behave differently;
4. whether the destination experiences persistent loss;
5. which interface owns the impairment.

Use `mtr`, `iperf3`, `ss`, `ip -s link`, and `tc qdisc show`; do not begin
with the scenario script's source.

## Retrieval test

Without notes:

- produce a numeric 30-cycle path report using TCP/8080;
- test forward and reverse TCP throughput;
- compare one and four parallel streams;
- offer 25 Mbit/s of UDP and interpret loss versus jitter;
- explain why 80% loss at an intermediate hop and 0% at the destination is
  not an 80%-loss path.

Next: [`iptables`](06-iptables.md).
