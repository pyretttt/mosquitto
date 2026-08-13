# `mtr` and `iperf3`: path and performance

These tools answer different questions:

- `mtr`: how do latency and apparent loss change by hop?
- `iperf3`: what throughput, retransmission, UDP loss, and jitter can two
  endpoints produce?

Neither identifies application latency by itself. Measure DNS, TLS, server
work, and payload transfer separately when debugging an application.

## `mtr` mental model

`mtr` repeatedly sends probes with increasing TTL. Each router where TTL
expires may return ICMP Time Exceeded. The destination returns a final
protocol-specific response.

Report columns commonly include:

- `Loss%`: missing probe replies at this hop;
- `Snt`: probes sent;
- `Last`, `Avg`, `Best`, `Wrst`: latency samples;
- `StDev`: variability.

Intermediate routers often rate-limit or ignore probes. Loss at one hop that
does **not** continue at later hops is not evidence of forwarding loss. Real
path loss normally appears at a hop and persists to the destination.

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

Probe protocol matters. ICMP may work while TCP/8080 is filtered. Test the
protocol closest to the symptom, but remember that an `mtr` TCP probe is not a
full application request.

## `iperf3` mental model

An `iperf3` client sends to an `iperf3` server, TCP/5201 by default. A test
measures the endpoints, their kernels, and the path together.

TCP output reports interval throughput and retransmissions. UDP output reports
the offered rate, received rate, jitter, and datagram loss. UDP `-b` is a
requested sending rate, not discovered capacity.

Key distinctions:

- `-R` reverses data direction while the same endpoint remains the client.
- `-P N` uses parallel TCP streams; it can hide a single-flow window or
  congestion limitation.
- `-u -b RATE` tests UDP at a chosen offered load.
- `-O N` omits warm-up seconds from the final summary.
- `--json` creates machine-readable evidence.

## `iperf3` anatomy

```bash
iperf3 -s
iperf3 -c 10.20.2.10 -t 10 -i 1
iperf3 -c 10.20.2.10 -R -t 10
iperf3 -c 10.20.2.10 -P 4 -t 10
iperf3 -c 10.20.2.10 -u -b 20M -t 10
iperf3 -c 10.20.2.10 -t 10 --json
```

One run is not a baseline. Record direction, protocol, stream count, duration,
packet size where relevant, endpoint CPU, and path conditions.

## Bandwidth-delay product

To fill a path, roughly this much unacknowledged data must be in flight:

```text
BDP bytes = bandwidth bits/second × RTT seconds ÷ 8
```

A high-bandwidth, high-latency path needs a large congestion and receive
window. Poor single-stream throughput can improve with parallel streams even
when the path itself is not congested.

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

For each test, predict what changes and record the result. While TCP runs, use
`ss -tin` in another shell to correlate RTT, congestion window, and
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

Use `mtr`, `iperf3`, `ss`, `ip -s link`, and `tc qdisc show`; do not begin with
the scenario script's source.

## Retrieval test

Without notes:

- produce a numeric 30-cycle path report using TCP/8080;
- test forward and reverse TCP throughput;
- compare one and four parallel streams;
- offer 25 Mbit/s of UDP and interpret loss versus jitter;
- explain why 80% loss at an intermediate hop and 0% at the destination is not
  an 80%-loss path.
