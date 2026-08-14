# Incidents

These are problem statements, not walkthroughs. Do not read
`scripts/scenarios/` or [solutions](../solutions/README.md) before solving.

You need [foundations](../00-foundations.md) and the
[debugging method](../01-debugging-method.md) before any incident. Each
incident also lists the tool modules that are fair game. If you have not
read those yet, you are not stuck — you are early.

For every incident:

1. start it from the `cli-practice` directory inside the lab container;
2. reproduce the stated symptom;
3. define the expected flow;
4. locate the first failed boundary;
5. preserve command output that proves the cause;
6. make the smallest fix;
7. run the validator;
8. reset before the next incident.

If blocked, reread only the linked theory. After 20 minutes, use the
progressive hints at the bottom of this page. Do not jump directly to a
solution.

The incident note from [integrated troubleshooting](../09-troubleshooting.md)
is the deliverable, not the fix alone.

## Incident 1: address works, remote subnet does not

Start:

```bash
./scripts/scenarios/broken-route.sh
```

Report:

> The server process is healthy. The router's two directly connected
> addresses answer. A client request to `http://10.20.2.10:8080/` fails.

Objective: restore HTTP while preserving the given subnet addressing.

Allowed theory: [`ip`](../02-ip.md) and the
[debugging method](../01-debugging-method.md). The trap is treating one
successful on-link ping as proof of a bidirectional routed path.

Validate:

```bash
./checks/check-routed.sh
./checks/check-service.sh
```

## Incident 2: ping works, HTTP times out

Start:

```bash
./scripts/scenarios/blocked-service.sh
```

Report:

> Client-to-server ICMP succeeds. TCP/8080 times out. The HTTP server is
> listening.

Objective: restore HTTP without removing the firewall table or changing
unrelated forwarding behavior.

Allowed theory: [`ss`](../03-ss.md), [`tcpdump`](../04-tcpdump.md), and
[`nft`](../07-nftables.md). ICMP and TCP are different matches; a counter
on the wrong rule is still evidence.

Validate:

```bash
./checks/check-service.sh
```

## Incident 3: reachable but slow

Start:

```bash
./scripts/scenarios/slow-link.sh
```

Report:

> HTTP works, but users report slow and inconsistent transfers.

Objective: identify every configured impairment, its interface/direction,
and its effect. Remove only the impairment and show before/after
measurements.

Allowed theory: [`mtr` and `iperf3`](../05-mtr-iperf3.md),
[`ss`](../03-ss.md), and [`ip`](../02-ip.md). Reachability is already
proven; this is a measurement problem.

There is no binary validator for a good performance investigation. Your
result must include:

- numeric path report;
- forward and reverse single-stream TCP results;
- one UDP result at a stated offered rate;
- `ss -ti` evidence during a transfer;
- interface/qdisc evidence;
- equivalent before/after test parameters.

## Incident 4: private client needs Internet-style egress

Start:

```bash
./scripts/scenarios/nat-missing.sh
```

Report:

> The private client can reach its gateway. The public server is directly
> connected to the router but has no route for private address space. The
> public server must not gain such a route.

Objective: allow client HTTP egress and make the server observe the
router's external source address. First solve with `iptables`, reset, then
solve with `nft`.

Allowed theory: [`iptables`](../06-iptables.md),
[`nft`](../07-nftables.md), and
[`conntrack` and NAT](../08-conntrack-nat.md). Adding a route on the
server would make packets flow and still be the wrong design.

Validate:

```bash
./checks/check-nat.sh
```

Your evidence must include captures on both router sides and the conntrack
original/reply tuples.

## Capstone: unknown failure

Ask another person to launch any scenario without naming it, or choose one
randomly after at least a week. Begin only with:

```bash
ip netns list
```

You may use all tools, but you may not inspect process command lines
outside `ss -p`, scenario source, or shell history. Produce the incident
note template from [integrated troubleshooting](../09-troubleshooting.md).

## Progressive hints

Read one hint, then investigate for another five minutes.

### Incident 1

1. Compare routing decisions at both endpoints, not only the client.
2. A successful request needs a return path.
3. `ip route get PEER_ADDRESS` on the server should name a usable next hop.

### Incident 2

1. A successful ping proves only ICMP.
2. Capture TCP/8080 on both router interfaces and compare rule counters.
3. Inspect rule order and handles in the router's forward hook.

### Incident 3

1. Test both directions with identical duration.
2. Compare destination loss with intermediate-hop loss.
3. Linux traffic impairments are commonly attached to an egress qdisc.

### Incident 4

1. Draw the source address that the server currently receives.
2. The required translation changes source after the routing decision.
3. Inspect the router's postrouting NAT hook and create fresh connections
   when testing changed NAT rules.
