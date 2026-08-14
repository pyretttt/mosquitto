# Integrated troubleshooting

The tools become useful when you can choose the next observation from the
last piece of evidence. A fixed checklist is a starting point, not a
substitute for a hypothesis.

## Where you are

You have one tool per layer of the path model. This module is the join:
given a symptom, which question is next, and which command answers it.
Then the [incidents](challenges/README.md) hide a single fault and ask you
to apply the join without a module title to hint.

## What you need, and what you do not

Need: every previous module's retrieval test, at least once with notes
allowed.

Do not need: production access, Kubernetes, or cloud consoles. The last
section translates those into the same questions so they do not feel like
a different subject.

## Workflow

### 1. Define the flow

Write:

```text
client IP:port → server IP:port, protocol, expected path
```

Record where the command runs. Namespace, container, pod, and host
observations are not interchangeable.

### 2. Reproduce narrowly

Use one request with a timeout. Record the exact error and time:

```bash
ip netns exec clp-client curl -v --connect-timeout 2 \
  http://10.20.2.10:8080/
```

If you cannot reproduce, you cannot locate a boundary. Change one variable
until it fails on demand.

### 3. Establish endpoint facts

At the client:

```bash
ip -n clp-client -br addr
ip -n clp-client route get 10.20.2.10
ip -n clp-client neigh
```

At the server:

```bash
ip -n clp-server -br addr
ip -n clp-server route get 10.20.1.10
ip netns exec clp-server ss -lntp sport = :8080
```

If `route get` already fails, skip tcpdump on that host's NIC. The kernel
has told you it will not send.

### 4. Locate the boundary

Capture only the target flow at successive interfaces. Find the first
expected packet that fails to appear. For TCP, separately track SYN and
SYN-ACK.

### 5. Inspect policy and state

On the router:

```bash
ip netns exec clp-router sysctl net.ipv4.ip_forward
ip netns exec clp-router nft -a list ruleset
ip netns exec clp-router iptables -nvL
ip netns exec clp-router iptables -t nat -nvL
ip netns exec clp-router conntrack -L -p tcp
```

Generate exactly one request and compare counters. First identify whether
the active rules were authored through nftables or the iptables
compatibility frontend.

### 6. Measure performance only when reachability exists

Use `mtr` for path latency/loss and `iperf3` for controlled capacity. Use
`ss -ti` and interface counters to explain the measurement.

### 7. Make and verify one fix

The verification should be the same test that failed, plus an observation
that proves why it now works. Remove diagnostic rules and captures when
done.

## Symptom-driven branches

### No route before any packet leaves

Use `ip route get`, addresses/prefixes, link state, and neighbor state. A
packet capture may be empty because the kernel correctly refused to send.

### SYN repeats and no SYN-ACK

Capture at client egress, router ingress/egress, and server ingress. If SYN
reaches the server, use `ss` for the listener and server-side route. If
SYN-ACK leaves but does not return, inspect reverse route, firewall state,
and NAT.

### Immediate RST

The network delivered a response. Inspect listener address/port and
deliberate firewall reject rules. `DROP` normally times out;
`REJECT --reject-with tcp-reset` fails immediately.

### Handshake works, large transfers stall

Inspect `ss -ti`, tcpdump sequence/retransmission behavior, MTU, MSS, ICMP
Packet Too Big / Fragmentation Needed, and offload artifacts.

### Possible gap: PMTUD black holes

If ICMP errors are filtered, endpoints keep sending packets the path cannot
carry. The parent `net/` week 7 lab is the dedicated reproduction. In this
track, check `ip link` MTUs and whether ICMP unreachable appears in a
capture. You can complete every included incident without a PMTUD fix.

### "The network is slow"

First separate connect time, server response time, and transfer time.
Compare forward/reverse iperf3, one/parallel streams, mtr destination
latency, `ss` retransmissions/RTT/cwnd, `tc qdisc`, and interface errors.
Do not infer a network fault from application duration alone.

### Intermittent new-connection failures

Inspect conntrack count/max, listener backlog, `SYN-RECV`, file descriptor
limits, packet drops, and rule counters. Existing connections working does
not prove new state can be created.

## Final practical

Solve every item in [challenges](challenges/README.md) without reading
scenario scripts or solutions. For each incident submit a note containing:

```text
Flow:
Expected:
Observed symptom:
First failed boundary:
Commands and relevant output:
Root cause:
Smallest fix:
Verification:
Cleanup:
```

A valid root cause names the incorrect state and location. "Routing,"
"firewall," and "network issue" are categories, not root causes.

Good: "server namespace has no default route, so SYN-ACK never leaves
`s-eth0`."
Bad: "routing is broken."

## Production additions

The lab builds the core method. Real incidents also require:

- change history and ownership;
- VRFs/policy routing and multiple network namespaces;
- cloud security groups, NACLs, load balancers, and managed NAT;
- container/Kubernetes network namespaces and CNI policy;
- DNS, TLS, proxies, and application telemetry;
- safe capture handling and authorization;
- avoiding blanket flushes on remote or shared systems.

Those are extra **objects**, not extra **questions**. A security group is a
stateful filter. A NACL is a stateless filter. A kube-proxy ClusterIP is
DNAT (or equivalent). An AWS NAT Gateway is SNAT on a managed hop. An
`Ingress` is an L7 reverse proxy in front of the TCP you already know how
to test.

The same discipline still applies: define the flow, identify the
observation point, locate the first failed boundary, and prove the fix.

## Retrieval test

Without notes, pick any failure signature from
[the debugging method](01-debugging-method.md) and write the **next**
command, not the whole runbook. Then start incident 1.
