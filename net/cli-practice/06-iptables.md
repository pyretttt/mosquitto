# `iptables`: legacy netfilter interface

A router that forwards every packet is a wire with extra latency. Real
networks **filter** (drop or reject some traffic) and **translate**
(rewrite addresses). Linux does both in **netfilter**: kernel hooks on the
packet path. `iptables` is the older user-space language for those hooks.
You will meet the newer language (`nft`) next. Learn the **hooks** once;
the syntax is the smaller part.

## Where you are

You can follow a packet from client to server with `ip` and `tcpdump`.
This module inserts a policy decision on that path. Until now the lab
router forwarded everything. That was a simplification.

## What you need, and what you do not

Need: host vs router from [foundations](00-foundations.md) (input versus
forward); capture points from [`tcpdump`](04-tcpdump.md).

Do not need: `nft` yet. Do not mix `iptables` and `nft` mutations during
one exercise. First identify the backend with `iptables --version`; output
containing `nf_tables` means the compatibility frontend is using nftables
underneath. The commands still matter: production, Docker, Kubernetes, and
interview loops are full of them.

## Why a firewall has to remember flows

A TCP reply uses a high ephemeral destination port. A naive rule "allow
destination port 8080" permits the SYN and **drops the SYN-ACK**, because
the reply's destination port is `45678`, not `8080`.

Two solutions:

1. Stateless: write matching reverse rules for every flow (painful).
2. Stateful: remember that a flow started, and allow its replies.

Linux conntrack implements (2). You will inspect it in module 8. For this
module, treat `-m conntrack --ctstate ESTABLISHED,RELATED` as "this packet
belongs to a conversation we already allowed."

`DROP` is silent (the sender times out). `REJECT` sends an error (TCP RST
or ICMP). The debugging method's "refused versus timeout" split is often
this choice.

## Packet path and hook choice

```text
incoming packet
      |
  PREROUTING          ← DNAT lives here (change destination before routing)
      |
 routing decision
   /          \
local       transit
INPUT       FORWARD   ← filter a host vs filter a router
  |            |
process     POSTROUTING → out
  |
OUTPUT → POSTROUTING → out
              ↑
              SNAT lives here (change source after routing)
```

Choose the chain from the packet's relationship to **this** host:

- addressed to a local process: `INPUT`;
- created by a local process: `OUTPUT`;
- routed through this host: `FORWARD`;
- destination translation before routing: `nat/PREROUTING`;
- source translation after routing: `nat/POSTROUTING`.

A common error is adding an `INPUT` rule on a router for transit traffic.
Client → server HTTP through `clp-router` never hits `clp-router`'s
`INPUT`. It hits `FORWARD`. Counters on `INPUT` will sit at zero and
teach you nothing unless you notice that.

Tables you will actually use:

- `filter` (default): accept / drop / reject;
- `nat`: SNAT / DNAT / MASQUERADE.

There are others (`mangle`, `raw`). Skip them until a counter or a tutorial
forces you there.

## Command anatomy

```text
iptables [-t TABLE] OPERATION CHAIN [MATCHES] -j TARGET
```

If `-t` is omitted, the table is `filter`.

Operations:

```text
-L list    -S print rule syntax    -A append    -I insert
-D delete  -F flush                -P policy    -C check
```

Frequent matches:

```text
-i IN_IFACE      -o OUT_IFACE      -s SOURCE       -d DESTINATION
-p tcp|udp|icmp  --sport PORT      --dport PORT
-m conntrack --ctstate NEW,ESTABLISHED,RELATED
```

Targets include `ACCEPT`, `DROP`, `REJECT`, `LOG`, `SNAT`, `DNAT`, and
`MASQUERADE`.

## Observe before changing

```bash
iptables -nvL
iptables -nvL FORWARD --line-numbers
iptables -t nat -nvL
iptables -S
iptables -t nat -S
iptables-save
```

`-n` avoids resolution, `-v` shows interfaces and counters, and `-x` shows
exact counters. Counters are evidence: generate one controlled request and
see which rule increments.

Rules are evaluated in order. The first terminating verdict wins. An
`ACCEPT` below an earlier broad `DROP` is unreachable. That is not a
syntax error; it is a logic error that looks like "I added the allow and
nothing happened."

## Stateful forwarding pattern

In `clp-router`, a minimal directional policy is:

```bash
ip netns exec clp-router iptables -P FORWARD DROP
ip netns exec clp-router iptables -A FORWARD \
  -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
ip netns exec clp-router iptables -A FORWARD \
  -i r-left -o r-right -p tcp --dport 8080 \
  -m conntrack --ctstate NEW -j ACCEPT
```

Read it as English:

1. If nothing matches, drop.
2. If this packet belongs to an allowed flow, accept (covers replies).
3. If this is a new TCP/8080 from left to right, accept and start a flow.

The first packet is `NEW` from left to right. Return packets are
`ESTABLISHED` and accepted regardless of their ephemeral destination port.
`RELATED` covers a new flow associated with an existing one, such as some
ICMP errors.

Put established/related **before** new accepts. Putting it after a drop
means replies never match.

## NAT anatomy

NAT exists because private addresses (`10.0.0.0/8` and friends) are not
supposed to appear on the public Internet, and because many hosts share few
public IPs. The kernel rewrites a header and **remembers** the mapping in
conntrack so replies can be rewritten back.

```bash
iptables -t nat -A POSTROUTING -o r-outside -j MASQUERADE
iptables -t nat -A POSTROUTING -o r-outside \
  -j SNAT --to-source 198.51.100.254
iptables -t nat -A PREROUTING -i r-outside -p tcp --dport 8080 \
  -j DNAT --to-destination 10.30.1.10:8080
```

Use `SNAT` for a known stable source address. `MASQUERADE` dynamically uses
the outgoing interface address and is useful for changing addresses. NAT
rule counters primarily reflect the first packet that creates a tracked
flow; NAT is then applied from conntrack state. That is why "the NAT rule
counter did not increment for packet 50" is normal.

You will implement this for real in module 8 and incident 4. Here, learn
**where** SNAT and DNAT attach so the later syntax is not a surprise.

## Common misconceptions

- **"I allowed port 8080 on INPUT."** Transit traffic uses FORWARD.
- **"I appended an ALLOW at the bottom."** An earlier DROP already won.
- **"Ping works, so the firewall is open."** ICMP is a different match
  from TCP/8080.
- **"Flushing iptables is a safe debug step."** On a real host it can drop
  your SSH session and Docker bridges. In this lab, flush only inside
  `clp-router`.

### Possible gap: Docker, Kubernetes, and nft backends

Docker and kube-proxy inject iptables (or nft) rules in the **host**
namespace. This lab never asks you to debug those. The same hook diagram
applies; the surprise is extra chains (`DOCKER`, `KUBE-SERVICES`) in the
middle of the order. List with `-nvL` and follow counters; do not start by
flushing.

## Muscle-memory drill

```bash
./scripts/setup-routed.sh
./scripts/setup-services.sh
```

In `clp-router`:

1. List every table and record existing policies.
2. Set a drop policy for forwarded traffic.
3. Permit established/related traffic.
4. Permit new client-to-server TCP/8080 only.
5. Prove HTTP works and ping does not.
6. Add ICMP forwarding in the correct position and prove its counter
   changes.
7. Delete that rule by specification, add it again, then delete it by line
   number.
8. Print restorable rule syntax.

Run commands through:

```bash
ip netns exec clp-router iptables ...
```

Never flush the container's root namespace.

## Retrieval test

Without notes, write commands to:

- list numeric verbose forward rules with line numbers;
- permit return traffic for existing flows;
- permit new TCP/8080 only from `r-left` to `r-right`;
- explain why appending an allow after a drop may not work;
- show NAT counters;
- identify whether a failed flow traverses `INPUT` or `FORWARD`.

After this module, implement the missing NAT incident once with
`iptables`, reset it, and later implement it with `nft`.

Next: [`nft`](07-nftables.md).
