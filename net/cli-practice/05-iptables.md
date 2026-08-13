# `iptables`: legacy netfilter interface

`iptables` is a rule-management interface to Linux netfilter. Modern
distributions often implement it through an nftables backend, but its command
model remains common in servers, Docker, Kubernetes, runbooks, and interviews.

Do not mix `iptables` and `nft` mutations during one exercise. First identify
the backend with `iptables --version`; output containing `nf_tables` means the
compatibility frontend is using nftables underneath.

## Packet path and hook choice

```text
incoming packet
      |
  PREROUTING
      |
 routing decision
   /          \
local       transit
INPUT       FORWARD
  |            |
process     POSTROUTING -> out
  |
OUTPUT -> POSTROUTING -> out
```

Choose the chain from the packet's relationship to this host:

- addressed to a local process: `INPUT`;
- created by a local process: `OUTPUT`;
- routed through this host: `FORWARD`;
- destination translation before routing: `nat/PREROUTING`;
- source translation after routing: `nat/POSTROUTING`.

A common error is adding an `INPUT` rule on a router for transit traffic.

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
exact counters. Counters are evidence: generate one controlled request and see
which rule increments.

Rules are evaluated in order. The first terminating verdict wins. An `ACCEPT`
below an earlier broad `DROP` is unreachable.

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

The first packet is `NEW` from left to right. Return packets are
`ESTABLISHED` and accepted regardless of their ephemeral destination port.
`RELATED` covers a new flow associated with an existing one, such as some ICMP
errors.

## NAT anatomy

```bash
iptables -t nat -A POSTROUTING -o r-outside -j MASQUERADE
iptables -t nat -A POSTROUTING -o r-outside \
  -j SNAT --to-source 198.51.100.254
iptables -t nat -A PREROUTING -i r-outside -p tcp --dport 8080 \
  -j DNAT --to-destination 10.30.1.10:8080
```

Use `SNAT` for a known stable source address. `MASQUERADE` dynamically uses the
outgoing interface address and is useful for changing addresses. NAT rule
counters primarily reflect the first packet that creates a tracked flow; NAT
is then applied from conntrack state.

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
6. Add ICMP forwarding in the correct position and prove its counter changes.
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

After this module, implement the missing NAT incident once with `iptables`,
reset it, and later implement it with `nft`.
