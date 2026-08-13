# `nft`: modern netfilter interface

`nftables` uses the same kernel packet hooks as `iptables`, with a more regular
data model:

```text
family -> table -> base chain/hook -> ordered rules
```

Unlike iptables table names, an nftables table is only a container. A chain
sees packets only if it is a **base chain** attached to a hook.

## Families

- `ip`: IPv4 only;
- `ip6`: IPv6 only;
- `inet`: IPv4 and IPv6;
- `bridge`: Ethernet frames crossing a Linux bridge;
- `arp` and `netdev`: specialized paths.

Prefer `inet` for dual-stack filtering. NAT expressions still select IP
version where the translated address requires it.

## Command anatomy

```text
nft VERB OBJECT FAMILY TABLE [CHAIN] [RULE]
```

Observation:

```bash
nft list ruleset
nft list tables
nft list table inet filter
nft list chain inet filter forward
nft -a list chain inet filter forward
nft monitor ruleset
```

`-a` prints rule handles. Handles are stable identifiers used for deletion:

```bash
nft delete rule inet filter forward handle 7
```

## Base-chain anatomy

```bash
nft 'add table inet filter'
nft 'add chain inet filter forward {
  type filter hook forward priority filter; policy drop;
}'
```

Shell quoting matters because braces and semicolons belong to nft syntax.

Each base chain declares:

- `type`: filter, nat, or route;
- `hook`: prerouting, input, forward, output, or postrouting as permitted;
- `priority`: order relative to other chains at the same hook;
- `policy`: fallback accept or drop.

Regular chains have no hook and are reached with `jump` or `goto`.

## Rule anatomy

Rules read as matches followed by statements and a verdict:

```bash
nft add rule inet filter forward \
  iifname "r-left" oifname "r-right" \
  ip daddr 10.20.2.10 tcp dport 8080 \
  ct state new counter accept

nft add rule inet filter forward \
  ct state established,related counter accept
```

Useful selectors:

```text
meta l4proto tcp
iifname "r-left" / oifname "r-right"
ip saddr 10.20.1.0/24 / ip6 saddr fd00::/64
tcp dport { 80, 443 }
ct state { established, related }
counter
```

Sets avoid repeated rules:

```bash
nft 'add set inet filter web_ports { type inet_service; }'
nft 'add element inet filter web_ports { 80, 443, 8080 }'
nft 'add rule inet filter forward tcp dport @web_ports counter accept'
```

## Load files atomically

Prefer a ruleset file for repeatable configurations:

```nft
table inet practice_filter {
    chain forward {
        type filter hook forward priority filter; policy drop;
        ct state established,related counter accept
        iifname "r-left" oifname "r-right" tcp dport 8080 counter accept
    }
}
```

Check and load:

```bash
nft -c -f /tmp/practice.nft
nft -f /tmp/practice.nft
```

`-c` validates without applying. Keep lab table names distinctive so you can
delete one table rather than flush an entire ruleset.

## NAT anatomy

```nft
table ip practice_nat {
    chain prerouting {
        type nat hook prerouting priority dstnat;
        iifname "r-outside" tcp dport 8080 dnat to 10.30.1.10:8080
    }
    chain postrouting {
        type nat hook postrouting priority srcnat;
        oifname "r-outside" snat to 198.51.100.254
    }
}
```

`masquerade` replaces `snat to ...` when the interface address is dynamic.

## Muscle-memory drill

```bash
./scripts/setup-routed.sh
./scripts/setup-services.sh
```

Create a file under `/tmp` and load a table named `inet practice_filter` that:

1. defaults forwarded traffic to drop;
2. counts and accepts established/related traffic;
3. counts and accepts new client-to-server TCP ports in a set containing 8080
   and 5201;
4. counts but drops other forwarded traffic.

Then:

- validate the file before loading;
- generate HTTP, iperf3, and ping traffic;
- explain every counter;
- add a temporary ICMP accept rule;
- display its handle and delete it by handle;
- delete only `practice_filter`.

Run all nft commands through:

```bash
ip netns exec clp-router nft ...
```

## Retrieval test

Without notes:

- explain why a chain with rules but no hook sees no packets;
- create an `inet` filter table and forward base chain;
- allow established/related flows before new flows;
- use a set for three TCP ports;
- show rule handles and delete one rule;
- validate and atomically load a file;
- explain where `dnat` and `snat` attach.
