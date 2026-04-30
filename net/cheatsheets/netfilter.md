# netfilter / iptables / nftables / conntrack cheatsheet

The kernel framework underneath every Linux firewall, every Docker network, every k8s service mesh. `nftables` is the modern interface; `iptables` is the legacy one but still common in production.

## The hooks (every packet flows through these)

```
                     +-----------+
                     | local proc|
                     +-----^-----+
                           |
                       INPUT
                           ^
   PREROUTING --> [routing decision] --> FORWARD --> POSTROUTING --> NIC out
       ^                                                ^
   NIC in                                          OUTPUT  <- local proc
```

Tables (in legacy iptables; nftables collapses these):

| Table   | Hooks                                  | What           |
|---------|----------------------------------------|----------------|
| `raw`   | prerouting, output                      | conntrack bypass, tracing |
| `mangle`| all                                    | header rewrites (TOS, MSS) |
| `nat`   | prerouting, output, postrouting        | DNAT, SNAT     |
| `filter`| input, forward, output                 | accept/drop    |

## nftables (modern)

Layout: `tables` -> `chains` -> `rules`. A chain attaches to one hook with a priority.

```bash
nft list ruleset
nft list table inet filter
nft list chain inet filter forward
nft -a list ruleset                          # show handles for delete
nft delete rule inet filter forward handle 7

nft flush ruleset                            # nuke everything
```

Idiomatic stateful firewall:

```nft
table inet filter {
    chain input {
        type filter hook input priority 0; policy drop;
        ct state established,related accept
        iifname "lo" accept
        ip protocol icmp accept
        ip6 nexthdr ipv6-icmp accept
        tcp dport 22 accept
    }
    chain forward {
        type filter hook forward priority 0; policy drop;
        ct state established,related accept
        iifname "veth-rin" oifname "veth-rout" accept
    }
    chain output {
        type filter hook output priority 0; policy accept;
    }
}

table inet nat {
    chain postrouting {
        type nat hook postrouting priority srcnat;
        oifname "veth-rout" masquerade
    }
}
```

Apply: `sudo nft -f rules.nft`.

Sets and maps for fast rule lookups:

```nft
set blocked4 { type ipv4_addr; flags interval; elements = { 10.0.0.0/8, 192.168.0.0/16 } }

ip saddr @blocked4 drop
```

## iptables (legacy, very common)

```bash
iptables -L -n -v                            # list filter table
iptables -t nat -L -n -v
iptables -S                                  # rules in iptables-save format
iptables-save > rules.txt
iptables-restore < rules.txt

iptables-translate -A INPUT -p tcp --dport 80 -j ACCEPT   # convert to nft

iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables -A INPUT -i lo -j ACCEPT
iptables -A INPUT -p tcp --dport 22 -j ACCEPT
iptables -A INPUT -j DROP

iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
iptables -t nat -A PREROUTING  -i eth0 -p tcp --dport 80 -j DNAT --to 10.0.0.5:8080

iptables -t mangle -A FORWARD -p tcp --tcp-flags SYN,RST SYN -j TCPMSS --clamp-mss-to-pmtu
```

## conntrack

```bash
conntrack -L                                 # current table
conntrack -L -p tcp --dport 80
conntrack -E                                 # event stream (great for debugging NAT)
conntrack -F                                 # flush (don't do this on production!)
conntrack -D -p tcp --dport 80 -d 10.0.0.5  # delete a specific entry

sysctl net.netfilter.nf_conntrack_count
sysctl net.netfilter.nf_conntrack_max
sysctl net.netfilter.nf_conntrack_tcp_timeout_established
sysctl net.netfilter.nf_conntrack_tcp_timeout_time_wait
```

## Common patterns

```bash
sudo nft list ruleset | grep -B2 'drop'
sudo nft list ruleset | grep -A1 counter

sudo iptables -t nat -L POSTROUTING -nv
sudo iptables -t nat -nvL --line-numbers
```

## When debugging "why is this packet dropped?"

1. Add a logging rule before each `drop` in the suspected chain:
   `log prefix "[drop-fwd] " level info` (nftables) or `-j LOG --log-prefix "[drop-fwd] "` (iptables).
2. Watch `journalctl -kf` (kernel log).
3. Check conntrack: `conntrack -L | grep <peer-ip>`.
4. Sanity-check forwarding: `sysctl net.ipv4.ip_forward`.
5. Use `nft monitor` for live rule changes.

## Cgroup / cgroupv2 / nftables interaction

Modern Docker/Cilium write rules per-cgroup. If you `nft list ruleset` and see `cgroup` matches, you're looking at workload-scoped rules - resist the urge to flush.
