# Performance / measurement cheatsheet (`iperf3`, `mtr`, `ethtool`, `nstat`)

When someone says "the network is slow", you measure.

## iperf3 (throughput)

```bash
iperf3 -s                                   # server (default port 5201)
iperf3 -s -p 9999

iperf3 -c <server> -t 30                    # 30s TCP
iperf3 -c <server> -t 30 -i 1               # 1s reports
iperf3 -c <server> -P 4                     # 4 parallel streams
iperf3 -c <server> -R                       # reverse (server -> client)
iperf3 -c <server> -u -b 100M               # UDP at 100 Mbps
iperf3 -c <server> -u -b 100M -l 1200       # UDP, 1200B datagrams
iperf3 -c <server> -O 5                     # omit first 5s (avoid slow-start in numbers)

iperf3 -c <server> -t 30 --json > result.json
```

Public iperf3 servers: https://iperf.fr/iperf-servers.php

## mtr (continuous traceroute + ping)

```bash
mtr 1.1.1.1
mtr -bw 1.1.1.1                             # show both names and IPs
mtr -n 1.1.1.1                              # numeric only
mtr --report-wide --report-cycles 30 1.1.1.1
mtr -T -P 443 example.com                   # TCP probes (works through firewalls)
mtr -u 8.8.8.8                              # UDP
```

"Loss%" climbing only at one hop and then dropping is *usually* ICMP rate limiting at that router, not real loss.

## ping

```bash
ping -c 50 -i 0.2 1.1.1.1                   # 50 packets, 200ms apart
ping -M do -s 1500 1.1.1.1                  # DF=1, no fragmentation
ping -i 0.01 -c 1000 1.1.1.1                # flood-ish (root)
ping -W 1 -c 1 1.1.1.1                      # 1s timeout, single probe (scriptable)
```

## tracepath / traceroute

```bash
traceroute -n 1.1.1.1                       # UDP probes by default on Linux
traceroute -I 1.1.1.1                       # ICMP echo probes
traceroute -T -p 443 1.1.1.1                # TCP SYN probes
tracepath 1.1.1.1                           # also discovers PMTU
```

## ethtool

```bash
ethtool eth0                                # speed/duplex/link
ethtool -i eth0                             # driver / firmware
ethtool -k eth0                             # offload features
ethtool -K eth0 tso off gso off gro off     # disable offloads (debugging)
ethtool -S eth0                             # NIC stats (drops, errors per ring)
ethtool -g eth0                             # ring sizes
ethtool -G eth0 rx 4096 tx 4096             # bump ring sizes
ethtool -c eth0                             # interrupt coalescing
ethtool -l eth0                             # combined queue counts
ethtool -L eth0 combined 8                  # set queue count
```

## nstat / netstat-style counters

```bash
nstat -a                                    # cumulative (since reboot) for everything
nstat                                       # delta since last call
nstat -t 5                                  # every 5s

cat /proc/net/snmp                          # protocol stats
cat /proc/net/netstat                       # extended TCP stats
cat /proc/net/dev                           # per-iface counters
ip -s -s link show eth0                     # detailed iface stats
```

What to look for:

- `Tcp.RetransSegs` / `Tcp.OutSegs` ratio - retransmit %.
- `TcpExt.TCPLostRetransmit` - retransmits that themselves were lost (bad).
- `TcpExt.ListenDrops` - SYN backlog overflow (raise `somaxconn` and `tcp_max_syn_backlog`).
- `IpExt.InNoRoutes` - no route to host.
- `Udp.RcvbufErrors` / `Udp.SndbufErrors` - socket buffer too small.

## ss -ti for live cwnd / retrans (already in `sockets.md`)

```bash
watch -n 0.5 'ss -ti dst :443 | head -50'
```

## bcc / bpftrace (modern observability)

```bash
sudo tcptop                                  # top-like for TCP byte rates
sudo tcplife                                 # connection lifetimes
sudo tcpretrans                              # live retransmits
sudo tcpconnect                              # who connected where
sudo tcpaccept                               # who got connected
```

If bcc isn't available, the same tools exist in `bpftrace` form. Also: `ss -tin` is a poor man's `tcptop`.

## Bandwidth-delay product math

```
BDP_bytes = bandwidth_bps * RTT_seconds / 8

100 Mbps * 20ms = 100e6 * 0.02 / 8 = 250,000 bytes  -> ~ 250KB window needed
100 Mbps * 200ms = 2,500,000 bytes                  -> ~2.5MB window needed
1 Gbps  * 100ms = 12,500,000 bytes                  -> ~12MB window needed
```

If `tcp_wmem`/`tcp_rmem` cap is below the BDP, you can't fill the link.

## Common debugging recipes

- "Throughput is bad." -> `iperf3` baseline; `ss -ti` to see cwnd/retrans/rtt; `nstat | grep -i retrans`; `ethtool -S` for NIC drops; check MTU/MSS.
- "Latency is bad." -> `mtr` to find which hop; `ping` distribution; check `qdisc` on egress (`tc qdisc show dev eth0`); check CPU softirq pressure (`mpstat -P ALL 1`, `irqbalance`).
- "Random packet loss." -> `nstat | grep -i drop`, `ip -s -s link`, ring buffer (`ethtool -S | grep -i drop`), conntrack overflow (`dmesg | grep conntrack`).
