# Sockets / `ss` cheatsheet

`ss` is the modern `netstat`. Use it.

## ss basics

```bash
ss                                       # established TCP
ss -t                                    # TCP only
ss -u                                    # UDP only
ss -tn                                   # numeric TCP
ss -tnl                                  # listening TCP
ss -tnlp                                 # + processes (needs root)
ss -tnu                                  # TCP and UDP
ss -tnp4                                 # IPv4 only
ss -tnp6                                 # IPv6 only

ss -s                                    # summary (counts per state)
ss -tan                                  # all TCP including TIME-WAIT
ss -tao                                  # show timer info (RTO, keepalive)
```

## Filters

```bash
ss -tn dst :443                          # connections to port 443 (anywhere)
ss -tn dst 10.0.0.5
ss -tn src :9000                         # listener with source port 9000
ss -tn '( dst :80 or dst :443 )'
ss -tn state established
ss -tn state time-wait
ss -tn '( state syn-sent or state syn-recv )'
```

## TCP info (the gold mine)

```bash
ss -ti                                   # detailed TCP per-socket
ss -tin dst :443
```

`-i` adds: rtt, rto, cwnd, snd_wnd, retrans, lost, send_rate, rcv_rate. This is what you screenshot to debug "why is my throughput bad".

```
ESTAB 0 0 10.0.0.1:54321 1.2.3.4:443
    ts sack cubic wscale:7,7 rto:204 rtt:3.012/2.5 ato:40 mss:1448
    pmtu:1500 rcvmss:1448 advmss:1448 cwnd:32 ssthresh:25 bytes_sent:154720
    bytes_acked:154720 bytes_received:8312 segs_out:113 segs_in:84
    data_segs_out:107 send 12.3Mbps lastsnd:204 lastrcv:200 lastack:200
    pacing_rate 24.6Mbps delivery_rate 8.0Mbps app_limited busy:212ms
    rcv_rtt:3 rcv_space:14600 minrtt:2.7
```

What to look at:

- `cwnd` close to `ssthresh` and stable: probably running at line rate or app-limited.
- `retrans` > 0: loss happened.
- `rtt:` and `minrtt:`: gap > 2x means queuing / bufferbloat.
- `snd_wbytes` close to `tcp_wmem`: sender buffer-bound.

## Other useful ss

```bash
ss -K dst 1.2.3.4 dport = 443            # kill matching sockets (root, careful)
ss -e                                    # show inode + uid (correlate to process)
```

## lsof (alternate path to "what is using this port")

```bash
sudo lsof -nP -i:80
sudo lsof -nP -iTCP -sTCP:LISTEN
sudo lsof -nP -iTCP@10.0.0.5:443
```

## /proc

```bash
cat /proc/net/tcp                         # raw view (hex addresses)
cat /proc/net/sockstat                    # alloc / inuse stats
cat /proc/<pid>/net/tcp                   # per-netns
ls -l /proc/<pid>/ns/net                  # which netns is this process in
```

## sysctls you should know

```bash
sysctl net.core.somaxconn                 # max accept queue
sysctl net.ipv4.tcp_max_syn_backlog       # SYN backlog
sysctl net.ipv4.tcp_syncookies            # 1 = enable cookies under SYN flood
sysctl net.ipv4.tcp_tw_reuse              # reuse TIME-WAIT for outbound
sysctl net.ipv4.tcp_fin_timeout
sysctl net.ipv4.tcp_keepalive_time
sysctl net.core.rmem_max
sysctl net.core.wmem_max
sysctl net.ipv4.tcp_rmem                  # min default max
sysctl net.ipv4.tcp_wmem
sysctl net.ipv4.tcp_congestion_control
sysctl net.ipv4.tcp_available_congestion_control
```

## Quick "what's going on" recipes

```bash
ss -s                                       # are we close to file/socket limits?
ss -tn state time-wait | wc -l              # too many TIME-WAITs?
ss -tn state syn-recv | wc -l               # SYN flood signature
ss -tnl | grep -E 'LISTEN' | sort -k4       # listeners
ss -tin dst :443 | grep -E 'rtt|cwnd|retrans'   # health of one direction
```

## When `netstat` is all you have

```bash
netstat -tnlp
netstat -an | grep ':80 '
netstat -i                                  # per-interface counters
```
