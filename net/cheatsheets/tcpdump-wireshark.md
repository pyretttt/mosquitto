# tcpdump / Wireshark cheatsheet

`tcpdump` is universal. Wireshark is what you use when you have a `.pcap` and want a UI.
Both speak the same filter language for capture filters (BPF). Display filters in Wireshark are different (Wireshark-specific syntax).

## tcpdump basics

```bash
tcpdump -D                              # list interfaces
tcpdump -i eth0                          # capture on eth0
tcpdump -i any                           # all (Linux)
tcpdump -ni eth0                         # don't resolve names (faster, doesn't pollute DNS)
tcpdump -nei eth0                        # also show MAC addresses (-e)
tcpdump -nvvi eth0                       # verbose
tcpdump -A                               # ASCII dump of packet (HTTP debug)
tcpdump -X                               # hex + ASCII
tcpdump -s 0                             # capture full packet (default in modern versions)

tcpdump -ni eth0 -w /tmp/cap.pcap        # write pcap
tcpdump -ni eth0 -w /tmp/cap-%H%M.pcap -G 60 -W 5    # rotate every 60s, keep 5
tcpdump -r /tmp/cap.pcap                 # read pcap

tcpdump -c 100                           # stop after 100 packets
```

## BPF capture filters (also valid as tcpdump filter)

```bash
host 10.0.0.1
src host 10.0.0.1
dst host 10.0.0.1
net 10.0.0.0/24
port 80
src port 80
portrange 1024-65535

tcp                                      # protocol
udp
icmp
icmp6
arp
proto gre

tcp port 80 and host 10.0.0.1
tcp and not port 22
tcp and (port 80 or port 443)
'tcp[tcpflags] & (tcp-syn) != 0'         # only SYN
'tcp[tcpflags] & (tcp-syn|tcp-fin) != 0' # SYN or FIN

vlan                                     # 802.1Q
ether host aa:bb:cc:dd:ee:ff             # by MAC
ether broadcast
ether multicast

ip6                                      # IPv6
ip proto 47                              # GRE
```

## Wireshark display filters (different syntax!)

Wireshark display filters use protocol field names, not BPF:

```
ip.addr == 10.0.0.1
ip.src == 10.0.0.1
ip.dst == 10.0.0.1
tcp.port == 80
tcp.flags.syn == 1
tcp.analysis.retransmission
tcp.analysis.duplicate_ack
tls.handshake.type == 1                  # ClientHello
tls.handshake.extensions_server_name == "example.com"
http.request
http.response.code == 500
dns.qry.name contains "example"
icmp.type == 3
```

Useful display-filter shortcuts:

```
tcp.analysis.flags                       # any TCP "interesting" flag
tcp.stream eq 4                          # follow stream #4
http.request.method == "POST"
```

## Common one-liners

```bash
sudo tcpdump -ni any -w /tmp/cap.pcap port 53                    # DNS
sudo tcpdump -ni any -w /tmp/cap.pcap port 80                    # HTTP
sudo tcpdump -ni any -w /tmp/cap.pcap port 443                   # HTTPS (just the bytes)
sudo tcpdump -ni any -w /tmp/cap.pcap icmp                       # pings
sudo tcpdump -ni any -w /tmp/cap.pcap arp                        # ARP storms
sudo tcpdump -ni any -e proto ospf -w /tmp/ospf.pcap             # OSPF
sudo tcpdump -ni any -e tcp port 179 -w /tmp/bgp.pcap            # BGP
sudo tcpdump -ni any -w /tmp/cap.pcap 'tcp[tcpflags] & (tcp-syn|tcp-rst|tcp-fin) != 0'

sudo tcpdump -ni any -w - port 53 | tshark -r - -Y dns -V        # decode in flight
```

## Wireshark "Statistics" menu favorites

- IO Graphs - throughput/retransmits over time.
- TCP Stream Graph -> Round Trip Time.
- Conversations - top talkers.
- Expert Information - retransmissions, dup ACKs, zero windows, all bundled.
- Analyze -> Decode As - tell Wireshark "this port is gRPC" or "this is HTTP/2".

## Decrypt TLS in Wireshark

1. `export SSLKEYLOGFILE=/tmp/keylog.txt` before running curl/firefox/...
2. Wireshark -> Edit -> Preferences -> Protocols -> TLS -> "(Pre)-Master-Secret log filename" -> point at `/tmp/keylog.txt`.
3. TLS records become readable.

## tshark (CLI Wireshark)

```bash
tshark -r cap.pcap -Y 'tls.handshake.type == 1' -T fields -e tls.handshake.extensions_server_name
tshark -r cap.pcap -q -z conv,tcp                           # TCP conversations
tshark -r cap.pcap -q -z io,stat,1                          # 1-second IO histogram
tshark -r cap.pcap -Y http -T fields -e http.host -e http.request.uri
```
