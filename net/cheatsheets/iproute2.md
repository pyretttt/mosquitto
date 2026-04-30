# iproute2 cheatsheet (`ip`, `tc`, `bridge`)

`iproute2` is the modern replacement for `ifconfig`, `route`, `arp`, `vconfig`, etc.
If you find yourself reaching for `ifconfig`, you're probably on a Mac.

## Interfaces

```bash
ip link                                  # all interfaces
ip -s link                               # with stats
ip link show eth0
ip link set eth0 up
ip link set eth0 down
ip link set eth0 mtu 1400
ip link set eth0 address aa:bb:cc:dd:ee:ff

ip link add veth0 type veth peer name veth1     # veth pair
ip link add br0 type bridge                     # Linux bridge
ip link set veth0 master br0                    # add veth to bridge

ip link add link eth0 name eth0.10 type vlan id 10  # VLAN sub-iface
```

## Addresses

```bash
ip addr                                  # all v4+v6 addresses
ip -4 addr
ip -6 addr
ip addr show eth0
ip addr add 10.0.0.1/24 dev eth0
ip addr del 10.0.0.1/24 dev eth0
ip addr flush dev eth0
```

## Routing

```bash
ip route                                 # default table
ip -6 route
ip route show table all                  # all routing tables
ip route get 8.8.8.8                     # which route would match?
ip route add default via 10.0.0.254
ip route add 10.10.0.0/24 via 10.0.0.254 dev eth0
ip route add blackhole 198.51.100.0/24
ip route del default

ip rule                                  # policy routing rules
ip rule add from 10.0.0.0/24 table 100
```

## Neighbors (ARP / NDP)

```bash
ip neigh                                 # ARP/NDP table
ip neigh show dev eth0
ip neigh flush all
ip neigh flush dev eth0
ip neigh add 10.0.0.5 lladdr aa:bb:cc:dd:ee:ff dev eth0
arping -I eth0 10.0.0.1                  # send ARP request explicitly
```

## Network namespaces

```bash
ip netns add h1
ip netns list
ip netns delete h1
ip netns exec h1 ip a                    # run a command inside the netns
ip -n h1 addr add 10.0.0.1/24 dev veth-h1   # shorthand for `netns exec ip ...`
ip link set veth-h1 netns h1
```

## Bridges

```bash
bridge link                              # bridge ports
bridge fdb show                          # bridge MAC table
bridge fdb show br br0
bridge vlan show                         # VLAN-aware bridge state
```

## tc (traffic control)

The classic "qdisc"/"class"/"filter" model. For *labs*, `tc qdisc ... root netem` is the only line you usually need.

```bash
tc qdisc                                 # show all qdiscs
tc qdisc show dev eth0
tc qdisc add dev eth0 root netem delay 100ms          # add 100ms latency
tc qdisc add dev eth0 root netem loss 10%             # 10% loss
tc qdisc add dev eth0 root netem delay 50ms loss 1% reorder 25%
tc qdisc change dev eth0 root netem delay 50ms loss 5%
tc qdisc del dev eth0 root

tc qdisc add dev eth0 root tbf rate 1mbit burst 32kbit latency 400ms  # rate limit
```

## ENCAP / GRE / VXLAN

```bash
ip link add gre1 type gre local 10.0.0.1 remote 10.0.0.2 ttl 64
ip link add vxlan10 type vxlan id 10 dev eth0 dstport 4789 group 239.1.1.1
```

## Often-forgotten

```bash
ip -d link show eth0                     # detailed (shows GSO/TSO/etc.)
ethtool eth0                             # speed/duplex
ethtool -k eth0                          # offloads
ethtool -K eth0 tso off gso off          # disable offloads (debugging)
ethtool -S eth0                          # NIC stats

sysctl net.ipv4.ip_forward
sysctl -w net.ipv4.ip_forward=1
```
