# Incident solutions

Read this only after recording your own diagnosis. Handle numbers and
ephemeral ports vary; discover them from current output.

Each solution starts with the **pedagogical point**: the misconception the
incident exists to break. If you got the right fix for the wrong reason,
redo the evidence steps.

## Incident 1: return route

**Point:** a path has two lookups. Proving the client's route proves only
the forward half.

Prove the client route and inspect the reverse decision:

```bash
ip -n clp-client route get 10.20.2.10
ip -n clp-server route get 10.20.1.10
ip netns exec clp-client curl -v --connect-timeout 2 \
  http://10.20.2.10:8080/
```

Capture TCP/8080 at the server. The arriving SYN proves the forward path;
the missing usable reverse lookup identifies the server route as the
failure.

Restore the endpoint's default route:

```bash
ip -n clp-server route replace default via 10.20.2.1
```

Verify:

```bash
./checks/check-routed.sh
./checks/check-service.sh
```

## Incident 2: ordered firewall verdicts

**Point:** ping and HTTP are different protocols. A drop on TCP/8080 can
leave ICMP untouched. Rule order and handles, not a second competing
table, are the fix.

Confirm listener and flow:

```bash
ip netns exec clp-server ss -lntp sport = :8080
ip netns exec clp-router tcpdump -ni r-left -c 4 'tcp port 8080'
ip netns exec clp-router tcpdump -ni r-right -c 4 'tcp port 8080'
```

Inspect the active forward chain with counters and handles:

```bash
ip netns exec clp-router nft -a list ruleset
```

Generate exactly one request and list again. The TCP/8080 drop rule's
counter increments before an accepting path can complete. Delete that
specific rule by its discovered family, table, chain, and handle:

```bash
ip netns exec clp-router nft delete rule FAMILY TABLE CHAIN handle HANDLE
```

Use the actual scenario family, table, chain, and handle shown by
`nft -a list ruleset`; do not create a second competing base chain. Verify
HTTP and confirm unrelated forwarding behavior is unchanged.

## Incident 3: egress qdisc

**Point:** "slow" is not a layer. Direction, latency, loss, and capacity
can be impaired independently, and a Linux qdisc only shapes **egress** on
one interface.

Establish controlled measurements:

```bash
ip netns exec clp-client mtr -n -r -c 20 10.20.2.10
ip netns exec clp-client iperf3 -c 10.20.2.10 -t 10
ip netns exec clp-client iperf3 -c 10.20.2.10 -R -t 10
ip netns exec clp-client iperf3 -c 10.20.2.10 -u -b 10M -t 10
ip -n clp-router -s link show dev r-right
ip netns exec clp-router tc qdisc show dev r-right
```

`tc qdisc` output exposes the netem/rate parameters and location. Because a
qdisc acts on egress, `r-right` impairs router-to-server traffic.
Directional tests and captures let you distinguish that from an
endpoint-wide condition.

Remove only the root qdisc:

```bash
ip netns exec clp-router tc qdisc del dev r-right root
```

Repeat the exact durations, directions, stream counts, and UDP offered
rate. Do not compare unmatched tests.

## Incident 4: source NAT

**Point:** connectivity to a host that must not learn your private prefix
is a translation problem, not a "add a route on the server" problem.

First prove the asymmetry:

```bash
ip -n clp-client route get 198.51.100.10
ip -n clp-server route get 10.30.1.10
```

The server must not learn the private route, so translate the private
source to the router's external address.

### iptables implementation

```bash
ip netns exec clp-router iptables -t nat -A POSTROUTING \
  -s 10.30.1.0/24 -o r-outside \
  -j SNAT --to-source 198.51.100.254
```

### nftables implementation

After resetting the scenario, create a distinct lab table:

```bash
ip netns exec clp-router nft 'add table ip practice_nat'
ip netns exec clp-router nft \
  'add chain ip practice_nat postrouting {
     type nat hook postrouting priority srcnat;
   }'
ip netns exec clp-router nft add rule ip practice_nat postrouting \
  ip saddr 10.30.1.0/24 oifname "r-outside" \
  snat to 198.51.100.254
```

Create a fresh flow after rule changes. Compare:

```bash
ip netns exec clp-router tcpdump -ni r-inside 'tcp port 8080'
ip netns exec clp-router tcpdump -ni r-outside 'tcp port 8080'
ip netns exec clp-router conntrack -L -p tcp --dport 8080
./checks/check-nat.sh
```

The inside capture shows `10.30.1.10`; the outside capture shows
`198.51.100.254`. Conntrack retains the mapping required to translate
replies.
