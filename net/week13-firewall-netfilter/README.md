# Week 13 - Ch8 part 2: IPsec, firewalls, IDS + Linux netfilter deep dive

This week pulls together everything you've seen and adds the security layer that runs on every Linux box: netfilter / nftables / conntrack. Cilium, kube-proxy, AWS SG, Docker networking - all of it ultimately runs on this machinery on a Linux node.

## Reading

- Kurose & Ross 8e, 8.6-8.9 (~30 pages):
  - 8.7 Network-layer security: IPsec, VPN
  - 8.8 Securing wireless LANs (skim)
  - 8.9 Operational security: firewalls and IDS
- nftables wiki: https://wiki.nftables.org/wiki-nftables/index.php/Main_Page
- "A Deep Dive into Iptables and Netfilter Architecture": https://www.digitalocean.com/community/tutorials/a-deep-dive-into-iptables-and-netfilter-architecture
- (optional) "WireGuard: Next Generation Kernel Network Tunnel" (Donenfeld 2017).

## Learning goals

- Understand the netfilter hooks (`prerouting`, `input`, `forward`, `output`, `postrouting`) and which "tables" (`raw`, `mangle`, `nat`, `filter`) attach where, and why nftables collapses these.
- Build a stateful firewall in nftables that allows established/related, drops new from outside, masquerades outbound.
- Inspect conntrack: see new/established/timewait entries, reason about conntrack table size and overflow behavior.
- Reason about IPsec (AH vs ESP, transport vs tunnel mode) at a high level.
- Stretch: bring up a WireGuard tunnel between two namespaces.

## Lab

### Stateful firewall on the NAT gateway

Reuse `lab/netns/nat-gateway.sh` and add a `filter` table on `r`:

```bash
sudo bash lab/netns/nat-gateway.sh

sudo ip netns exec r nft -f - <<'NFT'
table inet filter {
    chain forward {
        type filter hook forward priority 0;
        ct state established,related accept
        iifname "veth-rin" oifname "veth-rout" accept
        log prefix "[fwd-drop] " counter drop
    }
    chain input {
        type filter hook input priority 0;
        ct state established,related accept
        iifname "lo" accept
        ip protocol icmp accept
        log prefix "[in-drop] " counter drop
    }
}
NFT
```

Test:

```bash
sudo ip netns exec h1 ping -c 3 198.51.100.1
sudo ip netns exec public bash -c 'echo hi | nc -u 10.0.0.1 9000'   # should be dropped, not allowed in
sudo ip netns exec r conntrack -L
sudo ip netns exec r nft list ruleset
```

### Conntrack overflow

```bash
sudo ip netns exec r sysctl net.netfilter.nf_conntrack_count
sudo ip netns exec r sysctl net.netfilter.nf_conntrack_max

sudo ip netns exec h1 hping3 -S -p 9000 --flood --rand-source 198.51.100.1   # SYN flood
# in another shell, watch:
watch -n 0.5 'sudo ip netns exec r sysctl net.netfilter.nf_conntrack_count'
```

### WireGuard (stretch)

Set up a WireGuard tunnel between two namespaces:

```bash
sudo apt install -y wireguard
wg genkey | tee priv.key | wg pubkey > pub.key
# ... configure wg0 in each ns, exchange peers, ping over the tunnel
```

## Exercises

1. **Conceptual.** In `notes.md`:
   - Diagram the netfilter packet flow (prerouting -> [routing decision] -> forward/input -> output -> postrouting). Where does NAT happen? Where does filter happen?
   - Compare nftables vs iptables: list 3 reasons nftables is "the future".
   - What's the difference between AH and ESP? Why is ESP almost always what you want?
2. **Practical - default-deny firewall.** Build it in nftables on the router netns. Allow only outbound HTTP/HTTPS + DNS from h1. Verify with curl + dig + tcpdump.
3. **Practical - conntrack stress test.** Reproduce the SYN flood scenario, watch `nf_conntrack_count` climb, see what `dmesg` says when the table overflows. Mitigate with `tcp_syncookies=1` and document the result.
4. **Stretch - WireGuard.** Bring up a tunnel between two namespaces and `tcpdump -i wg0`.

## Self-check

- Without notes: which netfilter hook fires on a *forwarded* packet (one that's not destined for the local host)?
- What does `ct state established,related accept` actually mean?
- What's a conntrack zone and why might you need multiple?
- How does kube-proxy in iptables mode use netfilter?
- What's `tcp_syncookies` and when does it kick in?

## Useful one-liners

```bash
nft list ruleset
nft list chain inet filter forward
nft -a list ruleset                           # show handles for delete-by-handle
nft delete rule inet filter forward handle 7

conntrack -L
conntrack -E                                  # event stream
conntrack -F                                  # flush

sysctl net.netfilter.nf_conntrack_count
sysctl -w net.netfilter.nf_conntrack_max=262144

iptables-save                                 # legacy listing if a system uses iptables
iptables-translate -A INPUT ...               # convert iptables rule -> nftables
```
