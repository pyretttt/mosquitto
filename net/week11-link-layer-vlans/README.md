# Week 11 - Ch6 link layer + Ch7 wireless skim

The week we go below IP. ARP, Ethernet, switches, VLANs. We skim wireless because the bottom of L2 isn't your daily concern, but we cover enough to be useful when you debug a Wi-Fi issue.

## Reading

- Kurose & Ross 8e, 6.1-6.7 (~50 pages):
  - 6.1 Introduction to the link layer
  - 6.2 Error detection (CRC) - skim
  - 6.3 Multiple access protocols - skim CSMA/CD history
  - 6.4 Switched LANs (ARP, Ethernet, switches)
  - 6.5 Link virtualization (MPLS)
  - 6.6 Data center networking
  - 6.7 A day in the life of a web request - **read carefully**, it's a top-to-bottom recap
- 7.1-7.3 (~15 pages, *skim* only): wireless characteristics, 802.11 frame format, MAC algorithms.
- (optional) "Networking, ACK!" zine for casual reinforcement.

## Learning goals

- Read an Ethernet frame: dst MAC, src MAC, ethertype (0x0800 IPv4, 0x86dd IPv6, 0x8100 802.1Q tag, 0x8847 MPLS).
- Walk through ARP request/reply on the wire.
- Bridge multiple netns through a Linux `bridge` device. Add a VLAN tag.
- Explain why a switch with no ARP table starts as a hub (flooding) and "learns" over time.
- Articulate switch security pitfalls: MAC flooding, ARP spoofing, VLAN hopping.

## Lab

**Goal:** build a small switched LAN and play with VLANs.

1. Bring up two hosts behind a Linux bridge:
   ```bash
   sudo ip netns add h1
   sudo ip netns add h2
   sudo ip netns add sw

   sudo ip link add veth-h1 type veth peer name veth-sw1
   sudo ip link add veth-h2 type veth peer name veth-sw2
   sudo ip link set veth-h1 netns h1
   sudo ip link set veth-h2 netns h2
   sudo ip link set veth-sw1 netns sw
   sudo ip link set veth-sw2 netns sw

   sudo ip netns exec sw ip link add br0 type bridge
   sudo ip netns exec sw ip link set veth-sw1 master br0
   sudo ip netns exec sw ip link set veth-sw2 master br0
   sudo ip netns exec sw ip link set br0 up
   sudo ip netns exec sw ip link set veth-sw1 up
   sudo ip netns exec sw ip link set veth-sw2 up

   sudo ip netns exec h1 ip addr add 10.0.0.1/24 dev veth-h1
   sudo ip netns exec h2 ip addr add 10.0.0.2/24 dev veth-h2
   sudo ip netns exec h1 ip link set veth-h1 up
   sudo ip netns exec h2 ip link set veth-h2 up
   sudo ip netns exec h1 ip link set lo up
   sudo ip netns exec h2 ip link set lo up
   ```
2. Watch ARP:
   ```bash
   sudo ip netns exec h1 ip neigh flush all
   sudo ip netns exec h1 tcpdump -ni veth-h1 -e &
   sudo ip netns exec h1 ping -c 1 10.0.0.2
   ```
   You'll see ARP request (broadcast `ff:ff:ff:ff:ff:ff`) -> ARP reply (unicast).
3. Watch the bridge's MAC learning table:
   ```bash
   sudo ip netns exec sw bridge fdb show br br0
   ```
4. **VLANs.** Add VLAN sub-interfaces:
   ```bash
   sudo ip netns exec h1 ip link add link veth-h1 name veth-h1.10 type vlan id 10
   sudo ip netns exec h1 ip addr add 10.10.10.1/24 dev veth-h1.10
   sudo ip netns exec h1 ip link set veth-h1.10 up
   ```
   Configure the bridge as a VLAN-aware switch (`bridge vlan add ...`).

## Exercises

1. **Conceptual.** In `notes.md`:
   - Walk through what happens between sending `curl 10.0.0.2` from h1 and the first ICMP packet appearing on h2 - including ARP, switch MAC learning, frame fields.
   - What's the difference between *access* and *trunk* ports? When do you use each?
   - Why does putting two switches in a loop without STP create a broadcast storm?
2. **Practical - bridge MAC learning.** Capture frames at the bridge with `tcpdump -ni br0 -e` and show how the bridge floods first, then learns, then unicasts.
3. **Practical - ARP spoofing demo (lab only).** With `arpspoof` from `dsniff`, poison h1's ARP cache so it sends traffic for h2 to a third host. **Only do this in a netns.** Document mitigations (Dynamic ARP Inspection, port security, static ARP).
4. **Stretch - VXLAN.** Bridge two namespaces over a VXLAN tunnel. This is exactly the model k8s overlay networking is built on.

## Self-check

- Without notes: write the Ethernet header layout. What ethertype is IPv4? IPv6? VLAN tag? ARP?
- What does a switch do that a hub doesn't? Why was that a big deal in the 90s?
- What's *gratuitous ARP* and when do you see it?
- Why do data centers prefer L3 ECMP fabrics over big L2 domains today?

## Useful one-liners

```bash
ip neigh                                # ARP table (kernel calls it 'neighbour')
ip neigh flush dev eth0
arping -I eth0 10.0.0.1                  # send a manual ARP request
bridge fdb show                          # bridge MAC table
bridge vlan show                         # VLAN-aware bridge state
```
