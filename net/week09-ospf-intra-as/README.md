# Week 9 - Ch5 part 1: Intra-AS routing (OSPF) + link-state vs distance-vector

You move into the *control plane*: how do routers learn which way to forward packets? You'll run OSPF in a 4-router netns lab.

## Reading

- Kurose & Ross 8e, 5.1-5.3 (~30 pages):
  - 5.1 Introduction
  - 5.2 Routing algorithms (LS vs DV, Dijkstra vs Bellman-Ford)
  - 5.3 Intra-AS routing (OSPF)
- RFC 2328 (OSPF v2) - skim sections 1-3.
- FRR documentation: https://docs.frrouting.org/

## Learning goals

- Solve a shortest-path problem by hand with Dijkstra (small graph) and verify with FRR.
- Explain count-to-infinity and how split-horizon / poison-reverse address it.
- Bring up OSPF on 4 routers, see LSAs in tcpdump, watch convergence after a link failure.
- Explain OSPF areas (single-area vs multi-area).

## Lab

**Goal:** OSPF on a 4-router topology, observe convergence.

We'll extend the netns model. Drop a script in `lab/netns/ospf-4-routers.sh` (you write it - the existing scripts are templates):

```
   r1 --- r2
   |       |
   r3 --- r4
```

Each router runs FRR with OSPF. You can run FRR inside each netns by `apt install frr` and starting `ospfd` with a config in each netns. Or, use a docker-based approach with the `frrouting/frr` image.

Alternative quick path - `kathara`/`netkit` or `containerlab` if you'd like a higher-level lab framework. Start simple with FRR.

```bash
sudo apt install -y frr
echo 'ospfd=yes' | sudo tee -a /etc/frr/daemons

sudo ip netns exec r1 vtysh
# (config) router ospf
# (config) network 10.0.0.0/24 area 0
# (config) network 10.1.0.0/24 area 0
```

Inside the lab:

```bash
sudo ip netns exec r1 vtysh -c 'show ip ospf neighbor'
sudo ip netns exec r1 vtysh -c 'show ip route'
sudo ip netns exec r2 ip link set veth-r2-r4 down   # break a link
sudo ip netns exec r1 vtysh -c 'show ip route'      # watch convergence
```

Capture LSAs:

```bash
sudo ip netns exec r1 tcpdump -ni any -e proto ospf -w /tmp/ospf.pcap
```

## Exercises

1. **Conceptual.** In `notes.md`:
   - Solve 2 shortest-path problems by hand (book end-of-chapter). Pick a 5-6 node graph.
   - Explain why distance-vector can suffer count-to-infinity but link-state can't.
   - Why does OSPF have "areas"? When would you split a network into multiple areas?
2. **Practical.** Bring up the 4-router OSPF lab. Capture an LSA. Open it in Wireshark and identify the LSA type (Router-LSA, Network-LSA, ...).
3. **Practical - convergence timing.** With `ping` running between r1 and r4, break a link and measure how long traffic blackholes before OSPF reroutes. Repeat with different `hello` and `dead` intervals. Note the trade-off.
4. **Stretch.** Try IS-IS instead of OSPF (it's the protocol most ISPs actually use) and compare the config.

## Self-check

- Without notes: explain LS (link-state) vs DV (distance-vector) in 1 sentence each.
- What is an OSPF LSA and what types exist?
- What does "convergence" mean and what are the levers to make it faster?
- Why don't end hosts run OSPF?

## Useful one-liners

```bash
vtysh -c 'show ip ospf'
vtysh -c 'show ip ospf database'
vtysh -c 'show ip ospf interface'
vtysh -c 'show ip route ospf'
```
