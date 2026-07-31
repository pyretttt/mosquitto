# Week 08 theory: SDN data plane, OpenFlow, Cilium

Background reading for the SDN third of [README08.md](README08.md). Kurose & Ross 4.4 covers
generalized forwarding and the match-action abstraction well, so this is the shortest of the three
files. It fills in the parts K&R leaves as a sketch: what a flow entry contains beyond match and
action, why real pipelines are multi-table, and how the model maps onto Kubernetes CNIs — where
Cilium is the interesting case precisely because it *departs* from it.

This file contains **no answers** to the exercises. Each section ends with a pointer to the
question it arms you for.

## Where to look for what

One of three files for this week: [NAT](THEORY08-nat.md), [IPv6](THEORY08-ipv6.md),
**SDN** (this one).

| README08 item | Section |
| --- | --- |
| Learning goal: sketch an OpenFlow match-action table | [Anatomy of a flow entry](#anatomy-of-a-flow-entry) |
| Learning goal: why Kubernetes CNIs like the model | [Why CNIs like the model](#why-cnis-like-the-model) |
| Exercise 4 (stretch): OVS and `ovs-ofctl` flow rules | [Anatomy of a flow entry](#anatomy-of-a-flow-entry), [Multi-table pipelines](#multi-table-pipelines) |
| Self-check: data plane vs control plane, where Cilium fits | [Data plane, control plane, management plane](#data-plane-control-plane-management-plane), [Where Cilium actually sits](#where-cilium-actually-sits) |

Everything else in README08 is NAT or IPv6: see [THEORY08-nat.md](THEORY08-nat.md) and
[THEORY08-ipv6.md](THEORY08-ipv6.md). One direct dependency — the contrast between Cilium's
socket-level load balancing and kube-proxy's per-packet DNAT assumes the conntrack material from
the NAT file.

---

## Anatomy of a flow entry

The K&R sketch is match + action. A real OpenFlow flow entry has six parts, and the four extra ones
are where the operational behavior lives:

```
+---------------+----------+----------+--------------+----------+--------+
| match fields  | priority | counters | instructions | timeouts | cookie |
+---------------+----------+----------+--------------+----------+--------+
```

**Match fields.** OpenFlow 1.0 had a fixed 12-tuple. From 1.2 onward it is OXM, a TLV encoding, so
the match set is extensible — which is how OVS added registers, tunnel metadata, and (relevant to
last week's material) conntrack fields like `ct_state`, `ct_mark` and `ct_zone`. Every field can be
matched exactly, with a bitmask, or wildcarded entirely. Fields have **prerequisites**: you cannot
match `nw_dst` without also matching `dl_type=0x0800`, because the bytes at that offset are only an
IPv4 destination if the frame is IPv4. `ovs-ofctl` will silently add the prerequisite for you when
you use the `ip` shorthand, which is a frequent source of confusion when you then dump the flows
and see a match you did not type.

**Priority.** The highest-priority matching entry wins. The subtlety: **OpenFlow leaves ties
undefined.** Two overlapping entries at the same priority means the switch may pick either, and
different switches will pick differently. This is not a corner case — it is the single most common
way a hand-written flow table behaves non-deterministically. `ovs-ofctl add-flow --check-overlap`
refuses to install an entry that overlaps an existing one at the same priority, which is worth
using while you are learning.

**Counters.** Per-entry packet count, byte count and duration. Their presence in the *entry* rather
than in a separate telemetry system is a design decision worth noticing: the same table that
forwards is the table you read statistics from, so the controller's model of "what rules exist"
and "what traffic they matched" cannot drift apart.

**Instructions.** See [below](#instructions-versus-actions) — this changed meaningfully between
OpenFlow 1.0 and 1.1 and the distinction still trips people.

**Timeouts.** `idle_timeout` removes the entry after N seconds with no matching packets;
`hard_timeout` removes it N seconds after installation regardless of traffic. Zero means never.
These exist so that a controller can install reactive, per-flow state without having to remember to
clean it up — the table garbage-collects itself. A controller that installs only permanent entries
is using the model proactively, which is what production deployments actually do.

**Cookie.** An opaque 64-bit value the controller chooses. It never participates in matching. Its
purpose is bulk operations: tag every entry belonging to one policy or one tenant with the same
cookie, then modify or delete them in one message. If you have ever wondered how a controller
cleanly removes "everything I installed for pod X", this is the mechanism.

### A table you can actually read

```
priority  match                                            actions
--------  -----------------------------------------------  -----------------------------
   200    in_port=1, ip, nw_dst=10.0.0.5                    output:3
   200    in_port=1, ip, nw_dst=10.0.0.6                    output:4
   100    in_port=1, arp                                    flood
   100    dl_dst=00:11:22:33:44:55                          output:2
    50    ip, nw_dst=10.0.0.0/24                            goto_table:2
     0    *                                                 drop            <- table-miss
```

The last entry is the **table-miss** entry: priority 0, matching everything. In OpenFlow 1.0 an
unmatched packet was sent to the controller by default. From 1.3 the default is to **drop**, and if
you want packets punted to the controller you must install a table-miss entry that says so. This
reversal was deliberate — the old default made a controller disconnect turn into a control-channel
flood.

For exercise 4:

```bash
ovs-vsctl add-br br0
ovs-ofctl add-flow br0 "table=0,priority=200,ip,nw_dst=10.0.0.5,actions=output:3"
ovs-ofctl add-flow br0 "table=0,priority=0,actions=drop"
ovs-ofctl dump-flows br0
ovs-ofctl --check-overlap add-flow br0 "..."     # catch ambiguous priorities early
```

and the dump gives you every field discussed above in one line:

```
 cookie=0x0, duration=12.345s, table=0, n_packets=17, n_bytes=1666, idle_timeout=60,
 priority=200,ip,nw_dst=10.0.0.5 actions=output:3
```

The single most useful debugging tool is not `dump-flows` but the packet simulator, which walks a
hypothetical packet through every table and shows each resubmit and action:

```bash
ovs-appctl ofproto/trace br0 in_port=1,dl_type=0x0800,nw_dst=10.0.0.5
```

### Instructions versus actions

OpenFlow 1.0 had a flat list of actions, executed immediately in order. OpenFlow 1.1 replaced that
with **instructions**, which is what makes multi-table pipelines coherent:

- `Apply-Actions` — execute these now, immediately, mutating the packet before the next table sees
  it.
- `Write-Actions` — do not execute; merge into the packet's **action set**, which accumulates
  across tables and is executed once at the end of the pipeline.
- `Clear-Actions` — empty the accumulated set.
- `Write-Metadata` — set a 64-bit field that travels with the packet between tables and can be
  matched on later.
- `Goto-Table` — continue processing at a later table.
- `Meter` — rate-limit.

The action set is executed in an order fixed by the spec, not the order you wrote them: copy TTL
inwards, pop, push/copy TTL outwards, decrement TTL, set fields, QoS, group, output. That
determinism is the point — pipeline stages can each contribute an action without having to reason
about what the other stages did.

The practical difference: `Apply-Actions set_field` changes the packet that table 2 will match on;
`Write-Actions set_field` does not, because it happens at the end. Getting these backwards produces
a table that looks correct and behaves inexplicably.

*Arms you for:* the learning goal on sketching a match-action table, and exercise 4.

---

## Multi-table pipelines

The reason real switches have many tables is combinatorial, and it is the argument worth being able
to state.

Suppose you want to apply an ACL with M rules and then a forwarding decision with N rules. In one
table, every combination must be materialised as its own entry, because an entry has exactly one
match and one action: **M × N** entries. Split it into two tables and the ACL table holds M
entries, the forwarding table holds N, and the pipeline composes them at packet time: **M + N**.
With three independent concerns it is M×N×P versus M+N+P. This is the entire argument for the
multi-table pipeline, and it is the same argument as normalizing a database schema.

`Goto-Table` may only jump **forward**, to a higher-numbered table. That restriction is what
guarantees the pipeline terminates — you cannot build a loop. OVS adds a `resubmit` action that
lifts the restriction (you can re-enter any table, including one you came from), which is more
expressive and comes with the obligation not to write an infinite loop; OVS enforces a recursion
limit rather than a structural guarantee.

The OVS advanced tutorial builds a VLAN-aware learning switch as a worked example, and its stage
breakdown is a good template for how people actually decompose a pipeline:

```
 table 0   admission control    drop bogus source MACs, STP BPDUs, then goto 1
 table 1   VLAN input           determine the VLAN from the port config,      goto 2
 table 2   learn source         install a flow matching this src MAC,          goto 3
 table 3   lookup destination   known unicast -> a port; unknown -> flood,     goto 4
 table 4   output processing    tag or untag per the egress port's VLAN config
```

Each table has one concern and one reason to change. Table 2's `learn` action is worth seeing
because it is the data plane writing its own flow entries — a MAC-learning switch implemented with
no controller involvement at all, which is a useful demonstration that "SDN" does not have to mean
"a controller in the loop for every decision".

*Arms you for:* exercise 4, and the "why CNIs like the model" learning goal.

---

## Data plane, control plane, management plane

Stated precisely, because the self-check asks for it:

- **Data plane** (forwarding plane). The per-packet path. For each packet: look up state, act. It
  runs at line rate, so it must be simple and bounded. It contains no algorithms, only lookups.
- **Control plane.** Computes the state the data plane consults. It runs at *event* rate — a link
  goes down, a route is withdrawn, a pod is scheduled — which is many orders of magnitude slower
  than packet rate, so it can afford to be complex: Dijkstra, BGP best-path selection, policy
  compilation.
- **Management plane.** Configuration, monitoring, telemetry. The one everyone forgets in the
  answer, and the one you actually spend your day in: OVSDB, NETCONF, the CLI, the metrics
  endpoint.

The rate separation is the real content of the distinction. Anything that must happen per packet
belongs in the data plane and must therefore be a lookup; anything that can happen per event
belongs in the control plane and may therefore be an algorithm.

**Traditional router:** both planes live in the same chassis. Each box runs its own control plane
and computes its own forwarding table using distributed protocols, so network-wide behavior is
emergent from many local decisions and there is no single place that knows the whole state.

**The SDN proposition:** move the control plane off the box and centralise it — physically or just
logically — so it has a global view and network behavior can be *programmed* rather than coaxed out
of protocol interactions. The switch degrades to a programmable match-action pipeline. OpenFlow is
the southbound API between the two.

Where the boundary sits in OVS, since that is what you will run in exercise 4:

| Component | Plane |
| --- | --- |
| kernel datapath module (or DPDK/AF_XDP userspace datapath) | data plane, fast path — an exact-match "megaflow" cache |
| `ovs-vswitchd` | data plane, slow path — first packet of a flow, plus the OpenFlow table logic |
| `ovsdb-server` | management plane — bridge, port and interface configuration |
| external controller (Faucet, ONOS, `ovn-controller`) | control plane |

One honest complication for the answer: `ovs-vswitchd` contains a small control plane of its own,
because it compiles the OpenFlow tables into the megaflow entries the kernel caches. The three-plane
model is a lens, not a physical partition, and every real system smears the boundary somewhere.

*Arms you for:* the self-check on data plane vs control plane.

---

## Why CNIs like the model

Kubernetes networking is, structurally, the SDN problem restated:

1. **There is a single authoritative source of intent** — the API server, holding Pods, Services,
   EndpointSlices and NetworkPolicies. That is the SDN controller's global view, handed to you for
   free.
2. **Forwarding state is a pure function of that intent.** Given the set of objects, the correct
   forwarding tables are determined. This is exactly what an SDN controller does: compile declared
   policy into per-device state.
3. **The churn rate is enormous but is still event rate, not packet rate.** Pods come and go
   constantly, so the tables must be rewritten constantly — but a rewrite is still an event, and it
   is many orders of magnitude rarer than a packet. The clean split lets you absorb the churn in an
   agent that can be slow and correct, while packets take a path that is fast and dumb.
4. **The data plane must survive the control plane.** If the agent crashes or loses the API server,
   packets have to keep flowing on whatever tables were last programmed. A design where every
   forwarding decision needs the controller is unacceptable in a cluster; a design where the
   controller only *programs* is exactly right. This is why proactive flow installation, not the
   reactive controller-in-the-loop model of the original OpenFlow papers, is what production uses.

Two CNIs take this literally: **OVN-Kubernetes** and **Antrea** both use Open vSwitch and program
real OpenFlow tables, so exercise 4 is not an analogy for them — it is the actual mechanism, at a
smaller scale.

Cilium is the interesting one because it keeps the split and throws away the tables.

### Where Cilium departs from OpenFlow

Both are match-action. The difference is the lookup structure, and it is a real engineering
trade rather than a matter of taste.

| | OpenFlow / OVS | Cilium / eBPF |
| --- | --- | --- |
| State lives in | priority-ordered flow tables | typed eBPF maps: hash maps, LPM tries, arrays |
| Lookup | find the highest-priority match among wildcard entries | direct hash or longest-prefix lookup on a key |
| Cost as rules grow | grows; mitigated by a megaflow cache in front | O(1) for hash maps, independent of entry count |
| Expressiveness | any combination of fields, wildcarded arbitrarily | whatever the program's author coded — fixed key shapes |
| The "pipeline" | declarative tables, composed with goto/resubmit | imperative C compiled to bytecode, with tail calls between programs |
| Where it runs | kernel datapath cache, with a userspace slow path | entirely in-kernel at XDP, tc ingress/egress, and cgroup socket hooks |
| Cache invalidation | a real problem — table changes invalidate megaflows | none; there is no cache, the map *is* the state |

The shape is identical: extract fields from the packet, look up state keyed on those fields, act on
the result. Cilium's Service handling is a hash-map lookup on `(destination IP, destination port,
protocol)` yielding a backend slot, then a rewrite — that is match-action with a different index.
What Cilium gives up is general wildcard matching over arbitrary field combinations. What it buys
is constant-time lookup, no cache layer to invalidate, and no userspace slow path for the first
packet of a flow.

Two departures worth calling out because they go beyond "same idea, different data structure":

**Identity-based policy.** Cilium does not match NetworkPolicy on IP addresses. It allocates a
numeric *security identity* per unique set of labels, and matches policy on identity. The identity
travels with the packet (in a VXLAN header field, an IPv6 option, or resolved by a map lookup at the
destination). This decouples policy from addressing entirely: scaling a Deployment from 3 to 300
pods changes zero policy entries, because all 300 share one identity. In a pure match-on-header
model, that is 300 table changes. Given the churn rate in point 3 above, this is arguably the
biggest single win.

**Socket-level load balancing.** For pod-to-Service traffic, Cilium can attach to the cgroup
`connect()` hook and rewrite the destination **in the socket layer, once, at connect time** — before
a packet has ever been built. Compare with kube-proxy in iptables mode, which builds long chains of
DNAT rules evaluated per packet and creates a conntrack entry per flow (all of which should look
very familiar after [THEORY08-nat.md](THEORY08-nat.md)). Cilium's version is still DNAT, but it has
been lifted out of the per-packet path completely: no conntrack entry for the service translation,
no per-packet rule traversal, and the cost does not grow with the number of Services.

### Where Cilium actually sits

For the self-check, the mapping is:

- **Control plane** — `cilium-agent`, one per node. Watches the Kubernetes API, allocates
  identities, computes the desired state, then writes eBPF maps and loads/replaces eBPF programs.
  `cilium-operator` handles cluster-scoped work such as IPAM and identity garbage collection.
- **Data plane** — the eBPF programs themselves, attached at XDP, tc ingress/egress and the cgroup
  socket hooks, plus the maps they read. Entirely in-kernel. Keeps forwarding unchanged if the
  agent dies.
- **Management plane** — the `cilium` CLI, the Helm configuration, and Hubble for flow
  observability.

One nuance that makes for a better answer than the textbook version: **Cilium is not a centralised
SDN controller.** There is no single controller programming every node. Each node's agent
independently computes its own state from a shared source of truth. The *configuration* is
centralised (the API server); the *control plane* is distributed. It sits somewhere between the
classical OpenFlow model and traditional distributed routing, and it took the useful half of each —
a global declarative source of intent, without a central component in the failure path.

*Arms you for:* the learning goal on why CNIs like the match-action model, and the self-check on
where Cilium fits.

---

## Reading list

All verified reachable. Where a piece is long, the section to actually read is named.

**Flow entries and OVS**

- [ovs-ofctl(8)](https://www.openvswitch.org/support/dist-docs/ovs-ofctl.8.html)
  — the reference for flow-entry anatomy as you actually type it: priority semantics including the
  undefined tie behavior, `idle_timeout` / `hard_timeout`, cookies, `check_overlap`,
  `reset_counts`, and the full action vocabulary. Read the "Flow Syntax" section.
- [ovs-fields(7)](https://man7.org/linux/man-pages/man7/ovs-fields.7.html)
  — the match half: exact, masked and wildcard matching, field prerequisites, and the history from
  OpenFlow 1.0's fixed 12-tuple to OXM. Skim the introduction, then use it as a lookup table.
- [Open vSwitch Advanced Features Tutorial](https://docs.openvswitch.org/en/latest/tutorials/ovs-advanced/)
  — build a VLAN-aware learning switch stage by stage with `resubmit`, priorities, registers and
  the `learn` action. This is the best preparation for exercise 4; do it instead of reading about
  pipelines.
- [OpenFlow in a Day (NANOG tutorial, Wallace)](https://archive.nanog.org/sites/default/files/mon.tutorial.wallace.openflow.31.pdf)
  — a slide deck, so terser than the docs above, but it lays out the flow-entry tuple and the
  1.0-to-1.1 shift from actions to instructions and action sets more compactly than anything else.
  Use it as a supplement, not a primary.

**The SDN split**

- [SDN definition (Open Networking Foundation)](https://opennetworking.org/sdn-definition/)
  — short and canonical, from the body that standardised OpenFlow. Enough for the self-check.
- The McKeown et al. 2008 OpenFlow paper is already in README08's optional reading. Sections 1-3
  are the part worth reading; the rest is a 2008 deployment story.

**Cilium and eBPF**

- [eBPF datapath: introduction (Cilium docs)](https://docs.cilium.io/en/stable/network/ebpf/intro/)
  — the page that connects most directly to the match-action model. Enumerates the kernel hooks
  (XDP, tc ingress/egress, socket ops) and the objects layered on them (Prefilter, Endpoint Policy,
  Service, L7 Policy). Read this one first.
- [Life of a packet (Cilium docs)](https://docs.cilium.io/en/stable/network/ebpf/lifeofapacket/)
  — short companion showing three flows with and without socket-layer enforcement. The diagrams
  tie the datapath objects together.
- [Kubernetes without kube-proxy (Cilium docs)](https://docs.cilium.io/en/stable/network/kubernetes/kubeproxy-free/)
  — the authoritative guide to `kubeProxyReplacement`, socket-level load balancing at `connect()`,
  DSR and Maglev hashing. Read the "socket LoadBalancer" section for the contrast with iptables
  DNAT.
- [Life of a packet in Cilium: pod-to-service traffic path and BPF processing logic (Arthur Chiao)](https://arthurchiao.art/blog/cilium-life-of-a-packet-pod-to-service/)
  — the deepest free walkthrough available: traces a pod-to-Service packet across two nodes hop by
  hop, inspecting the loaded BPF programs with ordinary Linux tools at each step. Read it after the
  two Cilium docs pages.

## Specs, lookup only

| Spec | Look at | For |
| --- | --- | --- |
| OpenFlow 1.3 spec | §5.1-5.4 | pipeline processing, flow-entry structure, matching and the table-miss entry |
| OpenFlow 1.3 spec | §5.10, §5.12 | instructions vs the action set, and the fixed action-set execution order |
| OpenFlow 1.3 spec | §5.6 | group tables, if you need multipath or multicast semantics |
| RFC 7426 | §3-4 | the IETF's SDN layer/plane taxonomy, if you want a more careful vocabulary than the vendor version |
