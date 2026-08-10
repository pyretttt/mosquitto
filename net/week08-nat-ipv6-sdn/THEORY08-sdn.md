# Week 08 theory: software-defined networking (SDN)

Background reading for the SDN third of [README08.md](README08.md). Kurose & Ross 4.4
introduces *generalized forwarding* — the idea that a switch can match on many packet fields and
run a small action, not only "look up the destination IP." This file starts earlier than that
chapter: what problem SDN is trying to solve, what the words mean, and only then how OpenFlow and
Cilium use the idea.

This file contains **no answers** to the exercises. Each section ends with a pointer to the
question it arms you for.

You do **not** need prior SDN experience. You do need the usual networking basics (Ethernet, IP,
ports) and, for the Cilium comparison near the end, the conntrack material from
[THEORY08-nat.md](THEORY08-nat.md).

## Where to look for what

One of three files for this week: [NAT](THEORY08-nat.md), [IPv6](THEORY08-ipv6.md),
**SDN** (this one).

| README08 item | Section |
| --- | --- |
| Learning goal: sketch an OpenFlow match-action table | [Match-action in one picture](#match-action-in-one-picture), [What a flow entry contains](#what-a-flow-entry-contains) |
| Learning goal: why Kubernetes CNIs like the model | [Why this model fits Kubernetes](#why-this-model-fits-kubernetes) |
| Exercise 4 (stretch): OVS and `ovs-ofctl` flow rules | [What a flow entry contains](#what-a-flow-entry-contains), [Why more than one table](#why-more-than-one-table) |
| Self-check: data plane vs control plane, where Cilium fits | [Three jobs inside a network box](#three-jobs-inside-a-network-box), [Where Cilium fits](#where-cilium-fits) |

Everything else in README08 is NAT or IPv6. See those two files for the rest.

---

## The problem SDN is reacting to

Start with a normal router or switch — no new jargon yet.

Each box does two very different jobs:

1. **Move packets.** For every packet that arrives: look something up, decide an output port (or
   drop, or rewrite a header), send it on. This must be fast. A busy link can carry millions of
   packets per second.
2. **Decide what the lookups should say.** When a cable fails, a route is withdrawn, or a policy
   changes, something has to recompute the tables that step 1 consults. That happens far less
   often — when *events* happen, not when *packets* arrive.

On a traditional box, both jobs live on the same device. Every router runs its own routing
protocol (OSPF, BGP, …), builds its own forwarding table, and cooperates with neighbors. The
network as a whole has no single place that "knows the plan." Behavior emerges from many local
conversations.

That works, but it is awkward when you want something like:

- "Block this tenant from talking to that tenant, everywhere, now."
- "When a new workload appears, give it connectivity that matches this policy."
- "Change forwarding for a whole datacenter from one control point."

**Software-defined networking (SDN)** is the idea of separating those two jobs cleanly:

- keep the fast per-packet work on the switch (the **data plane**);
- move the "figure out the tables" work to software that can see the whole network (the **control
  plane**);
- talk between them with a clear interface.

Historically the best-known interface was **OpenFlow**: the controller writes rules into the
switch; the switch matches packets against those rules and runs the actions. You do not need
OpenFlow to understand SDN — it is one concrete API for the split. Kubernetes networking uses the
same *shape* of thinking even when the wire protocol is not OpenFlow.

*Arms you for:* the self-check vocabulary (data plane / control plane), before the details.

---

## Three jobs inside a network box

People often say "data plane vs control plane." A third piece belongs in the answer too.

| Job | Informal name | Rate it runs at | What it does |
| --- | --- | --- | --- |
| Forward packets | **Data plane** (also: forwarding plane) | Per packet | Look up state, act. Must stay simple. |
| Compute that state | **Control plane** | Per event (link down, pod start, route change) | Algorithms, policy, protocols. Can be slow and complex. |
| Configure and observe | **Management plane** | Human / ops rate | CLI, APIs, metrics, config databases. |

The important distinction is **rate**, not which process owns which file:

- anything that must happen for *every packet* belongs in the data plane → lookups only;
- anything that can wait for an *event* belongs in the control plane → algorithms allowed;
- the tools you use to configure and debug sit in the management plane.

**Traditional router:** control plane and data plane share a chassis. Each box computes its own
tables.

**SDN-style design:** the control plane is centralized (or at least logically centralized). It
has a global view, compiles intent into per-switch state, and pushes that state down. The switch
becomes a programmable pipeline of "if the packet matches X, do Y."

OpenFlow is one **southbound** API — "southbound" only means "toward the switches," as opposed to
northbound APIs that applications use to talk to the controller. You can forget the compass
metaphor; remember the split.

### Where the pieces sit in Open vSwitch (for exercise 4)

[Open vSwitch](https://www.openvswitch.org/) (OVS) is a software switch you will install for the
stretch exercise. Rough mapping:

| Component | Plane |
| --- | --- |
| Kernel datapath (or a fast userspace path like DPDK) | Data plane — fast path |
| `ovs-vswitchd` | Data plane slow path + OpenFlow table logic (and a bit of local "control" when it compiles tables into cache entries) |
| `ovsdb-server` | Management — bridges, ports, interfaces |
| External controller (optional: Faucet, ONOS, OVN, …) | Control plane |

Real systems smear the neat textbook lines. Treat the three planes as a lens, not a rigid org
chart.

*Arms you for:* self-check — data plane vs control plane.

---

## Match-action in one picture

Forget SDN branding for a moment. A **match-action** rule is just:

```
IF packet looks like THIS  →  DO that
```

Examples in plain language:

```
IF arrived on port 1 AND destination IP is 10.0.0.5  →  send out port 3
IF it is ARP                                          →  flood to all ports
IF nothing else matched                               →  drop
```

That is the whole abstraction Kurose & Ross call generalized forwarding. An **OpenFlow flow
table** is a list of such rules, each with a priority so the switch knows which one wins when
several could match.

A tiny table you can actually read:

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

Reading tips:

- Higher **priority** wins.
- `ip` / `arp` are shorthands for "this Ethernet type," not magic protocols the switch invents.
- The last line is the **table-miss** rule: priority 0, match anything. Modern OpenFlow defaults
  unmatched packets to **drop**. Older OpenFlow 1.0 sent them to the controller instead — which
  turned a dead controller into a flood of "what do I do with this packet?" messages. Install an
  explicit miss rule when you care about the behavior.

For exercise 4 you type rules like this:

```bash
ovs-vsctl add-br br0
ovs-ofctl add-flow br0 "table=0,priority=200,ip,nw_dst=10.0.0.5,actions=output:3"
ovs-ofctl add-flow br0 "table=0,priority=0,actions=drop"
ovs-ofctl dump-flows br0
```

A dump line includes more than match and action — that is the next section:

```
 cookie=0x0, duration=12.345s, table=0, n_packets=17, n_bytes=1666, idle_timeout=60,
 priority=200,ip,nw_dst=10.0.0.5 actions=output:3
```

The most useful debug tool walks a *hypothetical* packet through the tables:

```bash
ovs-appctl ofproto/trace br0 in_port=1,dl_type=0x0800,nw_dst=10.0.0.5
```

*Arms you for:* sketching a match-action table; starting exercise 4.

---

## What a flow entry contains

Textbooks often say "match + action." A real OpenFlow **flow entry** has six parts. The extra
four are where day-to-day behavior lives:

```
+---------------+----------+----------+--------------+----------+--------+
| match fields  | priority | counters | instructions | timeouts | cookie |
+---------------+----------+----------+--------------+----------+--------+
```

**Match fields.** Which packet bits must look a certain way. Early OpenFlow had a fixed list of
about a dozen fields. Newer versions use an extensible encoding (often called OXM — you can treat
that as "typed TLVs for match fields"). OVS can match tunnel metadata, registers, and even
conntrack-related fields such as `ct_state` — same family of ideas as [THEORY08-nat.md](THEORY08-nat.md).

You can match exactly, with a bitmask, or leave a field as a wildcard. Fields also have
**prerequisites**: you cannot ask for an IPv4 destination (`nw_dst`) unless the frame is IPv4
(`dl_type=0x0800`). `ovs-ofctl` often adds the prerequisite for you when you write `ip,...`. That
is why a dump can show match fields you did not type.

**Priority.** Highest matching priority wins. If two overlapping rules share the **same**
priority, OpenFlow does **not** define which one wins — different switches may disagree. That is
a common source of "my table looks fine but behaves randomly." While learning, prefer:

```bash
ovs-ofctl --check-overlap add-flow br0 "..."
```

**Counters.** Packet count, byte count, how long the entry has existed. Statistics live *on the
rule*, so "what rules exist" and "what traffic hit them" stay coupled.

**Instructions / actions.** What to do when the rule matches. See the next subsection — OpenFlow
1.1 made this richer than a flat "do these things in order" list.

**Timeouts.** `idle_timeout`: delete after N seconds with no hits. `hard_timeout`: delete N
seconds after install no matter what. Zero means "keep forever." Timeouts let a controller install
temporary per-flow state without remembering to clean up. Production systems usually install
**permanent** rules ahead of time (**proactive**), rather than asking the controller about every
new flow (**reactive**). Reactive was common in early demos; proactive is what clusters need.

**Cookie.** An opaque 64-bit tag chosen by the controller. It does **not** participate in
matching. It is for bulk admin: "delete everything I installed for tenant X" by cookie, in one
shot.

### Instructions versus actions (only if multi-table confuses you)

OpenFlow 1.0: a flat list of actions, run immediately in order.

OpenFlow 1.1+: **instructions**, which matter once you have several tables:

| Instruction | Meaning |
| --- | --- |
| `Apply-Actions` | Do these **now** (packet may change before the next table). |
| `Write-Actions` | Merge into an **action set**; run the set **once at the end** of the pipeline. |
| `Clear-Actions` | Empty that accumulated set. |
| `Write-Metadata` | Attach a 64-bit value the next tables can match. |
| `Goto-Table` | Continue at a later table. |
| `Meter` | Rate-limit. |

The action set runs in a **fixed** order defined by the spec (TTL tricks, header pushes/pops,
field sets, output, …), not in the order stages happened to write them. That is deliberate so
pipeline stages can each contribute without fighting over ordering.

Practical gotcha: `Apply-Actions` that set a field change what the **next** table sees;
`Write-Actions` that set a field do **not**, because the rewrite waits until the end.

*Arms you for:* learning goal on sketching a table; exercise 4.

---

## Why more than one table

One table can express anything — but the size explodes when concerns combine.

Suppose you have an access-control list with **M** rules and a forwarding decision with **N**
rules. In a **single** table you often need a separate entry for every combination: about
**M × N** rules. Split into two tables — ACL then forward — and you store about **M + N** rules;
the packet walks both tables. Three independent concerns: product versus sum. Same reason we
normalize database schemas.

`Goto-Table` may only jump **forward** (to a higher table number). That forbids loops, so the
pipeline always finishes. OVS also has `resubmit`, which *can* re-enter earlier tables; then you
must avoid infinite loops yourself (OVS caps recursion).

A common teaching pipeline (from the OVS advanced tutorial) looks like:

```
 table 0   admission     drop nonsense, then goto 1
 table 1   VLAN input    figure out VLAN from the port, goto 2
 table 2   learn source  remember this source MAC → port, goto 3
 table 3   find dest     known unicast → port; unknown → flood, goto 4
 table 4   output        add/remove VLAN tags for the egress port
```

Each table has one job. Table 2's `learn` action can install new flow entries from the data plane
itself — a MAC-learning switch with **no** controller in the path. SDN means "programmable split,"
not "controller consulted on every packet."

*Arms you for:* exercise 4; why CNIs like staged pipelines.

---

## Why this model fits Kubernetes

A Kubernetes cluster has the same structure as the SDN story, whether or not OpenFlow is involved.

1. **One place holds intent.** The API server stores Pods, Services, EndpointSlices,
   NetworkPolicies. That is the global view a controller wants.
2. **Forwarding state should follow from that intent.** Given the objects, you can compute what
   each node's data path should do. "Compile policy into device state" is the SDN control-plane
   job.
3. **Churn is high, but still event-rate.** Pods appear and disappear constantly, so tables are
   rewritten often — yet still vastly less often than packets arrive. Keep the smart, slower agent
   off the per-packet path.
4. **Packets must survive a dead control plane.** If the node agent crashes, existing programmed
   state should keep forwarding. A design that asks a central brain about every packet does not
   survive production. **Proactive** programming does.

Two CNIs (Container Network Interfaces — plugins that provide pod networking) take this literally
with Open vSwitch and real OpenFlow tables: **OVN-Kubernetes** and **Antrea**. Exercise 4 is a
small version of their mechanism.

**Cilium** keeps the *split* (smart agent vs fast path) but usually does **not** use OpenFlow
tables. It programs the Linux kernel with **eBPF** instead. Same idea, different lookup machinery.

### OpenFlow / OVS versus Cilium / eBPF

Both are match-action. The engineering trade is how you store and look up state:

| | OpenFlow / OVS | Cilium / eBPF |
| --- | --- | --- |
| State lives in | Priority-ordered flow tables | Typed maps (hashes, longest-prefix tries, arrays) |
| Lookup | Best (highest-priority) match among possibly wildcard rules | Direct hash or LPM on a fixed key shape |
| Cost as rules grow | Can grow; OVS mitigates with a fast-path cache ("megaflow") | Hash maps stay about O(1) in entry count |
| Flexibility | Arbitrary field combinations and wildcards | Whatever the program author coded |
| "Pipeline" | Tables + goto / resubmit | C compiled to bytecode; programs chain with tail calls |
| Where it runs | Kernel cache + userspace slow path for first packets | In-kernel hooks (XDP, tc, cgroup socket hooks) |
| Cache invalidation | Table edits can bust the megaflow cache | Maps *are* the state — no separate cache layer |

Shape is the same: pull fields from the packet → look up → act. Cilium Service load balancing is
essentially a hash on `(destination IP, port, protocol)` to pick a backend, then rewrite. You lose
fully general wildcard tables; you gain predictable lookup cost and no userspace first-packet slow
path for that work.

Two Cilium ideas that matter beyond "different dictionary structure":

**Identity-based policy.** Policy is not "match these pod IPs." Cilium assigns a numeric
**security identity** to a unique label set and matches on that identity. Scaling a Deployment
from 3 to 300 pods need not rewrite 300 policy entries — they share one identity. At high pod
churn, that is a large win over pure header-IP tables.

**Socket-level load balancing.** For pod → Service traffic, Cilium can hook `connect()` and
rewrite the destination **once, when the socket connects**, before packets exist. Compare
kube-proxy in iptables mode: long DNAT rule chains per packet plus a conntrack entry per flow
(see [THEORY08-nat.md](THEORY08-nat.md)). Still DNAT in spirit — but lifted out of the per-packet
path, so cost does not grow with the number of Services the same way.

### Where Cilium fits

For the self-check:

- **Control plane** — `cilium-agent` on each node: watches the API, allocates identities, computes
  desired state, loads eBPF programs and writes maps. `cilium-operator` does cluster-wide chores
  (IPAM, identity GC, …).
- **Data plane** — the eBPF programs and maps in the kernel. Keep forwarding if the agent dies.
- **Management plane** — `cilium` CLI, Helm values, Hubble for flow visibility.

Nuance worth saying out loud: **Cilium is not one central OpenFlow controller programming every
switch.** Each node's agent computes local state from the shared API server. Configuration intent
is centralized; the control plane is **distributed**. It takes the useful half of classical SDN
(declarative global intent) without putting a single controller on the critical path for every
node.

*Arms you for:* why CNIs like match-action; self-check on where Cilium sits.

---

## Mini glossary

Terms this file uses, in one place:

| Term | Meaning |
| --- | --- |
| SDN | Separate fast packet forwarding from the software that decides forwarding policy; make that policy programmable. |
| Data plane | Per-packet path: match and act. |
| Control plane | Event-driven software that computes what the data plane should contain. |
| Management plane | Config, CLI, monitoring. |
| OpenFlow | A protocol/API for a controller to program match-action rules on a switch. |
| Flow entry / flow rule | One match + priority + instructions (+ timers, counters, cookie). |
| Flow table | A priority-ordered set of flow entries; switches may have many tables. |
| OVS | Open vSwitch — software switch that speaks OpenFlow; used in exercise 4. |
| CNI | Kubernetes plugin interface for container/pod networking. |
| eBPF | Linux kernel mechanism to run small verified programs at hook points; Cilium's data path. |
| Proactive vs reactive | Install rules ahead of time vs ask the controller on first packet of a flow. |

---

## Reading list

All verified reachable. Where a piece is long, the section to actually read is named.

**Start here if the overview above is enough and you want practice**

- [Open vSwitch Advanced Features Tutorial](https://docs.openvswitch.org/en/latest/tutorials/ovs-advanced/)
  — build a VLAN-aware learning switch stage by stage. Best preparation for exercise 4.

**Flow entries and OVS**

- [ovs-ofctl(8)](https://www.openvswitch.org/support/dist-docs/ovs-ofctl.8.html)
  — how you actually type rules: priorities, timeouts, cookies, `check_overlap`, actions. Read
  "Flow Syntax."
- [ovs-fields(7)](https://man7.org/linux/man-pages/man7/ovs-fields.7.html)
  — match fields, wildcards, prerequisites. Skim the intro; keep as a lookup table.
- [OpenFlow in a Day (NANOG tutorial, Wallace)](https://archive.nanog.org/sites/default/files/mon.tutorial.wallace.openflow.31.pdf)
  — slides: flow-entry tuple and the shift from flat actions to instructions/action sets.

**The SDN split**

- [SDN definition (Open Networking Foundation)](https://opennetworking.org/sdn-definition/)
  — short and canonical. Enough for the self-check wording.
- McKeown et al. 2008 OpenFlow paper (optional in README08): sections 1–3 for the original
  motivation; the rest is a 2008 deployment story.

**Cilium and eBPF**

- [eBPF datapath: introduction (Cilium docs)](https://docs.cilium.io/en/stable/network/ebpf/intro/)
  — hooks (XDP, tc, socket) and how they map to match-action-ish objects. Read first.
- [Life of a packet (Cilium docs)](https://docs.cilium.io/en/stable/network/ebpf/lifeofapacket/)
  — short companion diagrams.
- [Kubernetes without kube-proxy (Cilium docs)](https://docs.cilium.io/en/stable/network/kubernetes/kubeproxy-free/)
  — socket-level load balancing vs iptables DNAT; read the "socket LoadBalancer" section.
- [Life of a packet in Cilium: pod-to-service (Arthur Chiao)](https://arthurchiao.art/blog/cilium-life-of-a-packet-pod-to-service/)
  — deep hop-by-hop walkthrough; read after the two Cilium doc pages above.

## Specs, lookup only

| Spec | Look at | For |
| --- | --- | --- |
| OpenFlow 1.3 spec | §5.1–5.4 | Pipeline, flow-entry structure, matching, table-miss |
| OpenFlow 1.3 spec | §5.10, §5.12 | Instructions vs action set; fixed action-set order |
| OpenFlow 1.3 spec | §5.6 | Group tables (multipath / multicast) if you need them |
| RFC 7426 | §3–4 | IETF plane/layer vocabulary, if you want more precision than vendor slides |
