# Linux networking CLI practice

This track teaches Linux networking tools by first teaching the ideas those
tools inspect. You do not need a prior networking course. You do need a
terminal, Docker, and the willingness to type commands instead of pasting
them.

The goal is not to memorize flags. The goal is to look at a broken path, name
the failed layer, pick the tool that can prove it, and change one thing.

## What you need before page one

Required:

- Comfort running commands in a shell (`cd`, `ls`, editing a file, reading
  error text).
- Docker Engine or Docker Desktop, with permission to run privileged
  containers.
- Roughly 45–60 minutes per module, plus a later pass on the incidents.

Not required:

- A computer-networks course, Kurose & Ross, or weeks 1–8 of the parent
  `net/` curriculum.
- IPv6, DNS, VLANs, BGP, Kubernetes, or cloud networking.
- Prior use of `ip`, `tcpdump`, `iptables`, or `nft`.

If a module uses a word you have not seen, look it up in
[GLOSSARY.md](GLOSSARY.md). If the glossary definition is still too thin, the
module that introduces the term is listed next to it.

## The one topology

Almost every command in this track is about this picture:

```text
clp-client              clp-router                 clp-server
10.20.1.10/24     10.20.1.1 | 10.20.2.1      10.20.2.10/24
    c-eth0 -------- r-left   |   r-right -------- s-eth0
         subnet 10.20.1.0/24 | subnet 10.20.2.0/24
```

Read it as three computers and two cables. The client can talk directly to
the left side of the router. The server can talk directly to the right side.
The client and server are **not** on the same cable, so a packet from client
to server must be **forwarded** by the router.

Everything runs in one privileged Linux container. Inside it, Linux **network
namespaces** behave like separate hosts. Lab scripts only create, change, or
delete namespaces whose names start with `clp-`.

A later NAT incident uses different addresses on purpose. Until then, live in
this picture.

## Start the lab

From macOS (or any host with Docker), in this directory:

```bash
./lab.sh up
./lab.sh shell
```

`up` builds and starts a Debian container with the tools installed. `shell`
opens a root shell **inside** that container. The prompt you see after
`shell` is Linux, not macOS. All `ip`, `ss`, `tcpdump`, and firewall commands
belong there.

From that Linux shell:

```bash
./scripts/setup-routed.sh
./scripts/setup-services.sh
ip netns exec clp-client ping -c 2 10.20.2.10
./scripts/reset.sh
```

Leave the container with `exit`. Stop it from macOS with `./lab.sh down`.

If `./lab.sh up` fails, Docker is not running, not installed, or not allowed
to start privileged containers. Fix that on the host before continuing. This
track cannot run in an unprivileged container: creating namespaces and
firewalls requires real kernel capabilities.

If you are already inside the container and a command says it must run as
root, you are in the right place — the lab shell is root. If you accidentally
run setup scripts on macOS, they will refuse; they are Linux-only.

## How to read this track

Read the modules **in order**. Each one states what you already know, what it
adds, and what it refuses to cover yet.

| # | Module | What it gives you |
|---|--------|-------------------|
| 0 | [Foundations](00-foundations.md) | Packets, addresses, subnets, ARP, TCP, and how the lab fakes three hosts |
| 1 | [A repeatable debugging method](01-debugging-method.md) | The question order you will use for every later failure |
| 2 | [`ip`](02-ip.md) | Interfaces, addresses, neighbors, routes |
| 3 | [`ss`](03-ss.md) | Sockets and TCP state |
| 4 | [`tcpdump`](04-tcpdump.md) | Packet evidence at a chosen observation point |
| 5 | [`mtr` and `iperf3`](05-mtr-iperf3.md) | Path latency versus path capacity |
| 6 | [`iptables`](06-iptables.md) | Legacy netfilter: filter, NAT, rule order |
| 7 | [`nft`](07-nftables.md) | Modern netfilter with the same hooks |
| 8 | [`conntrack` and NAT](08-conntrack-nat.md) | Kernel flow state and address translation |
| 9 | [Integrated troubleshooting](09-troubleshooting.md) | Choosing the next observation from the last evidence |

Then solve the [incidents](challenges/README.md). Solutions live in
[solutions](solutions/README.md), away from the problem statements. After the
first pass, use the [ten-minute recall drills](DRILLS.md) instead of rereading
whole modules.

Coming from the 16-week `net/` course: start at module 0 anyway. Skim any
section whose self-check you can already answer out loud. Do not skip the
debugging method or the drills; those are the point of this track.

## The learning loop

Use one 45–60 minute session per module. Foundations may take two sessions.
Do not copy commands into a personal script: typing is part of the exercise.

1. **Recall for 5 minutes.** On paper, write the commands and ideas you
   remember. Blank pages are useful data.
2. **Read for 10–20 minutes.** Read the module's theory before the command
   lists. If a paragraph assumes a word you do not have, stop and use the
   glossary.
3. **Type for 15 minutes.** Perform every drill. Say what you expect before
   pressing Enter.
4. **Solve for 20 minutes.** Open a challenge only after the matching
   modules, and investigate from symptoms.
5. **Record for 5 minutes.** Write the evidence that proved the cause and the
   smallest fix.
6. **Repeat tomorrow.** Redo the drills without the page, then use spaced
   reviews after 3, 7, and 14 days.

The objective is to remember the stable grammar, the first command to run,
and how to ask the tool for more detail.

## Which tool should I reach for?

- “Does this host have the right interface, address, neighbor, and route?”:
  `ip`
- “Is anything listening, and what state is this connection in?”: `ss`
- “Did a packet actually cross this interface, and what was in it?”: `tcpdump`
- “At which hop does latency or loss begin?”: `mtr`
- “What throughput, jitter, or loss can this path carry?”: `iperf3`
- “Which legacy firewall/NAT rule matched?”: `iptables`
- “What does the current modern firewall ruleset do?”: `nft`
- “What flow state and NAT translation does the kernel remember?”:
  `conntrack`

Use tools as a chain of evidence, not as substitutes for one another. A failed
`curl` does not prove packet loss. A clean `ping` does not prove TCP/8080
works.

## Completion standard

You are finished when, without notes, you can:

- explain why the client and server need a router, using the two `/24`
  prefixes;
- create and inspect the routed two-subnet topology;
- predict `ip route get` before running it;
- identify a listener and a stuck TCP state with `ss`;
- capture one flow with a narrow `tcpdump` filter on the correct interface;
- distinguish path latency from throughput with `mtr` and `iperf3`;
- explain the netfilter hook a packet reaches and read rule counters;
- implement SNAT, DNAT, and a stateful forward policy in both interfaces;
- correlate original and reply tuples in `conntrack`;
- diagnose every incident and state the evidence, not merely the fix.

## Possible gaps this track will not close

This lab is IPv4, one router, no DNS, no TLS, and no dynamic routing protocol.
That is intentional. Each module calls out the gap at the moment it would
otherwise confuse you.

You will **not** finish this track knowing:

- how names become addresses (DNS);
- how browsers set up HTTPS (TLS);
- how the Internet finds a path across many organizations (BGP);
- how switches, VLANs, or overlays isolate tenants;
- how Kubernetes or AWS encode the same ideas in different objects.

You **will** finish able to walk a packet from process to cable to router and
back, which is the prerequisite those topics actually need.

When a module skips something on purpose, it uses a **Possible gap** heading.
Those are not extra homework. They are there so a missing idea is named
instead of turning into a vague feeling that you are unprepared.

| Gap | Where it is named | Why it can wait |
|-----|-------------------|-----------------|
| IPv6 | [foundations](00-foundations.md) §2, [`ip`](02-ip.md) | Same jobs, different spelling (`-6`, NDP) |
| Subnetting by hand | [foundations](00-foundations.md) §3 | This lab only uses `/24`, `/32`, and `/0` |
| Switches and VLANs | [foundations](00-foundations.md) §5 | veth is a direct cable; L2 switching is a later course |
| DNS, TLS, HTTP versions | [foundations](00-foundations.md) §9, [`mtr` / `iperf3`](05-mtr-iperf3.md) | The lab types raw IPs and cleartext HTTP |
| PMTUD black holes | [debugging method](01-debugging-method.md), [troubleshooting](09-troubleshooting.md) | No included incident requires the fix |
| Policy routing / VRFs | [`ip`](02-ip.md) | One main table is enough |
| UDP / Unix sockets | [`ss`](03-ss.md) | Incidents are TCP/IPv4 |
| Checksum offload artifacts | [`tcpdump`](04-tcpdump.md) | Easy to over-read in a VM |
| Application vs network time | [`mtr` / `iperf3`](05-mtr-iperf3.md) | Lab HTTP is trivial |
| Docker/Kubernetes chains | [`iptables`](06-iptables.md) | Same hooks, extra generated rules |
| nft `bridge`/`netdev` families | [`nft`](07-nftables.md) | Unused here |
| Conntrack helpers, hairpin NAT | [conntrack](08-conntrack-nat.md) | Incident 4 is plain SNAT |
| Cloud SG/NACL, kube-proxy | [troubleshooting](09-troubleshooting.md) | Same questions, different objects |

## Safety

Run destructive commands such as `nft flush ruleset`, `iptables -F`, and
`conntrack -F` only inside a named lab namespace. On a real host these
commands can remove remote access, Docker/Kubernetes networking, or active
production state.

The lab container is privileged. Treat it as a scratch kernel, not as a
place to experiment with the Docker bridge or the host's real NICs.
