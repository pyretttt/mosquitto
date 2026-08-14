# Ten-minute recall drills

Use this page after completing the modules. Prompts intentionally omit
commands. Type from memory; consult `--help` or a module only after one
honest attempt.

If a prompt names an object you cannot picture (a four-tuple, a hook, a
qdisc), you are missing theory, not a flag. Return to
[foundations](00-foundations.md) or the module listed in the
[glossary](GLOSSARY.md), then retry the drill.

Start each session:

```bash
./scripts/setup-routed.sh
./scripts/setup-services.sh
```

## Drill 0: foundations (two minutes, no commands required)

On paper:

1. draw the two subnets and mark which addresses are on-link to which;
2. write why the client ARPs for `10.20.1.1` and not for `10.20.2.10`;
3. write the three-way handshake in flags (`[S]`, `[S.]`, `[.]`);
4. name one topic this track refuses to teach, and why that is acceptable.

## Drill A: `ip`

In three minutes:

1. show brief links and IPv4 addresses in all three namespaces;
2. predict and query both endpoint routes to their peer;
3. show only failed neighbors;
4. show detailed counters and MTU for both router interfaces;
5. replace the client default route with the correct next hop.

## Drill B: `ss`

In two minutes:

1. identify the server processes listening on 8080 and 5201;
2. start a transfer and show only established TCP sockets;
3. filter the client view by destination port;
4. display RTT, MSS, congestion window, and retransmissions;
5. show the TCP state summary.

## Drill C: `tcpdump`

In three minutes, capture:

1. four ICMP packets at the client;
2. ARP with Ethernet headers at the router;
3. only TCP/8080 open/close flags;
4. one flow to a pcap and read it numerically;
5. the same HTTP flow on both router links.

Before each capture, name the expected source, destination, direction, and
stopping condition.

## Drill D: `mtr` and `iperf3`

In five minutes:

1. produce a numeric ten-cycle path report;
2. test TCP in both directions;
3. compare one and four streams;
4. send UDP at a stated rate;
5. correlate one test with `ss -ti`.

## Drill E: netfilter

Use `iptables` one day and `nft` the next:

1. establish default-deny forwarding;
2. permit established/related packets;
3. permit new TCP/8080 from client side to server side;
4. prove counters with one request;
5. add and remove an ICMP rule;
6. display restorable/current syntax;
7. remove only your lab rules.

## Drill F: NAT and conntrack

Start `nat-missing.sh`:

1. watch flow events;
2. prove the missing reverse route;
3. add narrow source NAT;
4. capture pre- and post-translation;
5. draw original/reply tuples;
6. inspect NAT counters;
7. run the validator.

Alternate `iptables` and `nft`.

## Four-week rotation

- Week 1: foundations + debugging method, then modules in order; repeat
  that day's drill immediately. Foundations may consume two days.
- Week 2: A+B, C, D, E, F on separate days; solve incidents 1 and 2.
- Week 3: pair tools randomly; solve incidents 3 and 4.
- Week 4: one unknown incident, one full topology build by hand, then all
  retrieval tests without notes.

Track only three facts per session:

```text
Prompt I could not answer:
Command pattern I will recall:
Evidence I misinterpreted:
```

Do not measure progress by commands typed. Measure whether you selected the
correct observation point and made a justified next decision.
