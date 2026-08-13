# Linux networking CLI practice

This track fills the gap between understanding packets and operating Linux
networking tools without searching for every flag. It assumes weeks 1-8 are
complete. It does not replace weeks 9-16.

The main topology is:

```text
clp-client              clp-router                 clp-server
10.20.1.10/24     10.20.1.1 | 10.20.2.1      10.20.2.10/24
    c-eth0 -------- r-left   |   r-right -------- s-eth0
```

Everything runs in one privileged Linux container. Inside it, network
namespaces behave like three separate hosts. The scripts only touch namespaces
whose names start with `clp-`.

## Start here

From macOS:

```bash
cd /Users/bob/mosquitto/net/cli-practice
./lab.sh up
./lab.sh shell
```

From the Linux shell that opens:

```bash
./scripts/setup-routed.sh
./scripts/setup-services.sh
ip netns exec clp-client ping -c 2 10.20.2.10
./scripts/reset.sh
```

Leave the runtime with `exit`. Stop it from macOS with `./lab.sh down`.

## The learning loop

Use one 45-60 minute session per module. Do not copy commands into a personal
script: typing is part of the exercise.

1. **Recall for 5 minutes.** On paper, write the commands you remember.
2. **Read for 10 minutes.** Read only the module's theory and command anatomy.
3. **Type for 15 minutes.** Perform every drill. Say what you expect before
   pressing Enter.
4. **Solve for 20 minutes.** Open a challenge and investigate from symptoms.
5. **Record for 5 minutes.** Write the evidence that proved the cause and the
   smallest fix.
6. **Repeat tomorrow.** Redo the drills without the page, then use spaced
   reviews after 3, 7, and 14 days.

The objective is not to remember every option. Remember the stable grammar,
the first command to run, and how to ask the tool for more detail.

## Modules

Read these in order:

1. [A repeatable debugging method](00-debugging-method.md)
2. [`ip`: interfaces, addresses, neighbors, routes](01-ip.md)
3. [`ss`: sockets and TCP state](02-ss.md)
4. [`tcpdump`: packet evidence](03-tcpdump.md)
5. [`mtr` and `iperf3`: path and performance](04-mtr-iperf3.md)
6. [`iptables`: legacy netfilter interface](05-iptables.md)
7. [`nft`: modern netfilter interface](06-nftables.md)
8. [`conntrack` and NAT](07-conntrack-nat.md)
9. [Integrated troubleshooting](08-troubleshooting.md)

Then solve the [incidents](challenges/README.md). Solutions are deliberately
kept in [solutions](solutions/README.md), away from the problem statements.
After the first pass, use the [ten-minute recall drills](DRILLS.md) instead of
rereading whole modules.

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
`curl` does not prove packet loss. A clean `ping` does not prove TCP/8080 works.

## Completion standard

You are finished when, without notes, you can:

- create and inspect a routed two-subnet topology;
- predict `ip route get` before running it;
- identify a listener and a stuck TCP state with `ss`;
- capture one flow with a narrow `tcpdump` filter on the correct interface;
- distinguish path latency from throughput with `mtr` and `iperf3`;
- explain the netfilter hook a packet reaches and read rule counters;
- implement SNAT, DNAT, and a stateful forward policy in both interfaces;
- correlate original and reply tuples in `conntrack`;
- diagnose every incident and state the evidence, not merely the fix.

## Safety

Run destructive commands such as `nft flush ruleset`, `iptables -F`, and
`conntrack -F` only inside a named lab namespace. On a real host these commands
can remove remote access, Docker/Kubernetes networking, or active production
state.
