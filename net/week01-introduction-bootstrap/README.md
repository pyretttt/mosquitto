# Week 1 - Ch1 part 1: What the Internet is + lab bootstrap

The orientation week. You're (re)reading what the Internet *is*, what protocols *are*, and what packet- vs circuit-switching trade-offs look like - while also getting your Linux lab VM ready so future weeks don't lose time on tooling.

## Reading

- Kurose & Ross 8e, sections 1.1-1.3 (~30 pages):
  - 1.1 What is the Internet?
  - 1.2 The network edge
  - 1.3 The network core (packet vs circuit switching)
- (skim) RFC 1958 "Architectural Principles of the Internet" - 8 pages, classic.
- (optional) Cloudflare "What is a CDN?" intro post for a quick services-view reminder.

## Learning goals

- Articulate the difference between the *nuts-and-bolts* and *services* views of the Internet.
- Explain why we mostly use packet switching and what circuit switching trades off (predictable bandwidth vs efficiency).
- Sketch the hierarchy: end systems -> access network -> ISP -> regional ISP -> tier-1 -> IXP.
- Get a Linux VM running with the full DevOps networking toolbox installed.

## Lab

**Goal:** stand up the Linux lab environment that the rest of the course depends on.

1. Read [lab/netns/SETUP.md](../lab/netns/SETUP.md) and bring up `colima` (or `multipass` / a real Linux box).
2. Inside the VM, install the toolbox listed in `SETUP.md` (`iproute2`, `tcpdump`, `tshark`, `dnsutils`, `nftables`, `iperf3`, `mtr`, `nmap`, `netcat`, `socat`, `python3`, ...).
3. Run the smoke test:
   ```bash
   sudo ip netns add testns && sudo ip netns delete testns && echo ok
   sudo nft list tables && echo ok
   ```
4. Bring up the simplest lab to confirm everything is wired:
   ```bash
   cd /path/to/net/lab/netns
   sudo bash two-hosts.sh
   sudo ip netns exec h1 ping -c 3 10.0.0.2
   sudo bash teardown.sh
   ```

## Exercises

1. **Conceptual.** Answer in `notes.md`:
   - What is *propagation delay* vs *transmission delay*? Which one cares about distance, which one cares about bandwidth?
   - Why is packet switching "better" for bursty traffic? When would circuit switching make sense in 2026?
   - Where does *your* home router live in the ISP hierarchy?
2. **Practical.** Run `traceroute` (or `mtr --report-wide`) from your machine to 5 sites you use daily (e.g. `cursor.com`, `github.com`, `cloudflare.com`, `aws.com`, a Russian-region site). Look up each AS along the way with `whois -h whois.cymru.com " -v <ip>"`. Save the output as `exercises/01-traceroute.txt` and a hand-drawn or `mermaid` map in `notes.md`.
3. **Stretch.** Open https://stat.ripe.net/, drop in your home IP, look at the AS, IXPs, prefixes. Write a paragraph in `notes.md` explaining what you found.

## Self-check

- Can you draw the OSI 7 layers vs the TCP/IP 5 layers, and say which the book uses?
- Can you give two reasons (one technical, one economic) why ISPs peer at IXPs?
- Did you actually `ping` between two namespaces? If not, do it before moving on.

## Common gotchas

- On macOS host, `ip netns` doesn't exist. You *must* run inside a Linux VM.
- `colima` defaults to 2 CPU / 2 GB. The labs are happier with 4 GB.
- If `traceroute` shows `* * *` everywhere, your home router or ISP is rate-limiting ICMP. Try `traceroute -T -p 443` (TCP to 443) instead.
