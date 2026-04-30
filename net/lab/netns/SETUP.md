# Linux network namespace lab setup

macOS doesn't have `ip netns`, real `iptables`/`nftables`, or proper `tc`. We need a Linux kernel.

## Option A: colima (recommended)

```bash
brew install colima docker kubectl

colima start --cpu 2 --memory 4 --disk 20

colima ssh
```

Once inside the VM:

```bash
sudo apt update
sudo apt install -y \
  iproute2 iputils-ping \
  tcpdump tshark \
  dnsutils ldnsutils \
  curl wget netcat-openbsd socat \
  iperf3 mtr-tiny \
  nftables iptables conntrack \
  ethtool \
  bridge-utils vlan \
  hping3 \
  python3 python3-pip
```

## Option B: multipass

```bash
brew install multipass
multipass launch --name netlab --cpus 2 --memory 4G --disk 20G 22.04
multipass shell netlab
```

Same `apt install` line.

## Option C: a real Linux machine or a cloud VM

If you have one, just `ssh` in and `apt install` the same packages. Skip the Mac VM dance entirely.

## Verify

Inside the Linux box:

```bash
sudo ip netns add testns
sudo ip netns delete testns
echo "ok"

sudo nft list tables
echo "ok"
```

If both succeed you are ready for the labs.

## Persistent vs scratch

The scripts in this folder are idempotent in the sense that `teardown.sh` always cleans up. They are not idempotent across re-runs while still active - run teardown, then re-run setup. This keeps them simple.

## Why root

Network namespaces and netfilter rules require `CAP_NET_ADMIN`. We use `sudo`. If you mind, configure `unshare -rUn` for unprivileged user namespaces; expect more friction.

## Wireshark on the Mac

Capture inside the VM, copy the `.pcap` out, open it on the Mac:

```bash
sudo tcpdump -i veth-h1 -w /tmp/cap.pcap

multipass transfer netlab:/tmp/cap.pcap ./captures/cap.pcap
```

For colima:

```bash
colima ssh -- sudo cat /tmp/cap.pcap > ./captures/cap.pcap
```
