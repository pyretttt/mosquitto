FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        conntrack \
        curl \
        iperf3 \
        iproute2 \
        iptables \
        iputils-ping \
        jq \
        mtr-tiny \
        netcat-openbsd \
        nftables \
        procps \
        python3 \
        tcpdump \
        traceroute \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/cli-practice

CMD ["sleep", "infinity"]
