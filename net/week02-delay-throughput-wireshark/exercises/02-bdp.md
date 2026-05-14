Connecting to host speed-vld.vtt.net, port 5201
[  5] local 10.10.0.10 port 56934 connected to 91.204.97.36 port 5201
[ ID] Interval           Transfer     Bitrate         Retr  Cwnd
[  5]   0.00-1.01   sec  20.9 MBytes   174 Mbits/sec    5    102 KBytes
[  5]   1.01-2.01   sec  16.5 MBytes   138 Mbits/sec    2    120 KBytes
[  5]   2.01-3.01   sec  9.62 MBytes  80.7 Mbits/sec    1    146 KBytes
[  5]   3.01-4.01   sec  12.5 MBytes   105 Mbits/sec    0    185 KBytes
[  5]   4.01-5.01   sec  12.5 MBytes   105 Mbits/sec    2    106 KBytes
[  5]   5.01-6.01   sec  12.5 MBytes   105 Mbits/sec    1    126 KBytes
[  5]   6.01-7.01   sec  13.9 MBytes   116 Mbits/sec    4    109 KBytes
[  5]   7.01-8.00   sec  12.4 MBytes   104 Mbits/sec    1    117 KBytes
[  5]   8.00-9.00   sec  15.4 MBytes   129 Mbits/sec    0    171 KBytes
[  5]   9.00-10.00  sec  15.6 MBytes   131 Mbits/sec    0    218 KBytes
- - - - - - - - - - - - - - - - - - - - - - - - -
[ ID] Interval           Transfer     Bitrate         Retr
[  5]   0.00-10.00  sec   142 MBytes   119 Mbits/sec   16            sender
[  5]   0.00-10.05  sec   136 MBytes   114 Mbits/sec                  receiver

To compute BDP we need RTT which can be obtained from ping

# ping -c 5 91.204.97.36
PING 91.204.97.36 (91.204.97.36) 56(84) bytes of data.
64 bytes from 91.204.97.36: icmp_seq=1 ttl=63 time=1.91 ms
64 bytes from 91.204.97.36: icmp_seq=2 ttl=63 time=1.52 ms
64 bytes from 91.204.97.36: icmp_seq=3 ttl=63 time=2.09 ms
64 bytes from 91.204.97.36: icmp_seq=4 ttl=63 time=1.60 ms
64 bytes from 91.204.97.36: icmp_seq=5 ttl=63 time=1.89 ms

--- 91.204.97.36 ping statistics ---
5 packets transmitted, 5 received, 0% packet loss, time 4014ms
rtt min/avg/max/mdev = 1.522/1.803/2.093/0.211 ms

Let's take average 1.803ms and 114 Mbits/sec

then BDP is
1,14e+8 * 0.001803 = 205542 bits

in kbytes it is 25.69 kbytes

The TCP congestion window is much larger than BDP, so throughput is not window-limited. Throughput (~114 Mbps) is not limited by RTT or TCP window (BDP is small), and is likely constrained by path bottlenecks such as link capacity, congestion, or packet loss