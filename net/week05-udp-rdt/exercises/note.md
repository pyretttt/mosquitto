## Exercises

1. **Conceptual.** In `notes.md`:
   - Why is UDP useful when TCP exists? Give 3 real-world examples (DNS, QUIC, RTP, gaming, ...).

It's useful when latency is dominant over reliability, for example streaming, dns queries (can be cached hard) and etc.
Games use UDP, because low latency is over reliability, video streaming is the same. QUIC uses udp, but provides reliable data transfer.

   - In Go-Back-N with sequence numbers in [0..N-1], what's the maximum window size? Why?

Maximum window size is `N - 1`, it's not possible to make it `N` because retransmission then could be though as next cycle. That unused sequence number acts as a “gap” between the current window and the next one after wraparound. So the receiver never has to guess whether a packet with sequence number 0 is old or new

   - In Selective Repeat, why must the window be at most N/2?

If W > N/2, then after sequence number wraparound, an old packet can fall into the receiver’s current receive window, making it impossible to determine whether it’s a retransmission or a new transmission.

Sequence numbers:
```
0 1 2 3 4 5 6 7
```

Sender sends

```
0 1 2 3 4
```

Receiver window is now

```
0 1 2 3 4
```

Now imagine all ACKs are lost. The sender times out and retransmits packet 0. The receiver now asks - Is this:
- the old packet 0 being retransmitted because its ACK was lost?
- or the new packet 0 after sequence numbers wrapped around?

It must be at most N/2

2. **Practical - implement Stop-and-Wait.** In `exercises/05-rdt-saw.py` build a sender + receiver over UDP that:
   - Numbers messages with a 1-bit sequence number.
   - Receiver ACKs each correctly-numbered message.
   - Sender waits for ACK with a timeout (200ms is fine), retransmits on timeout.
   - Survives 10% loss (use `tc netem loss 10%`).
3. **Practical - implement Selective Repeat.** In `exercises/05-rdt-sr.py`, build the SR variant with window size 4 and 3-bit sequence numbers. Buffer out-of-order packets at the receiver. Cumulative + selective ACKs.
4. **Stretch.** Add `tc netem reorder 25%` + `delay 50ms` and see what your protocols do. Where do they break?
