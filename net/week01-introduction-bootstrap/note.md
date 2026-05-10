# Exercises

1. Transmission delay: is measure of time required to push R bits into output link. Propagation delay is time required to transfer single bit through the link.

2. Packet switching is more effective when network is shared by users with non-linear usage. It doesn't reserve resources, so it may provide better performance when there're low amount of users. Circuit switching is good when we need to specific amount of resources, for example in telephone calls.

3. Where does *your* home router live in the ISP hierarchy - it lives at access network hierarchy.


# Self-Check

- Can you draw the OSI 7 layers vs the TCP/IP 5 layers, and say which the book uses:

OSI: App, Presentation, Session, Transport, Network, Link, Physical layers
TCP/IP: App, Transport, Network, Link, Physical layers

Book uses 5 layer TCP/IP stack.

- Can you give two reasons (one technical, one economic) why ISPs peer at IXPs?

Technical reason is that it improves stability (ISP is no longer point of failure), performance (less routing hopping).

Economic reason is that providers are no longer required to pay ISP.

- Did you actually `ping` between two namespaces? If not, do it before moving on