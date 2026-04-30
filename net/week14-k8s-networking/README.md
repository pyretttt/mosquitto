# Week 14 - Kubernetes networking

The single most useful week for a DevOps. By the end you should be able to draw the packet flow from a curl inside a pod to another pod's container, and reason about *every* failure mode along the way.

## Reading

- Kubernetes networking model: https://kubernetes.io/docs/concepts/services-networking/
- "A Visual Guide to Kubernetes Networking Fundamentals" (LWN): https://lwn.net/Articles/831087/
- Learnk8s "Kubernetes networking model" series: https://learnk8s.io/kubernetes-network-packets
- Cilium docs - read these in order:
  - https://docs.cilium.io/en/stable/concepts/overview/
  - https://docs.cilium.io/en/stable/concepts/networking/ipam/
  - https://docs.cilium.io/en/stable/network/concepts/routing/
  - https://docs.cilium.io/en/stable/network/kubernetes/concepts/
- (optional but excellent) "Cilium - eBPF-based Networking, Observability, and Security" KubeCon talks on YouTube.

## The 4 promises of the k8s networking model

1. Every pod gets its own IP address.
2. Pods on the same node communicate without NAT.
3. Pods across nodes communicate without NAT.
4. The IP a pod sees on itself is the IP others see it as.

## Learning goals

- Explain how a pod gets its IP (CNI ADD/DEL flow, IPAM).
- Walk through pod-to-pod traffic on the same node (veth -> bridge / eBPF) and across nodes (overlay vs native routing).
- Walk through pod-to-Service traffic: DNS lookup -> ClusterIP -> kube-proxy iptables/IPVS or Cilium eBPF -> EndpointSlice -> destination pod IP -> conntrack on return.
- Read a NetworkPolicy and predict its effect.
- Reason about Ingress vs Gateway API vs LoadBalancer.

## Lab

**Goal:** stand up a kind cluster with Cilium and trace a real request end-to-end.

1. Bring up the cluster:
   ```bash
   cd lab/kind
   kind create cluster --config kind-config.yaml --name net-lab
   bash cilium-install.sh
   cilium status --wait
   ```
2. Apply the demo app:
   ```bash
   kubectl apply -f manifests/01-frontend-backend.yaml
   kubectl get pods -o wide
   ```
3. From inside the frontend pod, hit the backend Service:
   ```bash
   FRONTEND=$(kubectl get pod -l app=frontend -o jsonpath='{.items[0].metadata.name}')
   kubectl exec -it $FRONTEND -- curl -v http://backend/
   kubectl exec -it $FRONTEND -- dig backend.default.svc.cluster.local
   ```
4. Trace the packet path. While `curl` runs in a loop, on each node:
   ```bash
   docker exec -it net-lab-worker bash
   # inside the node:
   ip a
   ip route
   nft list ruleset | head
   conntrack -L | grep <frontend pod IP>
   ```
   Use Cilium tools:
   ```bash
   cilium monitor --type drop
   cilium hubble observe --follow
   ```
5. Apply NetworkPolicy:
   ```bash
   kubectl apply -f manifests/02-network-policy.yaml
   ```
   Confirm: traffic from frontend -> backend still works; from a third pod, blocked.
6. Break and observe:
   - `kubectl delete endpointslice -l kubernetes.io/service-name=backend` - what happens? CoreDNS? Service?
   - Stop CoreDNS: `kubectl scale deploy coredns -n kube-system --replicas=0`. Try a DNS-based curl. Now try `curl http://<clusterIP>/` directly.
   - On a node, `iptables -F` (or skip - Cilium isn't using iptables for service routing). Compare to a non-Cilium cluster.

## Exercises

1. **Conceptual.** In `notes.md`, write a 1-page document titled "Life of a packet: `curl http://backend.default.svc.cluster.local/` from inside the frontend pod". Include:
   - DNS resolution (which CoreDNS pod, what NodeLocal DNS would change).
   - ClusterIP lookup -> EndpointSlice -> pod IP selection (kube-proxy iptables vs IPVS vs Cilium eBPF).
   - veth out of the pod, into the host netns.
   - Cross-node hop: native routing (Cilium with `routing-mode=native`) vs VXLAN overlay.
   - Conntrack entry on the return path.
   - Diagram with mermaid.
2. **Practical.** Reproduce 3 failure scenarios from the lab. Document what kubectl/cilium output each one produces. Save as `exercises/14-failures.md`.
3. **Practical - NetworkPolicy.** Apply default-deny + explicit allow. Test with a "third party" pod (`kubectl run debug --image=nicolaka/netshoot --rm -it --restart=Never -- curl http://backend/`). Show it gets blocked. Use `hubble observe --verdict DROPPED`.
4. **Stretch - Gateway API.** Replace ingress-nginx with Cilium Gateway API. Compare configurations.

## Self-check

- Without notes, draw pod-to-pod traffic across two nodes. Where does the encapsulation happen (if any)? Where does conntrack live?
- What's a `Service`'s ClusterIP technically? (Hint: it's not assigned to any interface.)
- What's the difference between `iptables` mode and `IPVS` mode kube-proxy?
- What does Cilium do *differently* from kube-proxy? Why is it faster?
- What does a NetworkPolicy with `policyTypes: [Egress]` and an empty `egress:` mean?

## Useful one-liners

```bash
kubectl get pods -A -o wide                             # IPs everywhere
kubectl get svc -A
kubectl get endpointslices -A
kubectl get networkpolicy -A

cilium status
cilium connectivity test
cilium monitor --type drop
cilium hubble observe --follow
cilium hubble observe --verdict DROPPED

kubectl debug node/<node> -it --image=nicolaka/netshoot
kubectl run net --rm -it --image=nicolaka/netshoot -- bash
```
