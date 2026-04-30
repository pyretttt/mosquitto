# Kubernetes networking cheatsheet (`kubectl`, `cilium`, `hubble`, `ksniff`)

## Quick checks

```bash
kubectl get pods -A -o wide                            # IPs, nodes
kubectl get svc -A
kubectl get endpointslices -A
kubectl get networkpolicy -A
kubectl get ingress -A
kubectl get gateway -A                                  # Gateway API

kubectl describe svc <name>
kubectl describe endpointslice <name>
kubectl describe networkpolicy <name>
```

## Run a debug pod

```bash
kubectl run net --rm -it --restart=Never --image=nicolaka/netshoot -- bash

kubectl debug -it <pod> --image=nicolaka/netshoot --target=<container>     # ephemeral container

kubectl debug node/<node> -it --image=nicolaka/netshoot                     # full node netns
```

Inside the netshoot pod:

```bash
dig backend.default.svc.cluster.local
nslookup backend
curl -v http://backend/
ss -tn
ip a
ip route
nft list ruleset | head
```

## Trace DNS in cluster

```bash
kubectl exec -it <pod> -- cat /etc/resolv.conf
kubectl exec -it <pod> -- dig kubernetes.default.svc.cluster.local
kubectl logs -n kube-system -l k8s-app=kube-dns --tail=50

kubectl run dns --rm -it --image=nicolaka/netshoot -- \
  dig +trace +short backend.default.svc.cluster.local
```

`/etc/resolv.conf` in a pod typically:

```
search default.svc.cluster.local svc.cluster.local cluster.local
nameserver 10.96.0.10
options ndots:5
```

`ndots:5` means "anything with fewer than 5 dots is searched first against the local search list" - that's why `kubernetes` resolves but `kubernetes.default.svc.cluster.local` works too.

## kube-proxy modes

```bash
kubectl get cm -n kube-system kube-proxy -o yaml | grep mode
ipvsadm -L -n                                # if IPVS mode
iptables -t nat -L KUBE-SERVICES -n           # if iptables mode (lots of chains)
```

## Cilium

```bash
cilium status
cilium status --verbose
cilium connectivity test                                # full functional test
cilium config view
cilium endpoint list

cilium hubble observe --follow                          # live flow log
cilium hubble observe --pod default/web --follow
cilium hubble observe --verdict DROPPED
cilium hubble observe --type policy-verdict
cilium hubble observe --to-fqdn 'github.com'

cilium monitor --type drop                              # raw kernel drops
cilium policy get
```

Inside a Cilium agent pod (`-n kube-system`):

```bash
cilium-dbg endpoint list
cilium-dbg map list
cilium-dbg bpf lb list                                  # service load-balancing entries
cilium-dbg bpf ct list global                           # eBPF conntrack
```

## NetworkPolicy quick patterns

Default-deny everything in a namespace:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: { name: default-deny }
spec:
  podSelector: {}
  policyTypes: [Ingress, Egress]
```

Allow DNS from any pod:

```yaml
egress:
  - to:
      - namespaceSelector:
          matchLabels: { kubernetes.io/metadata.name: kube-system }
        podSelector:
          matchLabels: { k8s-app: kube-dns }
    ports:
      - { port: 53, protocol: UDP }
      - { port: 53, protocol: TCP }
```

Allow a specific app to call another by label:

```yaml
ingress:
  - from:
      - podSelector:
          matchLabels: { app: web }
    ports:
      - { port: 8080, protocol: TCP }
```

## Capture pod traffic

If `ksniff` plugin is installed:

```bash
kubectl sniff <pod> -p -o capture.pcap
```

Otherwise, attach a netshoot ephemeral container to the same network namespace:

```bash
kubectl debug -it <pod> --image=nicolaka/netshoot --target=<container>
# inside:
tcpdump -ni eth0 -w /tmp/cap.pcap
```

Or, on the node:

```bash
kubectl debug node/<node> -it --image=nicolaka/netshoot
# the node's host netns is mounted at /host
chroot /host
tcpdump -ni any -w /tmp/cap.pcap
```

## Service troubleshooting recipe

When "the service is broken":

1. Pod ready? `kubectl get pods -l app=foo`
2. Endpoints populated? `kubectl get endpointslice -l kubernetes.io/service-name=foo`
3. From a debug pod, `dig foo` -> resolves to the ClusterIP?
4. From a debug pod, `curl http://<podIP>:<containerPort>/` works directly? -> rules out service routing.
5. From a debug pod, `curl http://<clusterIP>:<port>/`? -> tests kube-proxy/Cilium.
6. From a debug pod, `curl http://foo:<servicePort>/`? -> tests DNS + service routing.
7. Cross-node? Check `cilium hubble observe` for drops on the destination node.

If step 3 fails: NetworkPolicy / DNS issue.
If step 4 fails: app issue, not network.
If step 5 fails: kube-proxy/Cilium service routing broken.
If only step 6 fails: DNS or `ndots`/search-domain quirk.

## Ingress / Gateway API

```bash
kubectl get ingressclass
kubectl get ingress -A
kubectl describe ingress <name>

kubectl get gatewayclass
kubectl get gateway -A
kubectl get httproute -A

kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx --tail=200
```

## kubectl-trace for L4/L7 deep dives (advanced)

```bash
kubectl trace run node/<node> --program-text 'tracepoint:tcp:tcp_send_reset { printf("%d -> %d\n", args->skaddr, args->dport); }'
```
