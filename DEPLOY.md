# Deployment: egress control for the `Bash` tool

This is the deploy-time half of the pod's security posture (DESIGN §11). The in-process
**key-injecting proxy** (DESIGN §7, `pod/key_proxy.py`) already removes provider keys from the
daemon and its `Bash` children — so a run can no longer *read* a key. What remains is a
**network** concern the pod process cannot enforce from Python: an LLM-authored `Bash` script
could still open its own socket to an arbitrary host and exfiltrate the **workspace data** it was
given. Blocking that is an egress allowlist applied to the container/host the pod runs in.

The pod does not implement this itself on purpose: a firewall enforced by the same process that runs
the untrusted code is not a real boundary. Enforce it one layer out, at the network.

## What to allow

Deny all outbound traffic by default, then allow exactly:

1. **The provider hosts the pod actually calls** — the upstreams the proxy forwards to:
   - `api.anthropic.com`
   - `api.openai.com`
   - `generativelanguage.googleapis.com`
   - `api.x.ai`
   - plus the host of any custom provider declared in `AGENT_CUSTOM_PROVIDERS`.
2. **Loopback** (`127.0.0.1`) — the daemon reaches the key proxy here; keep it host-local.
3. **DNS** to your resolver, if the allowlist is by hostname.

Everything else — outbound to arbitrary IPs/hosts — is denied. Model traffic keeps working because
it flows daemon → `127.0.0.1` proxy → allowlisted provider host; an exfiltration attempt to an
unlisted host is dropped.

Keep the proxy bound to `127.0.0.1` (`AGENT_POD_KEY_PROXY_HOST`, the default). It holds the real
keys and must never be reachable off-host, regardless of how the pod's own listener binds.

## Docker

Run the pod on an internal network with no default route to the internet, and reach providers
through an egress gateway/proxy that only forwards the allowlisted hosts:

```bash
docker network create --internal pod-internal
# The pod has no direct egress; an egress-proxy sidecar on both networks forwards only allowed hosts.
docker run --network pod-internal --name pod opencode-agent-pod
```

Or, with host firewalling, restrict the container's outbound set with `iptables`/`nftables` to the
provider IP ranges + loopback and drop the rest (owner-match the container's uid/netns).

## Kubernetes

A default-deny egress `NetworkPolicy`, opened only for DNS and the provider hosts (via an
allowlisting egress proxy or a CIDR set for the provider ranges):

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: pod-egress-allowlist
spec:
  podSelector:
    matchLabels: { app: opencode-agent-pod }
  policyTypes: ["Egress"]
  egress:
    - to: []                       # DNS
      ports: [{ protocol: UDP, port: 53 }, { protocol: TCP, port: 53 }]
    - to:                          # provider CIDRs, or route via an allowlisting egress gateway
        - ipBlock: { cidr: 0.0.0.0/0 }   # replace with the provider ranges / gateway address
      ports: [{ protocol: TCP, port: 443 }]
```

Plain hostname allowlists aren't expressible in vanilla `NetworkPolicy` — use a CNI that supports
FQDN policies (Cilium `toFQDNs`) or route all egress through an allowlisting forward proxy and deny
everything else.

## Verifying

With the policy applied, a `Bash` run that tries to reach an unlisted host (e.g.
`curl https://example.com`) should fail/time out, while a normal model run still completes. The
key-custody half is independently checked by `scripts/key_proxy_probe.py`.
