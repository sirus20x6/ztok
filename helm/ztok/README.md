# ztok Helm chart

Production-ready Helm chart for deploying [`ztok serve`](../../README.md) — a fast HTTP tokenizer server — on Kubernetes 1.23+.

* App version: **1.22.0** (overridable via `image.tag`)
* Chart version: **0.1.0**
* API: `apps/v1`, `autoscaling/v2`, `policy/v1`, `networking.k8s.io/v1`, `monitoring.coreos.com/v1` (optional)

## TL;DR

```bash
# 1. Put your vocab / model file into a ConfigMap (small models) or PVC.
kubectl create configmap llama2-vocab \
  --from-file=vocab=/path/to/llama2.model

# 2. Create the bearer-token secret out-of-band.
kubectl create secret generic ztok-auth \
  --from-literal=token="$(openssl rand -hex 32)"

# 3. Install.
helm install ztok ./helm/ztok \
  --set vocab.configMap.enabled=true \
  --set vocab.configMap.name=llama2-vocab \
  --set vocab.path=/vocabs/llama2.model

# 4. Smoke-test.
helm test ztok
```

## Installing

```bash
helm install <release> ./helm/ztok [--values my-values.yaml] [--set key=value ...]
```

## Uninstalling

```bash
helm uninstall <release>
```

PVCs (the optional persistent prefix cache) are NOT garbage-collected by `helm uninstall` — that's standard Helm behavior. Delete them by hand if you want the storage back:

```bash
kubectl delete pvc <release>-ztok-cache
```

## Vocab / model file

`ztok serve --model PATH` needs a vocab or model file inside the container. The chart supports three mount modes:

1. **ConfigMap** (`vocab.configMap.enabled=true`). Suitable for files **up to ~1 MiB** (the etcd object size cap). Create the ConfigMap yourself — the chart doesn't template the file in because it's usually binary (SentencePiece protobuf, raw tiktoken BPE) and ill-suited to `helm install --set-file`.

   ```bash
   kubectl create configmap my-vocab --from-file=vocab=tokenizer.model
   ```

   ```yaml
   vocab:
     path: /vocabs/tokenizer.model
     configMap:
       enabled: true
       name: my-vocab
       key: vocab
   ```

2. **PVC** (`vocab.pvc.enabled=true`). For models that exceed the ConfigMap limit. Pre-populate the PVC with a Job, init container, or `kubectl cp`.

   ```yaml
   vocab:
     path: /vocabs/llama2.model
     pvc:
       enabled: true
       claimName: llama2-models
       subPath: llama2.model
       readOnly: true
   ```

3. **Baked into the image**. If your CI bakes the vocab into a custom image at `vocab.path`, leave both `configMap.enabled` and `pvc.enabled` false. Nothing extra is mounted.

## Auth

Bearer-token auth is enabled by default (`auth.enabled=true`). `/health` is exempt. Two patterns:

* **Bring your own Secret** (recommended). Create a `Secret` with key `token`, set `auth.tokenSecretName` to its name, leave `auth.create=false`.
* **Chart-managed Secret** (`auth.create=true`). Sets the token from `auth.tokenValue`. Convenient but ties the token to the values file — combine with sealed-secrets / Vault for production.

Disable auth entirely with `auth.enabled=false` (e.g. inside a fully internal mesh where mTLS is the boundary).

## Rate limit

Per-client-IP token-bucket via `--rate-limit REQ_PER_SEC`. `/health` is exempt. Defaults to 100 req/s/IP; tune `rateLimit.reqPerSec` for your workload.

## Probes

| Probe | Path | Default |
|---|---|---|
| Startup | `/health` | 5 s × 60 tries = 5 min for vocab load |
| Readiness | `/health` | period 10 s, failure 3 |
| Liveness | `/health` | period 30 s, failure 3 |

Bump `probes.startup.failureThreshold` for vocab/model files in the hundreds of MiB.

## Security

The chart ships with a hardened pod and container security context:

* `runAsNonRoot: true`, uid/gid `65532` (matches the `nonroot` user in distroless base images)
* `readOnlyRootFilesystem: true` — `/tmp` is writable via a 64 MiB `emptyDir`
* `allowPrivilegeEscalation: false`
* `capabilities.drop: [ALL]`
* `seccompProfile: RuntimeDefault` at the pod level

These line up with the Kubernetes `restricted` Pod Security Standard.

## Autoscaling, PDB, ServiceMonitor, Ingress

All optional. Toggle their `.enabled` flag in values.yaml. Defaults install zero optional objects so the chart works in a vanilla cluster.

## Persistent prefix cache

`persistentCache.enabled=true` provisions a PVC and mounts it at `persistentCache.mountPath`. Forward-looking: today's `ztok serve` does not expose a `--prefix-cache-dir` flag; when it does, no chart changes will be needed. Until then, this block is harmless to enable but doesn't change runtime behavior.

## Values reference

See [`values.yaml`](./values.yaml). Highlights:

| Key | Default | What |
|---|---|---|
| `image.repository` / `image.tag` | placeholder / `1.22.0` | image ref |
| `replicaCount` | `3` | replicas when HPA disabled |
| `service.port` | `7890` | matches `ztok serve` default port |
| `vocab.path` | `/vocabs/llama2.model` | `--model` argument |
| `auth.enabled` | `true` | bearer-token auth |
| `rateLimit.reqPerSec` | `100` | per-client-IP RPS |
| `resources.requests` | `500m CPU / 256 Mi` | tune to your vocab |
| `autoscaling.enabled` | `false` | HPA v2 |
| `podDisruptionBudget.enabled` | `false` | recommended `true` in prod |
| `prometheus.serviceMonitor.enabled` | `false` | needs the Operator CRD |

## Testing

```bash
helm lint ./helm/ztok
helm template ztok ./helm/ztok > /tmp/rendered.yaml
helm install --dry-run --debug ztok ./helm/ztok
```

Bundled connectivity test:

```bash
helm test <release> -n <namespace>
```

The test pod GETs `/health` over the in-cluster Service and asserts a `{"ok":true}` response.
