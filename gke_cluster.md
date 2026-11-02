# GKE Cluster & LLM-D Serving Refresher (`gke_cluster.md`)

**Snapshot Timestamp:** `2026-10-23T14:55:00Z`  
**Cluster Name:** `gemma-serving-cluster`  
**GCP Project:** `<YOUR_GCP_PROJECT_ID>`  
**Region:** `us-west1` (`us-west1-a`, `us-west1-b`)

---

## 1. Executive Summary

1. **Shared GPU Node (`g2-standard-8`, 1x NVIDIA L4 24GB) via GKE GPU Time-Sharing:**
   - The cluster runs a shared **`g2-standard-8`** node (`gk3-gemma-serving-cluste-nap-12ma5s3n-95509cb6-tr8j`) with **GKE Autopilot GPU Time-Sharing** (`cloud.google.com/gke-gpu-sharing-strategy: time-sharing`, `cloud.google.com/gke-max-shared-clients-per-gpu: "2"`).
   - Both **`gemma3-worker`** (`10.20.0.135`, `google/gemma-3-1b-it`) and **`qwen-worker`** (`10.20.0.134`, `Qwen/Qwen2.5-1.5B-Instruct`) are `1/1 Running` and co-located on this single physical node via cross-namespace `podAffinity`, each capped at `--gpu-memory-utilization=0.45`.
2. **Payload-Aware Multi-Model & Multi-Tenant Routing (`GCPRoutingExtension` + LLM-D IPP):**
   - The **Inference Payload Processor (IPP)** (`ghcr.io/llm-d/llm-d-inference-payload-processor:v0.1.0`) and `GCPRoutingExtension` (`payload-processor`) are deployed and healthy in `cluster-inference-gateway`.
   - Clients can send standard OpenAI requests (`POST https://10.10.0.9/v1/chat/completions`) without custom headers; IPP automatically extracts `"model"` into `X-Gateway-Model-Name` and resolves base models and tenant aliases (`gemma3-critical`, `qwen-critical`, `gemma3-batch`, `qwen-batch`, etc.) into `X-Gateway-Base-Model-Name`.
3. **Priority-Tiered Multi-Tenancy (`InferenceObjective` v1alpha2):**
   - Six `InferenceObjective` resources (`gemma3-critical` / `qwen-critical` at priority `100`, `gemma3-standard` / `qwen-standard` at `10`, and `gemma3-batch` / `qwen-batch` at `-10`) are reconciled by `gemma3-epp-router` and `qwen-epp-router`.

---

## 2. GKE Cluster Infrastructure Profile

| Property | Live Value |
| :--- | :--- |
| **Cluster Mode** | GKE Autopilot (`OPTIMIZE_UTILIZATION`) |
| **Release Channel & Version** | `REGULAR` — Kubernetes `v1.35.8-gke.1626001` |
| **VPC & Subnet** | `gemma-serving-vpc` / `gemma-serving-subnet-us-west1` |
| **Network Stack** | IPv4, GKE Dataplane V2 (Cilium/Antrea unified CNI) |
| **Cluster CIDRs** | Pods: `10.20.0.0/16` (`gke-pods`) \| Services: `10.30.0.0/20` (`gke-services`) |
| **Control Plane Endpoints** | Private: `10.10.0.2` (`enablePrivateNodes: true`) |
| **Workload Identity Pool** | `<YOUR_GCP_PROJECT_ID>.svc.id.goog` |
| **Active GPU Node** | `gk3-gemma-serving-cluste-nap-12ma5s3n-95509cb6-tr8j` (`g2-standard-8`, `nvidia-l4`, `time-sharing`, `max-shared-clients: 2`, `86%` CPU / `92%` RAM allocated) |

---

## 3. Live Inventory by Namespace

### 3.1 `cluster-inference-gateway` (Shared Ingress & Payload Processing Tier)
* **Gateway (`shared-inference-gateway`):**
  * **Class:** `gke-l7-rilb` (Regional Internal Application Load Balancer)
  * **VIP Address:** `10.10.0.9:443` (HTTPS listener terminating TLS with `Secret/gemma-gateway-tls-cert`)
  * **Status:** `Accepted: True`, `Programmed: True`, `GatewayHealthy: True`, `Attached Routes: 2`
* **LLM-D Inference Payload Processor (`Deployment/payload-processor`):**
  * **Status:** `1/1` Running (`10.20.0.211`, `ghcr.io/llm-d/llm-d-inference-payload-processor:v0.1.0`)
  * **Extensions:** `GCPRoutingExtension/payload-processor` (`Accepted: True`, `ResolvedRefs: True`, `FullDuplexStreamed`), `HealthCheckPolicy/payload-processor-healthcheck` (`Attached: True`, gRPC `:9005`)

### 3.2 `gemma-serving` (Gemma 3 1B Multi-Tenant Subsystem)
* **Model Worker (`Deployment/gemma3-worker`):**
  * **Status:** `1/1` Running (`pod/gemma3-worker-cbd5476cc-khmnq` at `10.20.0.135` on shared node `gk3-gemma-serving-cluste-nap-12ma5s3n-95509cb6-tr8j`)
  * **Served Models:** `google/gemma-3-1b-it`, `gemma3-critical`, `gemma3-standard`, `gemma3-batch`
* **LLM-D EPP Router (`Deployment/gemma3-epp-router`):**
  * **Status:** `1/1` Running (`10.20.0.200`, `registry.k8s.io/gateway-api-inference-extension/epp:v1.0.1`)
* **InferencePool & Multi-Tenant Objectives:**
  * `InferencePool/gemma3-inference-pool` + `HealthCheckPolicy/gemma3-pool-healthcheck` (`requestPath: /health`, `HEALTHY`)
  * `InferenceObjective`s: `gemma3-critical` (`priority: 100`), `gemma3-standard` (`priority: 10`), `gemma3-batch` (`priority: -10`)
  * `HTTPRoute/gemma3-http-route`: Injects `x-gateway-inference-objective` per tenant tier
* **Observability & Autoscaling:**
  * `PodMonitoring/gemma3-epp-monitoring` (`:9090/metrics`) & `PodMonitoring/gemma3-vllm-monitoring` (`:8000/metrics`)
  * `ScaledObject/gemma3-keda-scaler` & `HPA/keda-hpa-gemma3-worker`: Scales `1 ↔ 5` replicas (`0/5`, `0/16`)

### 3.3 `qwen-serving` (Qwen 2.5 1.5B Multi-Tenant Subsystem)
* **Model Worker (`Deployment/qwen-worker`):**
  * **Status:** `1/1` Running (`pod/qwen-worker-5b7bfc448-tzg49` at `10.20.0.134` on shared node `gk3-gemma-serving-cluste-nap-12ma5s3n-95509cb6-tr8j`)
  * **Served Models:** `Qwen/Qwen2.5-1.5B-Instruct`, `qwen-critical`, `qwen-standard`, `qwen-batch`
* **LLM-D EPP Router (`Deployment/qwen-epp-router`):**
  * **Status:** `1/1` Running (`10.20.0.201`, `registry.k8s.io/gateway-api-inference-extension/epp:v1.0.1`)
* **InferencePool & Multi-Tenant Objectives:**
  * `InferencePool/qwen-inference-pool` + `HealthCheckPolicy/qwen-pool-healthcheck` (`requestPath: /health`, `HEALTHY`)
  * `InferenceObjective`s: `qwen-critical` (`priority: 100`), `qwen-standard` (`priority: 10`), `qwen-batch` (`priority: -10`)
  * `HTTPRoute/qwen-http-route`: Injects `x-gateway-inference-objective` per tenant tier
* **Observability & Autoscaling:**
  * `PodMonitoring/qwen-epp-monitoring` (`:9090/metrics`) & `PodMonitoring/qwen-vllm-monitoring` (`:8000/metrics`)
  * `ScaledObject/qwen-keda-scaler` & `HPA/keda-hpa-qwen-worker`: Scales `1 ↔ 5` replicas (`0/5`, `0/16`)
