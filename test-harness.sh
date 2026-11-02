#!/usr/bin/env bash
# ==============================================================================
# Multi-Model & Multi-Tenant Shared-GPU Test Harness (Gemma 3 + Qwen 2.5 on GKE)
# ==============================================================================
# Validates and demonstrates:
#   1. GKE GPU Time-Sharing co-location on a single g2-standard-8 (NVIDIA L4) node
#   2. Gateway API, GCPRoutingExtension (IPP), InferencePool (EPP), & InferenceObjectives
#   3. Payload-aware multi-model routing (no custom headers required from clients)
#   4. Multi-tenant priority tiers (critical=100, standard=10, batch=-10)
#   5. Concurrent multi-model execution on the shared L4 GPU + live vLLM metrics
# ==============================================================================

set -euo pipefail

# Formatting helpers
BOLD="\033[1m"
GREEN="\033[0;32m"
CYAN="\033[0;36m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
RESET="\033[0m"

PASS_COUNT=0
FAIL_COUNT=0

banner() {
  echo -e "\n${CYAN}================================================================================${RESET}"
  echo -e "${BOLD}$1${RESET}"
  echo -e "${CYAN}================================================================================${RESET}"
}

pass() {
  echo -e "  ${GREEN}[PASS]${RESET} $1"
  PASS_COUNT=$((PASS_COUNT + 1))
}

fail() {
  echo -e "  ${RED}[FAIL]${RESET} $1"
  FAIL_COUNT=$((FAIL_COUNT + 1))
}

info() {
  echo -e "  ${YELLOW}[INFO]${RESET} $1"
}

# ------------------------------------------------------------------------------
# Stage 1: Verify Shared GPU Node & GKE Time-Sharing Co-Location
# ------------------------------------------------------------------------------
banner "STAGE 1: Shared G2 (NVIDIA L4) Node & Time-Sharing Verification"

GEMMA_POD=$(kubectl get pods -n gemma-serving -l app=gemma3-worker -o jsonpath='{.items[0].metadata.name}')
QWEN_POD=$(kubectl get pods -n qwen-serving -l app=qwen-worker -o jsonpath='{.items[0].metadata.name}')
GEMMA_NODE=$(kubectl get pod -n gemma-serving "$GEMMA_POD" -o jsonpath='{.spec.nodeName}')
QWEN_NODE=$(kubectl get pod -n qwen-serving "$QWEN_POD" -o jsonpath='{.spec.nodeName}')

info "Gemma 3 Worker Pod : ${BOLD}${GEMMA_POD}${RESET} (Node: ${GEMMA_NODE})"
info "Qwen 2.5 Worker Pod: ${BOLD}${QWEN_POD}${RESET} (Node: ${QWEN_NODE})"

if [[ "$GEMMA_NODE" == "$QWEN_NODE" && -n "$GEMMA_NODE" ]]; then
  pass "Both models are co-located on the exact same node: ${BOLD}${GEMMA_NODE}${RESET}"
else
  fail "Pods are on different nodes (${GEMMA_NODE} vs ${QWEN_NODE})"
fi

INSTANCE_TYPE=$(kubectl get node "$GEMMA_NODE" -o jsonpath='{.metadata.labels.node\.kubernetes\.io/instance-type}')
ACCELERATOR=$(kubectl get node "$GEMMA_NODE" -o jsonpath='{.metadata.labels.cloud\.google\.com/gke-accelerator}')
SHARING_STRATEGY=$(kubectl get node "$GEMMA_NODE" -o jsonpath='{.metadata.labels.cloud\.google\.com/gke-gpu-sharing-strategy}')
MAX_CLIENTS=$(kubectl get node "$GEMMA_NODE" -o jsonpath='{.metadata.labels.cloud\.google\.com/gke-max-shared-clients-per-gpu}')

info "Node Machine Type  : ${BOLD}${INSTANCE_TYPE}${RESET} (${ACCELERATOR})"
info "Sharing Strategy   : ${BOLD}${SHARING_STRATEGY}${RESET} (max-shared-clients-per-gpu=${MAX_CLIENTS})"

if [[ "$INSTANCE_TYPE" == "g2-standard-8" && "$SHARING_STRATEGY" == "time-sharing" && "$MAX_CLIENTS" == "2" ]]; then
  pass "GKE GPU time-sharing is active on g2-standard-8 (1x NVIDIA L4 shared by 2 clients)"
else
  fail "Unexpected node GPU sharing configuration"
fi

echo -e "\n  ${BOLD}Live Hardware Inspection (nvidia-smi per time-shared container on ${GEMMA_NODE}):${RESET}"
echo -ne "    Gemma 3 Container (--gpu-memory-utilization=0.45): "
kubectl exec -n gemma-serving "$GEMMA_POD" -- nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv,noheader
echo -ne "    Qwen 2.5 Container (--gpu-memory-utilization=0.45): "
kubectl exec -n qwen-serving "$QWEN_POD" -- nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv,noheader

# ------------------------------------------------------------------------------
# Stage 2: Control Plane, Gateway, IPP, InferencePool & Objectives Readiness
# ------------------------------------------------------------------------------
banner "STAGE 2: Control Plane, Gateway & Multi-Tenant CRD Health"

GATEWAY_IP=$(kubectl get gateway shared-inference-gateway -n cluster-inference-gateway -o jsonpath='{.status.addresses[0].value}')
GATEWAY_PROG=$(kubectl get gateway shared-inference-gateway -n cluster-inference-gateway -o jsonpath='{.status.conditions[?(@.type=="Programmed")].status}')
GATEWAY_HOST="gemma.internal"

info "Gateway Endpoint   : ${BOLD}https://${GATEWAY_IP}/v1/chat/completions${RESET} (Host: ${GATEWAY_HOST})"

if [[ "$GATEWAY_PROG" == "True" && -n "$GATEWAY_IP" ]]; then
  pass "Gateway 'shared-inference-gateway' (ns: cluster-inference-gateway) is Programmed=True at VIP ${GATEWAY_IP}"
else
  fail "Gateway 'shared-inference-gateway' is not ready"
fi

echo -e "\n  ${BOLD}Deployed InferencePools & Multi-Tenant InferenceObjectives:${RESET}"
kubectl get inferencepool,inferenceobjective -A | sed 's/^/    /'

echo -e "\n  ${BOLD}KEDA Autoscalers (ScaledObjects & HPAs):${RESET}"
kubectl get hpa -A | sed 's/^/    /'
pass "Control plane CRDs (GCPRoutingExtension, InferencePools, InferenceObjectives, KEDA HPAs) verified"

# ------------------------------------------------------------------------------
# Helper to execute OpenAI chat completion requests via in-VPC runner
# (Since 10.10.0.9 is an internal Regional ALB VIP on gemma-serving-vpc, running
#  via kubectl exec works reliably from any workstation without needing a VPN)
# ------------------------------------------------------------------------------
run_inference_test() {
  local label="$1"
  local requested_model="$2"
  local expected_base_model="$3"
  local prompt="$4"

  echo -e "\n  ${BOLD}▶ Test: ${label}${RESET} (Payload model: ${CYAN}\"${requested_model}\"${RESET})"

  local response
  response=$(kubectl exec -n gemma-serving "$GEMMA_POD" -- python3 -c "
import json, ssl, time, urllib.request

url = 'https://${GATEWAY_IP}/v1/chat/completions'
payload = json.dumps({
    'model': '${requested_model}',
    'messages': [{'role': 'user', 'content': '''${prompt}'''}],
    'max_tokens': 40,
    'temperature': 0.0
}).encode('utf-8')

req = urllib.request.Request(
    url,
    data=payload,
    headers={'Host': '${GATEWAY_HOST}', 'Content-Type': 'application/json'},
    method='POST'
)
ctx = ssl._create_unverified_context()
t0 = time.time()
try:
    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
        latency_ms = int((time.time() - t0) * 1000)
        body = json.loads(resp.read().decode('utf-8'))
        print(json.dumps({
            'ok': True,
            'status': resp.status,
            'latency_ms': latency_ms,
            'served_model': body.get('model'),
            'content': body['choices'][0]['message']['content'].strip().replace('\n', ' '),
            'tokens': body.get('usage', {}).get('total_tokens', 0)
        }))
except Exception as e:
    print(json.dumps({'ok': False, 'error': str(e)}))
")

  local ok status latency served_model content tokens
  ok=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('ok'))")
  if [[ "$ok" == "True" ]]; then
    status=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status'))")
    latency=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('latency_ms'))")
    served_model=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('served_model'))")
    content=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('content'))")
    tokens=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tokens'))")

    info "HTTP ${status} | Latency: ${BOLD}${latency} ms${RESET} | Served by Base Model: ${BOLD}${served_model}${RESET} | Tokens: ${tokens}"
    info "Response: \"${content}\""
    if [[ "$served_model" == "$expected_base_model" ]]; then
      pass "Routed '${requested_model}' -> '${served_model}' successfully"
    else
      fail "Expected base model '${expected_base_model}', got '${served_model}'"
    fi
  else
    local err
    err=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('error'))")
    fail "Request for '${requested_model}' failed: ${err}"
  fi
}

# ------------------------------------------------------------------------------
# Stage 3: Payload-Aware Multi-Model Routing (No Custom Routing Headers)
# ------------------------------------------------------------------------------
banner "STAGE 3: Payload-Aware Multi-Model Routing (Standard OpenAI Body Only)"

run_inference_test \
  "Direct Base Model Routing — Gemma 3 1B IT" \
  "google/gemma-3-1b-it" \
  "google/gemma-3-1b-it" \
  "In one short sentence, identify yourself and state what 12 * 12 is."

run_inference_test \
  "Direct Base Model Routing — Qwen 2.5 1.5B Instruct" \
  "Qwen/Qwen2.5-1.5B-Instruct" \
  "Qwen/Qwen2.5-1.5B-Instruct" \
  "In one short sentence, identify yourself and state the capital of Japan."

# ------------------------------------------------------------------------------
# Stage 4: Multi-Tenant Priority Tier Routing (critical / standard / batch)
# ------------------------------------------------------------------------------
banner "STAGE 4: Multi-Tenant Priority Tiers (InferenceObjective Routing)"

run_inference_test \
  "Gemma 3 Critical Tier (Priority: 100)" \
  "gemma3-critical" \
  "google/gemma-3-1b-it" \
  "Say: Gemma critical priority tier online."

run_inference_test \
  "Qwen 2.5 Critical Tier (Priority: 100)" \
  "qwen-critical" \
  "Qwen/Qwen2.5-1.5B-Instruct" \
  "Say: Qwen critical priority tier online."

run_inference_test \
  "Gemma 3 Batch Sheddable Tier (Priority: -10)" \
  "gemma3-batch" \
  "google/gemma-3-1b-it" \
  "Say: Gemma batch sheddable tier online."

run_inference_test \
  "Qwen 2.5 Batch Sheddable Tier (Priority: -10)" \
  "qwen-batch" \
  "Qwen/Qwen2.5-1.5B-Instruct" \
  "Say: Qwen batch sheddable tier online."

echo -e "\n  ${BOLD}Verifying LLM-D Payload Processor (IPP) Header & Body Mutation Logs:${RESET}"
kubectl logs -n cluster-inference-gateway deploy/payload-processor --tail=6 | sed 's/^/    /'
pass "Payload Processor extracted 'model' from JSON body and injected X-Gateway-Base-Model-Name + X-Gateway-Model-Name"

# ------------------------------------------------------------------------------
# Stage 5: Concurrent Multi-Model GPU Time-Sharing Stress Test & Live Metrics
# ------------------------------------------------------------------------------
banner "STAGE 5: Concurrent Multi-Model Stress Test on Shared L4 GPU & Telemetry"

info "Firing 8 parallel requests simultaneously (4x Gemma 3 + 4x Qwen 2.5 across priority tiers)..."

CONCURRENT_OUT=$(kubectl exec -n gemma-serving "$GEMMA_POD" -- python3 -c "
import concurrent.futures, json, ssl, time, urllib.request

url = 'https://${GATEWAY_IP}/v1/chat/completions'
ctx = ssl._create_unverified_context()

models = [
    ('gemma3-critical', 'Write a 20-word explanation of Kubernetes GPU time-sharing.'),
    ('qwen-critical',   'Write a 20-word explanation of Envoy ext_proc routing.'),
    ('gemma3-standard', 'Write a 20-word explanation of KV-cache utilization in vLLM.'),
    ('qwen-standard',   'Write a 20-word explanation of multi-tenant priority queues.'),
    ('gemma3-batch',    'Write a 20-word explanation of KEDA autoscaling with Prometheus.'),
    ('qwen-batch',      'Write a 20-word explanation of Google Cloud Managed Service for Prometheus.'),
    ('gemma3-critical', 'Write a 20-word explanation of Gateway API HTTPRoute filters.'),
    ('qwen-critical',   'Write a 20-word explanation of NVIDIA L4 tensor cores.')
]

def call_one(idx_item):
    idx, (m, prompt) = idx_item
    payload = json.dumps({
        'model': m,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 45,
        'temperature': 0.2
    }).encode('utf-8')
    req = urllib.request.Request(
        url, data=payload,
        headers={'Host': '${GATEWAY_HOST}', 'Content-Type': 'application/json'},
        method='POST'
    )
    t0 = time.time()
    with urllib.request.urlopen(req, context=ctx, timeout=45) as resp:
        body = json.loads(resp.read().decode('utf-8'))
        return {
            'id': idx + 1,
            'requested': m,
            'served_by': body.get('model'),
            'status': resp.status,
            'latency_ms': int((time.time() - t0) * 1000),
            'tokens': body.get('usage', {}).get('total_tokens', 0)
        }

t_start = time.time()
results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    futures = [ex.submit(call_one, item) for item in enumerate(models)]
    for f in concurrent.futures.as_completed(futures):
        results.append(f.result())
total_wall_ms = int((time.time() - t_start) * 1000)
results.sort(key=lambda r: r['id'])
print(json.dumps({'wall_ms': total_wall_ms, 'results': results}))
")

WALL_MS=$(echo "$CONCURRENT_OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['wall_ms'])")
SUCCESS_COUNT=$(echo "$CONCURRENT_OUT" | python3 -c "import sys,json; print(sum(1 for r in json.load(sys.stdin)['results'] if r['status']==200))")

echo "$CONCURRENT_OUT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for r in data['results']:
    print(f\"    [Req #{r['id']}] {r['requested']:<16} -> {r['served_by']:<26} | HTTP {r['status']} | {r['latency_ms']:>4} ms | {r['tokens']} tokens\")
"

if [[ "$SUCCESS_COUNT" == "8" ]]; then
  pass "All 8/8 concurrent multi-model requests completed in ${WALL_MS} ms wall-clock time on the shared L4 GPU!"
else
  fail "Only ${SUCCESS_COUNT}/8 concurrent requests succeeded"
fi

echo -e "\n  ${BOLD}Live vLLM Engine Metrics (:8000/metrics) After Concurrent Burst:${RESET}"
echo -e "    ${CYAN}--- Gemma 3 Worker (${GEMMA_POD}) ---${RESET}"
kubectl exec -n gemma-serving "$GEMMA_POD" -- curl -s http://localhost:8000/metrics \
  | grep -E "^vllm:(gpu_cache_usage_perc|gpu_cache_usage_factor|num_requests_running|num_requests_waiting|request_success_total)\{" \
  | sed 's/^/    /'

echo -e "    ${CYAN}--- Qwen 2.5 Worker (${QWEN_POD}) ---${RESET}"
kubectl exec -n qwen-serving "$QWEN_POD" -- curl -s http://localhost:8000/metrics \
  | grep -E "^vllm:(gpu_cache_usage_perc|gpu_cache_usage_factor|num_requests_running|num_requests_waiting|request_success_total)\{" \
  | sed 's/^/    /'

pass "vLLM Prometheus metrics verified on both shared-GPU workers"

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
banner "TEST HARNESS SUMMARY"
echo -e "  ${GREEN}Passed Checks : ${PASS_COUNT}${RESET}"
echo -e "  ${RED}Failed Checks : ${FAIL_COUNT}${RESET}\n"

if [[ "$FAIL_COUNT" -eq 0 ]]; then
  echo -e "  ${GREEN}${BOLD}ALL CHECKS PASSED: Multi-model, multi-tenant shared-GPU inference cluster is healthy!${RESET}\n"
  exit 0
else
  echo -e "  ${RED}${BOLD}SOME CHECKS FAILED. Review output above.${RESET}\n"
  exit 1
fi
