#!/bin/bash
# specacc.sh [outfile]: cumulative speculative-decoding acceptance from the Prometheus endpoint.
# Exact per-config acceptance, which the bench itself does not record.
O=${1:-/dev/stdout}
curl -s -m 20 http://127.0.0.1:8000/metrics > /tmp/specmetrics.txt
{
  grep -E "^vllm:spec_decode" /tmp/specmetrics.txt | grep -vE "_bucket|_created" | head -20
  python3 - <<'PY'
import re
t=open("/tmp/specmetrics.txt").read()
def g(pat):
    m=re.findall(r"^"+pat+r"\{[^}]*\}\s+([0-9.eE+-]+)$",t,re.M) or re.findall(r"^"+pat+r"\s+([0-9.eE+-]+)$",t,re.M)
    return sum(float(x) for x in m) if m else None
d=g("vllm:spec_decode_num_draft_tokens_total"); a=g("vllm:spec_decode_num_accepted_tokens_total")
e=g("vllm:spec_decode_num_drafts_total")
if d and a: print(f"  draft {d:.0f} accepted {a:.0f} acceptance {a/d:.3f}")
if e and a: print(f"  drafts {e:.0f} accepted/draft {a/e:.2f} tokens per step {(a+e)/e:.2f}")
PY
} | tee -a $O
