#!/bin/bash
# glm_run.sh <label> [knob=val ...]  (root on Reddie): one GLM TP4 experiment = boot + screen + table row.
LBL=${1:?label}; shift
D=/var/tmp/boot-results/glm53
for kv in "$@"; do export "$kv"; done
# every boot tonight skips the 1M-token dummy-video warmup and matches DeepSeek's mm config
export VLLM_EXTRA="${VLLM_EXTRA:---limit-mm-per-prompt {\"image\":4,\"video\":0}}"
echo "$(date -u +%T) BOOT $LBL ($*)" >> $D/status.txt
bash /root/glm_boot.sh "$LBL"; rc=$?
if [ $rc != 0 ]; then echo "$(date -u +%T) BOOT-FAILED $LBL rc=$rc" >> $D/status.txt; exit $rc; fi
echo "$(date -u +%T) SERVING $LBL" >> $D/status.txt
bash /root/glm_screen.sh "$LBL" > $D/screen-$LBL.log 2>&1
O=$D/$LBL
kv=$(grep -aoE "GPU KV cache size: [0-9,]+" $O/kv-context.txt 2>/dev/null | tail -1 | grep -oE "[0-9,]+")
q=$(grep -aoE "PASS|FAIL" $O/quality.txt 2>/dev/null | head -1)
row=$(python3 - "$O" "$LBL" <<'PY'
import json,sys,statistics as st
O,L=sys.argv[1],sys.argv[2]
try:
    d=json.load(open(f"{O}/bench-{L}.json"))
except Exception as e:
    print("no bench json"); raise SystemExit
m={(b["category"],b["c"]):b for b in d["batches"]}
a={}
for b in d["batches"]:
    if b["category"]=="ceiling_count": continue
    a.setdefault(b["c"],[]).append(b["agg_tok_s"])
g=lambda c,k="per_stream_tok_s": (f"{m[(c,1)][k]:.1f}" if (c,1) in m else "-")
pf=d.get("prefill") or []
p=lambda t: next((f"{x['prefill_tok_s']:.0f}" for x in pf if x.get("target")==t), "-")
print(f"agg " + " ".join(f"C{c} {st.mean(v):.1f}" for c,v in sorted(a.items()))
      + f" | code {g('coding')} json {g('json')} math {g('math')} prose {g('prose')} count {g('ceiling_count')} fmt {g('format')}"
      + f" | pf8k {p(8000)} pf32k {p(32000)}")
PY
)
echo "$LBL | KV $kv | quality $q | $row" >> $D/table.txt
echo "$(date -u +%T) SCREENED $LBL" >> $D/status.txt
tail -1 $D/table.txt
