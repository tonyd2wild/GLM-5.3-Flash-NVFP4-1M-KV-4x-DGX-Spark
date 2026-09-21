#!/bin/bash
# Build the Blackfrost DERISKED lane 1:1 with Lane A: same quantizer, same proven 14-entry ignore list, same
# W4A16_NVFP4 declaration, same knobs. Only the weights differ, which is what makes the comparison single-variable.
#
# Blackfrost ships attention UNFUSED (q/k/v/b/f_a/g_a, 34 each) exactly like nvidia stock, so mknvfp4b.py applies
# unchanged. Its config.json declares quant_algo NVFP4 (the W4A4 path) while shipping ZERO input_scale tensors -
# fix_ignore.py rewrites that to W4A16_NVFP4, which is both the ignore-list fix and the W4A4 fix.
#
# Lane A KEEPS SERVING through the whole build. Nothing here touches the running container.
set -u
G=/var/tmp/boot-results/glm53; mkdir -p $G
SRC=/var/tmp/models/GLM-5.3-Flash-DERISKED-NVFP4-blackfrost
DST=/var/tmp/models/blackfrost-glm53-derisked-attn
CACHE=/var/tmp/glm53-vllm-cache/bf-attn
IMG=ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2
say(){ echo "$(date -u +%T) BFBUILD $*" | tee -a $G/status.txt; }

say "waiting for download to finish"
for i in $(seq 1 480); do
  grep -q "^DONE " /tmp/blackfrost-dl.log 2>/dev/null && break
  grep -q "^GAVE UP" /tmp/blackfrost-dl.log 2>/dev/null && { say "DOWNLOAD FAILED"; exit 2; }
  sleep 30
done
grep -q "^DONE " /tmp/blackfrost-dl.log 2>/dev/null || { say "download did not finish in time"; exit 2; }
say "download complete: $(du -sh $SRC | cut -f1)"

# The launcher hard-requires the multimodal chat template; Blackfrost may not ship it.
if [ ! -f "$SRC/chat_template_mm.jinja" ]; then
  say "chat_template_mm.jinja absent -> copying from the nvidia pack (same architecture)"
  sudo cp /var/tmp/models/GLM-5.3-Flash-NVFP4-nvidia/chat_template_mm.jinja "$SRC/chat_template_mm.jinja"
fi

say "verifying the download (tensor/shard census vs index)"
sudo python3 - "$SRC" <<'PY'
import json,os,sys
D=sys.argv[1]
idx=json.load(open(os.path.join(D,"model.safetensors.index.json")))["weight_map"]
files=sorted(set(idx.values()))
missing=[f for f in files if not os.path.exists(os.path.join(D,f))]
tot=sum(os.path.getsize(os.path.join(D,f)) for f in files if os.path.exists(os.path.join(D,f)))
print("  tensors %d  shards %d  missing %d  bytes %.1f GiB"%(len(idx),len(files),len(missing),tot/2**30))
if missing: print("  MISSING:",missing[:5]); sys.exit(3)
PY
[ $? -ne 0 ] && { say "DOWNLOAD INCOMPLETE - refusing to build"; exit 3; }

say "step 1/3 quantize attention groups (mknvfp4b.py, shared amax per fused group)"
rm -rf $CACHE; mkdir -p $CACHE
sudo docker rm -f conv_bf >/dev/null 2>&1
sudo docker run --rm --name conv_bf --gpus all -e OUT=/cache/bf-attn -e SRC=/models/src \
  -v /root/mknvfp4b.py:/tmp/s.py:ro -v $SRC:/models/src:ro -v /var/tmp/glm53-vllm-cache:/cache \
  --entrypoint python3 $IMG /tmp/s.py > $G/bf-convert.log 2>&1
say "convert rc=$? $(grep -aoE 'groups: [0-9]+|tensors written: [0-9]+|REFUSING.*' $G/bf-convert.log | tail -2 | tr '\n' ' ')"
grep -qa "REFUSING" $G/bf-convert.log && { say "QUANTIZER REFUSED - see bf-convert.log"; exit 4; }

say "step 2/3 assemble the lane (build_nv.py, hardlinks clean shards)"
sudo docker run --rm --entrypoint python3 -e SRC=/out/GLM-5.3-Flash-DERISKED-NVFP4-blackfrost \
  -e DST=/out/blackfrost-glm53-derisked-attn -e NEWDIR=/newshard \
  -v /root/build_nv.py:/tmp/s.py:ro -v $CACHE:/newshard:ro -v /var/tmp/models:/out \
  $IMG /tmp/s.py > $G/bf-build.log 2>&1
say "build rc=$? $(grep -aE 'VERIFY|dup|missing|mis-pointed' $G/bf-build.log | tail -2 | tr '\n' ' ')"
[ -f $DST/config.json ] || { say "BUILD PRODUCED NO config.json"; exit 5; }

say "step 3/3 proven ignore list + W4A16_NVFP4 (also closes the W4A4/no-input_scale trap)"
sudo python3 /root/fix_ignore.py $DST | tee -a $G/status.txt
[ -f $DST/chat_template_mm.jinja ] || sudo cp $SRC/chat_template_mm.jinja $DST/ 2>/dev/null

say "census of the built lane"
sudo python3 - "$DST" <<'PY'
import json,sys
D=sys.argv[1]
i=json.load(open(D+"/model.safetensors.index.json"))["weight_map"]
def c(p): return sum(1 for k in i if k.endswith(p))
print("  q_proj.weight/scale/scale_2 : %d / %d / %d"%(c("self_attn.q_proj.weight"),c("self_attn.q_proj.weight_scale"),c("self_attn.q_proj.weight_scale_2")))
print("  o_proj.weight/scale         : %d / %d"%(c("self_attn.o_proj.weight"),c("self_attn.o_proj.weight_scale")))
print("  expert down_proj weight     : %d"%c("down_proj.weight"))
print("  total tensors / shards      : %d / %d"%(len(i),len(set(i.values()))))
q=json.load(open(D+"/config.json"))["quantization_config"]
print("  quant_algo                  : %s   ignore entries: %d"%(q.get("quant_algo"),len(q.get("ignore",[]))))
PY
say "BUILD DONE - lane dir $DST ready to boot. Lane A still serving; not swapping without the operator."
