#!/usr/bin/env python3
"""Download Blackfrost-AI/GLM-5.3-Flash-DERISKED-NVFP4 (191.0 GiB, 120 shards).

Settings copied from qwen38fn-nvidia-dl.py, which completed a 124 GiB pull on this host: hf_transfer OFF,
max_workers 8. snapshot_download resumes, so the retry loop is safe to re-enter after a network drop.
Lane A keeps serving throughout - this is disk and network only.
"""
import os,time,sys
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="0"
from huggingface_hub import snapshot_download
REPO="Blackfrost-AI/GLM-5.3-Flash-DERISKED-NVFP4"
DEST="/var/tmp/models/GLM-5.3-Flash-DERISKED-NVFP4-blackfrost"
t0=time.time()
for attempt in range(1,9):
    try:
        p=snapshot_download(REPO,local_dir=DEST,max_workers=8,token=False)
        print("DONE %s after %d s (attempt %d)"%(p,round(time.time()-t0),attempt),flush=True)
        sys.exit(0)
    except Exception as e:
        print("attempt %d failed after %d s: %r"%(attempt,round(time.time()-t0),e)[:400],flush=True)
        time.sleep(20)
print("GAVE UP after %d s"%round(time.time()-t0),flush=True)
sys.exit(1)
