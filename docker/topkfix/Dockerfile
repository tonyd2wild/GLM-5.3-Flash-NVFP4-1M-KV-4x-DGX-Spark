FROM ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2
ARG VLLM=/usr/local/lib/python3.12/dist-packages/vllm
COPY patch_kpool_topk.py /tmp/patch_kpool_topk.py
RUN python3 /tmp/patch_kpool_topk.py && \
    python3 -c "import ast,pathlib; p=pathlib.Path('$VLLM/model_executor/layers/sparse_attn_indexer_kpool.py'); ast.parse(p.read_text()); t=p.read_text(); assert '_USE_TORCH_TOPK' in t and t.count('_torch_decode_topk')>=2 and t.count('_torch_prefill_topk')>=2, 'topkfix markers missing'; print('topkfix apply + AST check OK')"
