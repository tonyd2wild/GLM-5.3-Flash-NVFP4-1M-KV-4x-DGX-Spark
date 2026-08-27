import json
import time
import urllib.request

URL = "http://100.113.138.96:8000/v1/chat/completions"
with open("glm-bench.json", "rb") as handle:
    payload = handle.read()

for run in range(1, 4):
    request = urllib.request.Request(
        URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    start = time.perf_counter()
    first_token = None
    completion_tokens = None
    finish_reason = None
    with urllib.request.urlopen(request, timeout=300) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            choices = event.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if first_token is None and (delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")):
                    first_token = time.perf_counter()
                finish_reason = choices[0].get("finish_reason") or finish_reason
            usage = event.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens")
    end = time.perf_counter()
    ttft = (first_token - start) if first_token is not None else float("nan")
    decode_seconds = max(end - (first_token or start), 1e-9)
    decode_tokens = max((completion_tokens or 0) - 1, 0)
    print(json.dumps({
        "run": run,
        "ttft_s": round(ttft, 4),
        "total_s": round(end - start, 4),
        "completion_tokens": completion_tokens,
        "decode_tok_s": round(decode_tokens / decode_seconds, 3),
        "finish_reason": finish_reason,
    }), flush=True)
