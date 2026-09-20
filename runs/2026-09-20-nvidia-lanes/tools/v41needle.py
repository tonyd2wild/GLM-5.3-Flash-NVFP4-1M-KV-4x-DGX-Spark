#!/usr/bin/env python3
"""Long-context needle test for DeepSeek-V4.1-Flash. One fact is buried at a given depth inside
deterministic filler; the model must return it. Measures prompt tokens (server usage), TTFT
(= prefill time) and whether the answer contains the needle. Temperature 0, thinking off.

usage: v41needle.py --base http://127.0.0.1:8000/v1 --model deepseek-v4.1-flash \
         --targets 131072,262144 [--depth 0.5] [--out results.json]
"""
import os
import argparse, json, random, time, urllib.request

WORDS = ("amber basin cedar delta ember fjord garnet harbor iris juniper kestrel lumen meadow nimbus orchid "
         "pylon quartz raven sierra tundra umber vessel willow xenon yarrow zephyr").split()
NEEDLE = "Note for the record: the vault passphrase is COPPER-LANTERN-8315."
QUESTION = "\n\nWhat is the vault passphrase mentioned in the text above? Reply with the passphrase only."
ANSWER = "COPPER-LANTERN-8315"


def filler(n_words, seed):
    rnd = random.Random(seed)
    out = []
    for i in range(n_words):
        out.append(f"{rnd.choice(WORDS)}{rnd.randint(0, 999)}")
        if i % 16 == 15:
            out.append(".")
    return " ".join(out)


def chat(base, model, prompt, max_tokens, timeout):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; usage = None; text = ""
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece and ttft is None:
                    ttft = time.time() - t0
                text += piece
    return {"ttft_s": ttft, "total_s": time.time() - t0, "usage": usage, "text": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default=os.environ.get("BENCH_MODEL", "deepseek-v4.1-flash"))
    ap.add_argument("--targets", default="131072")
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--out")
    args = ap.parse_args()
    probe = chat(args.base, args.model, filler(4000, 1), 1, 600)
    per_word = probe["usage"]["prompt_tokens"] / 4000.0
    print(f"calibration: {per_word:.3f} prompt tokens per filler word", flush=True)
    rows = []
    for tgt in [int(x) for x in args.targets.split(",")]:
        n = int((tgt - 200) / per_word)
        cut = int(n * args.depth)
        doc = filler(cut, tgt) + " " + NEEDLE + " " + filler(n - cut, tgt + 1)
        r = chat(args.base, args.model, f"[needle {tgt}] " + doc + QUESTION, 24, 3600)
        pt = (r["usage"] or {}).get("prompt_tokens")
        ok = ANSWER in r["text"].upper().replace(" ", "")
        row = {"target": tgt, "prompt_tokens": pt, "depth": args.depth, "ttft_s": round(r["ttft_s"] or 0, 1),
               "prefill_tok_s": round(pt / r["ttft_s"], 1) if pt and r["ttft_s"] else None,
               "answer": r["text"].strip()[:80], "pass": ok}
        rows.append(row)
        print(json.dumps(row), flush=True)
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
