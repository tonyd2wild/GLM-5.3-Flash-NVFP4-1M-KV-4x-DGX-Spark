#!/usr/bin/env python3
"""kvtest.py N TARGET_TOKENS LABEL [ABORT_GIB]
N concurrent distinct long prompts (salted filler, distinct cache_salt), min_tokens=max_tokens=512, ignore_eos.
Samples every 1 s: vLLM kv_cache_usage_perc / running / waiting / preemptions, MemAvailable on all 4 nodes.
Watchdog: any node below ABORT_GIB (default 5) -> shut every stream's socket (vLLM aborts the requests).
Run on the head (reddie). Stdlib only. Writes ~/kvtest-LABEL.json."""
import http.client, json, os, random, socket, subprocess, sys, threading, time, urllib.request, uuid

BASE, MODEL = "http://localhost:8000", "glm-5.3-flash"
N, TARGET, LABEL = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
ABORT_GIB = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
# The three non-head ranks of your fleet (edit for yours).
REMOTES = {"spark4_r1": "tonyspark4@192.168.192.4", "asusi_r2": "tonyspark3@192.168.192.3",
           "bluey_r3": "tonyspark1@192.168.192.1"}
mem, conns, samples, results = {}, [], [], {}
stop, abort = threading.Event(), threading.Event()

def local_mem():
    while not stop.is_set():
        with open("/proc/meminfo") as f:
            for l in f:
                if l.startswith("MemAvailable"): mem["reddie_r0"] = int(l.split()[1]) / 1048576
        time.sleep(1)

def remote_mem(name, host):
    p = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=5", host,
                          "while :; do awk '/MemAvailable/{print $2}' /proc/meminfo; sleep 1; done"],
                         stdout=subprocess.PIPE, text=True)
    for line in p.stdout:
        if stop.is_set(): break
        try: mem[name] = int(line) / 1048576
        except ValueError: pass
    p.kill()

def metrics():
    out = {}
    for l in urllib.request.urlopen(BASE + "/metrics", timeout=5).read().decode().splitlines():
        for k in ("kv_cache_usage_perc", "num_requests_running", "num_requests_waiting", "num_preemptions_total",
                  "kv_cache_size_tokens"):
            if l.startswith("vllm:" + k + "{"): out[k] = float(l.rsplit(" ", 1)[1])
        if l.startswith("vllm:cache_config_info{"):
            out["pool_tokens"] = int(l.split('kv_cache_size_tokens="')[1].split('"')[0])
    return out

def kill_streams(why):
    if abort.is_set(): return
    abort.set(); print(f"ABORT: {why}", flush=True)
    for c in conns:
        try: c.sock.shutdown(socket.SHUT_RDWR)
        except Exception: pass

def sampler(t0):
    last_print = -10
    while not stop.is_set():
        s = {"t": round(time.monotonic() - t0, 1)}
        try: s.update(metrics())
        except Exception as e: s["merr"] = str(e)[:80]
        s["mem"] = {k: round(v, 1) for k, v in sorted(mem.items())}
        samples.append(s)
        low = {k: round(v, 1) for k, v in mem.items() if v < ABORT_GIB}
        if low: kill_streams(f"{low} below {ABORT_GIB} GiB at t={s['t']}")
        if s["t"] - last_print >= 10:
            last_print = s["t"]
            print(f"t={s['t']:6.0f}s kv={100 * s.get('kv_cache_usage_perc', -1):5.1f}% run={s.get('num_requests_running')} "
                  f"wait={s.get('num_requests_waiting')} pre={s.get('num_preemptions_total')} mem={s['mem']}", flush=True)
        time.sleep(1)

def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))

W = ("system memory network cache latency throughput kernel thread process scheduler compiler garden river mountain "
     "village market harbour teacher student library museum theatre orchestra painter novel poem history economy "
     "policy budget contract engineer doctor patient hospital battery engine turbine bridge tunnel railway airport "
     "weather forecast season harvest winter summer autumn spring ocean island forest desert canyon glacier").split()

def paras(seed, n):
    rng = random.Random(seed)
    return [f"{i}. " + " ".join((" ".join(rng.choice(W) for _ in range(rng.randint(8, 16)))).capitalize() + "."
                                for _ in range(rng.randint(3, 6))) for i in range(n)]

def build(seed, per):
    doc = f"[{uuid.uuid4()}]\n" + "\n".join(paras(seed, int(TARGET / per)))
    return doc + "\n\nSummarise the main themes of the document above in a few sentences."

def run_req(i, prompt, t0):
    r = {"i": i, "start": round(time.monotonic() - t0, 1)}
    results[i] = r
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 512, "min_tokens": 512,
            "ignore_eos": True, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "cache_salt": f"{LABEL}-{i}-{uuid.uuid4()}", "chat_template_kwargs": {"reasoning_effort": "low"}}
    c = http.client.HTTPConnection("localhost", 8000, timeout=3600)
    conns.append(c)
    try:
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        resp = c.getresponse(); r["status"] = resp.status
        if resp.status != 200:
            r["err"] = resp.read()[:400].decode(errors="replace"); return
        for line in resp:
            if not line.startswith(b"data: "): continue
            p = line[6:].strip()
            if p == b"[DONE]": break
            e = json.loads(p)
            if "error" in e: r["err"] = json.dumps(e["error"])[:400]
            if e.get("usage"): r["usage"] = e["usage"]
            if "ttft" not in r and any((ch.get("delta") or {}).get(k) for ch in e.get("choices", [])
                                      for k in ("content", "reasoning_content", "reasoning")):
                r["ttft"] = round(time.monotonic() - t0 - r["start"], 1)
                print(f"req {i}: first token at t={time.monotonic() - t0:.0f}s (ttft {r['ttft']}s)", flush=True)
    except Exception as ex:
        r["err"] = f"{type(ex).__name__}: {ex}"[:400]
    finally:
        r["end"] = round(time.monotonic() - t0, 1)
        print(f"req {i}: done t={r['end']}s usage={r.get('usage')} err={r.get('err')}", flush=True)

def main():
    tk = post("/tokenize", {"model": MODEL, "prompt": "\n".join(paras(7, 400))})
    per = (tk.get("count") or len(tk["tokens"])) / 400  # tokens per paragraph
    prompts = [build(1000 + i, per) for i in range(N)]
    lens = [post("/tokenize", {"model": MODEL, "messages": [{"role": "user", "content": p}]}) for p in prompts]
    lens = [l.get("count") or len(l["tokens"]) for l in lens]
    print(f"{LABEL}: {N} prompts, rendered tokens {lens}", flush=True)
    m0 = metrics(); print(f"before: {m0}", flush=True)
    t0 = time.monotonic()
    threading.Thread(target=local_mem, daemon=True).start()
    for k, h in REMOTES.items(): threading.Thread(target=remote_mem, args=(k, h), daemon=True).start()
    time.sleep(3)
    threading.Thread(target=sampler, args=(t0,), daemon=True).start()
    ts = [threading.Thread(target=run_req, args=(i, p, t0)) for i, p in enumerate(prompts)]
    for t in ts: t.start()
    for t in ts: t.join()
    time.sleep(3); stop.set()
    pool = m0.get("pool_tokens")
    out = {"label": LABEL, "n": N, "prompt_tokens": lens, "pool_tokens": pool, "abort_gib": ABORT_GIB,
           "aborted": abort.is_set(), "results": results, "samples": samples}
    json.dump(out, open(os.path.expanduser(f"~/kvtest-{LABEL}.json"), "w"))
    one = [s for s in samples if s.get("num_requests_running") == 1]
    peak1 = max((s.get("kv_cache_usage_perc", 0) for s in one), default=None)
    print(f"SUMMARY {LABEL}: pool {pool} tok; expected per request ~{100 * lens[0] / pool:.1f}% "
          f"(prompt/pool); observed peak with 1 running {None if peak1 is None else round(100 * peak1, 1)}%; "
          f"peak overall {100 * max(s.get('kv_cache_usage_perc', 0) for s in samples):.1f}%; "
          f"min MemAvailable { {k: min(s['mem'].get(k, 999) for s in samples) for k in sorted(mem)} }; "
          f"preemptions {samples[-1].get('num_preemptions_total')}; aborted {abort.is_set()}", flush=True)

main()
