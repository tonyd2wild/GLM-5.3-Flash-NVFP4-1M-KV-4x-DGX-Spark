#!/usr/bin/env python3
"""Token-corruption probe for vLLM #54150.

ModelOpt NVFP4 builds are reported to emit intermittent corrupted token IDs: nearly invisible in English until
one lands in a tool-call block. This sends English-only prompts at temperature 0 and counts output characters
that have no business being there (CJK, Cyrillic, Hangul, Arabic, and replacement chars).

The hypothesis under test: the corruption comes from the W4A4 activation path reading absent or placeholder
input_scale values, so a W4A16_NVFP4 (weight-only) lane should score zero.
"""
import json,os,re,sys,urllib.request
MODEL=os.environ.get("BENCH_MODEL","glm-5.3-flash")
BASE=os.environ.get("BASE","http://127.0.0.1:8000/v1")
BAD=re.compile(r"[　-鿿가-힯Ѐ-ӿ؀-ۿ�]")
PROMPTS=[
 ("long prose","Write 400 words of plain English prose about the history of the shipping container. No lists."),
 ("code","Write a complete Python module implementing an LRU cache with type hints and docstrings. Code only."),
 ("json","Emit a JSON array of 30 objects, each with id, name, city and score fields. JSON only, no prose."),
 ("tool-ish","Produce a JSON tool call: {\"name\":\"search\",\"arguments\":{\"query\":\"...\"}} repeated for 20 different queries about European rivers."),
 ("repetitive","Count from 1 to 200 in words, one per line, English only."),
]
tot_bad=0; tot_chars=0
for tag,p in PROMPTS:
    body={"model":MODEL,"max_tokens":900,"temperature":0,"messages":[{"role":"user","content":p}]}
    try:
        r=urllib.request.urlopen(urllib.request.Request(BASE+"/chat/completions",json.dumps(body).encode(),
            {"Content-Type":"application/json"}),timeout=300)
        txt=json.load(r)["choices"][0]["message"]["content"]
    except Exception as e:
        print("  %-12s request failed: %s"%(tag,repr(e)[:70])); continue
    hits=BAD.findall(txt); tot_bad+=len(hits); tot_chars+=len(txt)
    ex=""
    if hits:
        i=BAD.search(txt).start(); ex="  e.g. ...%s..."%txt[max(0,i-28):i+28].replace("\n"," ")
    print("  %-12s %5d chars, %2d suspect%s"%(tag,len(txt),len(hits),ex))
print("\nTOTAL: %d suspect characters in %d generated (%s)"%(tot_bad,tot_chars,
      "CLEAN" if tot_bad==0 else "CORRUPTION PRESENT"))
