reps loaded: baseline (LibertAI+NVFP4)=3, Lane A (nvidia)=3, Lane B (nvidia+ablit)=3, Blackfrost DERISKED=3

## C1 per-stream tok/s (medians), and whether the difference is resolvable

| category | baseline (LibertAI+NVFP4) | Lane A (nvidia) | Lane B (nvidia+ablit) | Blackfrost DERISKED |
|---|---|---|---|---|
| **code** | 92.7 (1.12x) | 78.7 (1.17x) within noise | 94.3 (1.88x) within noise | 95.8 (1.01x) within noise |
| **JSON** | 74.0 (1.10x) | 75.9 (1.04x) within noise | 78.2 (1.58x) within noise | 80.8 (1.09x) within noise |
| **math** | 101.3 (1.08x) | 88.8 (1.09x) **-12%** | 78.0 (1.36x) within noise | 83.3 (1.01x) **-18%** |
| **prose** | 43.7 (1.06x) | 40.8 (1.10x) within noise | 41.1 (1.56x) within noise | 50.8 (1.12x) **+16%** |
| **structure** | 114.7 (1.05x) | 106.3 (1.15x) within noise | 111.1 (1.32x) within noise | 106.4 (1.18x) within noise |
| **counting** | 120.0 (1.46x) | 107.7 (1.12x) within noise | 106.0 (1.35x) within noise | 138.1 (1.03x) within noise |
| **reasoning** | 70.1 (1.37x) | 71.5 (1.15x) within noise | 67.6 (1.09x) within noise | 76.9 (1.05x) within noise |
| **summary** | 53.9 (1.12x) | 51.5 (1.11x) within noise | 55.8 (1.07x) within noise | 58.3 (1.09x) within noise |
| **narrative** | 33.3 (1.13x) | 38.2 (1.12x) **+15%** | 38.0 (1.59x) within noise | 41.2 (1.07x) **+24%** |

## aggregate by concurrency, and TTFT

| level | baseline (LibertAI+NVFP4) | Lane A (nvidia) | Lane B (nvidia+ablit) | Blackfrost DERISKED |
|---|---|---|---|---|
| C1 | 64.1 (ttft 0.219) | 63.2 (ttft 0.200) | 64.7 (ttft 0.221) | 66.0 (ttft 0.219) |
| C2 | 96.1 (ttft 0.256) | 93.3 (ttft 0.254) | 95.1 (ttft 0.255) | 96.5 (ttft 0.251) |
| C3 | 114.9 (ttft 0.286) | 117.0 (ttft 0.288) | 115.2 (ttft 0.283) | 125.0 (ttft 0.283) |
| C4 | 130.7 (ttft 0.412) | 140.0 (ttft 0.408) | 139.8 (ttft 0.406) | 137.4 (ttft 0.397) |
| C5 | 146.2 (ttft 0.417) | 152.9 (ttft 0.439) | 149.3 (ttft 0.425) | 148.8 (ttft 0.411) |
| C6 | 167.2 (ttft 0.407) | 172.4 (ttft 0.413) | 162.9 (ttft 0.417) | 165.5 (ttft 0.438) |

## cold prefill, rep 1 only

| target | baseline (LibertAI+NVFP4) | Lane A (nvidia) | Lane B (nvidia+ablit) | Blackfrost DERISKED |
|---|---|---|---|---|
| 2000 | 1979 tok/s (3814 tok, ttft 1.9s) | 1092 tok/s (3814 tok, ttft 3.5s) | 2008 tok/s (3814 tok, ttft 1.9s) | 1958 tok/s (3814 tok, ttft 1.9s) |
| 8000 | 2009 tok/s (15168 tok, ttft 7.5s) | 926 tok/s (15168 tok, ttft 16.4s) | 1935 tok/s (15168 tok, ttft 7.8s) | 2016 tok/s (15168 tok, ttft 7.5s) |
| 32000 | 1998 tok/s (60917 tok, ttft 30.5s) | 1557 tok/s (60917 tok, ttft 39.1s) | 1964 tok/s (60917 tok, ttft 31.0s) | 1993 tok/s (60917 tok, ttft 30.6s) |
| 64000 | 1977 tok/s (121681 tok, ttft 61.6s) | 1576 tok/s (121681 tok, ttft 77.2s) | 1945 tok/s (121681 tok, ttft 62.6s) | 1761 tok/s (121681 tok, ttft 69.1s) |

Lane A C8-C16: C8: mean-of-medians 193.1 | forma 292 math 238 json 236 codin 230 | C12: mean-of-medians 241.9 | forma 391 json 293 ceili 292 math 289 | C16: mean-of-medians 271.1 | forma 441 math 341 codin 338 ceili 301

Lane A C24-C32: C24: mean agg 315.4 | forma 518 (27.8/stream) math 397 (20.3/stream) codin 381 (19.2/stream) ceili 356 (25.2/stream) reaso 297 (14.8/stream) json 296 (20.5/stream) summa 247 (13.7/stream) prose 210 (9.6/stream) narra 178 (8.3/stream) | C32: mean agg 352.8 | forma 522 (22.5/stream) math 466 (17.5/stream) codin 460 (17.3/stream) ceili 424 (21.3/stream) json 356 (17.1/stream) reaso 347 (12.7/stream) summa 256 (10.5/stream) prose 222 (7.8/stream) narra 192 (6.8/stream)

Lane A acceptance: draft 584745 accepted 230269 acceptance 0.394 | drafts 83535 accepted/draft 2.76 tokens per step 3.76

Lane B C8-C16: C8: mean-of-medians 202.3 | forma 318 ceili 305 json 253 codin 252 | C12: mean-of-medians 244.9 | forma 412 ceili 355 json 299 codin 292 | C16: mean-of-medians 273.8 | forma 439 ceili 381 codin 339 math 327

Lane B C24-C32: C24: mean agg 319.6 | forma 557 (29.8/stream) ceili 470 (24.9/stream) codin 415 (19.9/stream) math 398 (21.0/stream) json 298 (19.7/stream) reaso 279 (14.2/stream) summa 238 (13.1/stream) prose 200 (9.6/stream) narra 171 (8.1/stream) | C32: mean agg 354.3 | forma 551 (23.6/stream) ceili 523 (20.4/stream) codin 504 (17.9/stream) math 452 (18.2/stream) json 343 (16.2/stream) reaso 315 (11.4/stream) summa 255 (10.5/stream) prose 224 (7.8/stream) narra 190 (6.7/stream)

Lane B acceptance: draft 586537 accepted 231996 acceptance 0.396 | drafts 83791 accepted/draft 2.77 tokens per step 3.77
