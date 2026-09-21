## blackfrost-derisked-bench-r2 (2026-09-20T23:58:36Z)

blackfrost-derisked-bench rep 2

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 67.77 | 75.54 | 0.24 |
| C2 | 94.97 | 55.01 | 0.29 |
| C3 | 117.64 | 48.58 | 0.335 |
| C4 | 136.79 | 42.94 | 0.394 |
| C5 | 145.21 | 36.57 | 0.458 |
| C6 | 175.93 | 36.31 | 0.497 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 95.89 | 67.8 | 45.45 | 49.24 | 39.84 | 47.18 |
| json | 80.8 | 63.46 | 52.24 | 54.45 | 42.77 | 38.74 |
| narrative | 38.56 | 29.24 | 23.64 | 20.61 | 19.97 | 17.87 |
| prose | 51.11 | 37.07 | 34.21 | 23.32 | 22.76 | 21.93 |
| math | 83.04 | 57.27 | 65.63 | 49.9 | 45.18 | 41.81 |
| reasoning | 74.65 | 49.22 | 43.62 | 36.47 | 33.45 | 30.48 |
| summary | 60.9 | 43.57 | 38.73 | 34.88 | 26.35 | 31.36 |
| format | 119.35 | 92.48 | 85.11 | 74.64 | 62.2 | 61.09 |
| ceiling_count | 134.59 | 106.54 | 105.65 | 87.54 | 80.71 | 76.18 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 3814 | 2.098 | 1817.9 |
| 8000 | 15168 | 1.994 | 7605.8 |
| 32000 | 60917 | 1.909 | 31910.3 |
| 64000 | 121681 | 2.469 | 49276.2 |
