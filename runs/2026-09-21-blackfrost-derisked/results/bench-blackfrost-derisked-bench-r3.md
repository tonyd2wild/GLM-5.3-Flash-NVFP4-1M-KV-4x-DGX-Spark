## blackfrost-derisked-bench-r3 (2026-09-21T00:03:01Z)

blackfrost-derisked-bench rep 3

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 65.91 | 73.14 | 0.236 |
| C2 | 96.57 | 56.66 | 0.27 |
| C3 | 130.06 | 52.96 | 0.319 |
| C4 | 139.42 | 43.46 | 0.394 |
| C5 | 150.47 | 38.35 | 0.433 |
| C6 | 165.86 | 35.86 | 0.536 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 94.62 | 70.79 | 65.99 | 50.82 | 40.03 | 40.46 |
| json | 78.43 | 64.58 | 56.52 | 55.59 | 42.76 | 37.95 |
| narrative | 41.38 | 30.16 | 24.63 | 21.94 | 19.39 | 18.14 |
| prose | 50.82 | 35.57 | 32.31 | 26.82 | 23.21 | 21.62 |
| math | 83.26 | 68.05 | 65.11 | 52.4 | 44.23 | 42.98 |
| reasoning | 76.87 | 47.47 | 47.09 | 38.75 | 33.61 | 29.94 |
| summary | 58.28 | 41.18 | 41.61 | 32.62 | 32.76 | 26.92 |
| format | 101.47 | 95.44 | 90.46 | 68.73 | 70.8 | 68.88 |
| ceiling_count | 138.66 | 115.32 | 102.44 | 87.2 | 84.56 | 74.89 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 3814 | 0.927 | 4112.4 |
| 8000 | 15168 | 0.788 | 19244.3 |
| 32000 | 60917 | 0.734 | 82981.3 |
| 64000 | 121681 | 1.232 | 98761.6 |
