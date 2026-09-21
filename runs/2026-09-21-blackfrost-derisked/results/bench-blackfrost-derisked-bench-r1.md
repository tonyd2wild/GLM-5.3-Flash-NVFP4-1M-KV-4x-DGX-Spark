## blackfrost-derisked-bench-r1 (2026-09-20T23:52:29Z)

blackfrost-derisked-bench rep 1

Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 66.11 | 74.03 | 0.238 |
| C2 | 96.64 | 57.25 | 0.268 |
| C3 | 123.23 | 49.78 | 0.344 |
| C4 | 134.57 | 42.8 | 0.396 |
| C5 | 152.92 | 38.93 | 0.443 |
| C6 | 165.56 | 36.06 | 0.508 |

### Per-stream tok/s by category

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| coding | 95.85 | 70.69 | 59.02 | 39.61 | 37.69 | 40.09 |
| json | 85.19 | 68.24 | 53.58 | 57.79 | 48.28 | 36.46 |
| narrative | 41.22 | 29.56 | 25.09 | 23.19 | 20.0 | 17.65 |
| prose | 45.57 | 35.14 | 34.27 | 28.39 | 25.29 | 21.01 |
| math | 83.67 | 66.98 | 60.27 | 48.2 | 42.45 | 43.57 |
| reasoning | 78.22 | 47.52 | 39.6 | 33.48 | 32.79 | 32.67 |
| summary | 56.12 | 47.2 | 39.01 | 33.17 | 29.64 | 29.72 |
| format | 106.4 | 92.65 | 87.37 | 78.53 | 75.29 | 67.33 |
| ceiling_count | 138.14 | 114.68 | 96.36 | 85.33 | 77.83 | 74.66 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 3814 | 1.948 | 1958.2 |
| 8000 | 15168 | 7.525 | 2015.6 |
| 32000 | 60917 | 30.569 | 1992.8 |
| 64000 | 121681 | 69.102 | 1760.9 |
