#!/usr/bin/env python3
"""gb10_alloc_probe.py — walk GPU allocations on GB10 until the NVRM wall.

WARNING: allocates all available GPU/system memory. Maintenance window only —
never run on a node that is serving. Run cold, then after
`sync; echo 3 > /proc/sys/vm/drop_caches`, to quantify the page-cache term.
Capture `dmesg -T | grep NV_ERR` between runs.

Usage: python3 gb10_alloc_probe.py [--step-gb 1] [--max-gb 128] [--managed] [--no-touch]
"""
import argparse, ctypes, ctypes.util, sys, time

def load_cudart():
    for name in ("libcudart.so", "libcudart.so.13", "libcudart.so.12",
                 ctypes.util.find_library("cudart")):
        if not name:
            continue
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
    sys.exit("FATAL: libcudart not found (set LD_LIBRARY_PATH to CUDA runtime)")

def meminfo():
    d = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            d[k] = int(v.split()[0])  # kB
    return d

def gb(kb):
    return kb / (1024 * 1024)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-gb", type=float, default=1.0)
    ap.add_argument("--max-gb", type=float, default=128.0)
    ap.add_argument("--managed", action="store_true",
                    help="use cudaMallocManaged instead of cudaMalloc")
    ap.add_argument("--no-touch", action="store_true",
                    help="skip cudaMemset (measures reservation wall only)")
    a = ap.parse_args()

    rt = load_cudart()
    rt.cudaGetErrorString.restype = ctypes.c_char_p
    def errstr(e):
        return rt.cudaGetErrorString(e).decode()

    step = int(a.step_gb * (1 << 30))
    ptrs = []
    print(f"# mode={'managed' if a.managed else 'device'} step={a.step_gb} GiB "
          f"touch={not a.no_touch}")
    print(f"{'step':>4} {'total_GiB':>9} {'malloc':>8} {'touch':>8} "
          f"{'MemFree':>8} {'MemAvail':>8} {'Cached':>8}  t_ms")

    total = 0
    i = 0
    while total + step <= int(a.max_gb * (1 << 30)):
        i += 1
        p = ctypes.c_void_p()
        t0 = time.time()
        if a.managed:
            e = rt.cudaMallocManaged(ctypes.byref(p), ctypes.c_size_t(step),
                                     ctypes.c_uint(1))  # cudaMemAttachGlobal
        else:
            e = rt.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(step))
        malloc_res = "ok" if e == 0 else f"E{e}"
        touch_res = "-"
        if e == 0:
            ptrs.append(p)
            total += step
            if not a.no_touch:
                e2 = rt.cudaMemset(p, ctypes.c_int(0xA5), ctypes.c_size_t(step))
                e3 = rt.cudaDeviceSynchronize()   # async faults surface here
                bad = e2 or e3
                touch_res = "ok" if not bad else f"E{e2}/E{e3}"
        m = meminfo()
        print(f"{i:>4} {total/(1<<30):>9.1f} {malloc_res:>8} {touch_res:>8} "
              f"{gb(m['MemFree']):>8.1f} {gb(m['MemAvailable']):>8.1f} "
              f"{gb(m['Cached']):>8.1f}  {int((time.time()-t0)*1000)}")
        if e != 0:
            print(f"# WALL(reserve): cudaMalloc failed at +{a.step_gb} GiB past "
                  f"{total/(1<<30):.1f} GiB: {errstr(e)}")
            break
        if touch_res not in ("ok", "-"):
            print(f"# WALL(backing): first-touch fault at {total/(1<<30):.1f} GiB "
                  f"(phantom backing confirmed): memset/sync = {touch_res}")
            break
    else:
        print(f"# reached --max-gb without hitting a wall ({total/(1<<30):.1f} GiB)")

    print("# freeing...")
    for p in ptrs:
        rt.cudaFree(p)
    rt.cudaDeviceSynchronize()
    m = meminfo()
    print(f"# after free: MemFree={gb(m['MemFree']):.1f} "
          f"MemAvail={gb(m['MemAvailable']):.1f} GiB "
          f"(watch for the GB10 delayed-return lag: re-check in 60s)")

if __name__ == "__main__":
    main()
