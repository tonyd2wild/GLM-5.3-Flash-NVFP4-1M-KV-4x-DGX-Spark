from pathlib import Path

# FA2 MLA fp8-KV on SM12x. The fp8 branch forces CTA_TILE_KV=32 (a Hopper
# 228KB-smem assumption); on GB10's ~101KB opt-in max that doubles the tile
# picked by DISPATCH_SMEM_CONFIG (16) and over-requests smem (117,312B) ->
# cudaFuncSetAttribute "invalid argument". Cap instead of force: fp8 keeps
# TKV<=32, i.e. 16 on 100KB devices (91,680B, verified fitting + correct on
# GB10: all probe cases clean, rel_err ~0.005 vs fp32 reference).
p = Path("/usr/local/lib/python3.12/dist-packages/flashinfer/data/include/flashinfer/attention/mla.cuh")
s = p.read_text()
old = "    constexpr uint32_t EFF_CTA_TILE_KV = std::is_same_v<DTypeKV, __nv_fp8_e4m3> ? 32 : CTA_TILE_KV;\n"
new = "    constexpr uint32_t EFF_CTA_TILE_KV = std::is_same_v<DTypeKV, __nv_fp8_e4m3> ? (CTA_TILE_KV < 32u ? CTA_TILE_KV : 32u) : CTA_TILE_KV;\n"
if s.count(old) != 1:
    raise SystemExit("mla.cuh fp8 tile line match count: %d" % s.count(old))
p.write_text(s.replace(old, new))

p = Path("/usr/local/lib/python3.12/dist-packages/flashinfer/mla/_core.py")
s = p.read_text()
old = "            major, minor = get_compute_capability(self.device)\n            if major != 9:\n"
new = "            major, minor = get_compute_capability(self.device)\n            if major not in (9, 12):\n"
if s.count(old) != 1:
    raise SystemExit("_core.py fp8 gate match count: %d" % s.count(old))
p.write_text(s.replace(old, new))
print("fp8 MLA sm12x patches applied")
