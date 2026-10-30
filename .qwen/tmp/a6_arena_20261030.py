#!/usr/bin/env python3
"""A6 prerequisite check, numpy-free: rebuild the 512-aligned weight arena
straight out of the on-box DLL and diff it against the committed index."""
import json, struct, sys
from pathlib import Path

ROOT = Path(r"C:\kien\dlss5-on-amd-9070xt-porting")
sys.path.insert(0, str(ROOT / "Development"))
import parse_weights_archive as pwa  # noqa: E402  (main() is not called)

DLL = Path(r"G:\Steam\steamapps\common\Wuthering Waves\Client\Binaries\Win64\dxgi.dll")
INDEX = ROOT / "Development" / "weights-arena-index.json"
AMD = Path(r"G:\Steam\steamapps\common\Wuthering Waves\Client\Binaries\Win64\dlssnr_on_amd_weights.bin")

dll = DLL.read_bytes()
marker = struct.pack("<Q", 17) + b"block0.layer0.layer"
hits = [i for i in range(len(dll)) if dll.startswith(marker, i)]
print(f"archive marker hits in {DLL.name}: {len(hits)} -> {[hex(h) for h in hits]}")
if len(hits) != 1:
    raise SystemExit("ambiguous marker; abort rather than guess")
start = hits[0] - 8
length = struct.unpack_from("<Q", dll, start)[0]
archive = dll[start:start + length]
print(f"archive start 0x{start:x}  declared size {length}  slice len {len(archive)}")

records = pwa.parse_archive(archive)
arena, built = pwa.build_aligned_arena(archive, records)
print(f"records parsed {len(records)}  arena built {len(arena)} bytes")

want = json.loads(INDEX.read_text(encoding="utf-8"))
print(f"committed index: records={len(want['records'])} alignment={want['alignment']} arena_size={want['arena_size']}")
print(f"size match: {len(arena) == want['arena_size']}  count match: {len(built) == len(want['records'])}")

fields = ("name", "index", "arena_offset", "payload_size", "aligned_size")
diffs = []
for b, w in zip(built, want["records"]):
    for f in fields:
        if b.get(f) != w.get(f):
            diffs.append((w["index"], w["name"], f, b.get(f), w.get(f)))
print(f"field diffs: {len(diffs)}")
for row in diffs[:10]:
    print("  DIFF", row)

# Does the AMD container's payload byte-match the DLL arena's payload?
def amd_records(payload):
    cur, out = 16, {}
    while cur < len(payload):
        n = payload[cur]
        name = payload[cur + 1:cur + 1 + n].decode("ascii")
        off, size = struct.unpack_from("<QQ", payload, cur + 1 + n)
        out[name] = (off, size)
        cur = cur + 1 + n + 16
    return out

amd = amd_records(AMD.read_bytes())
by_name = {r["name"]: r for r in built}
same = diff = 0
for name, (off, size) in list(amd.items())[:200]:
    r = by_name.get(name)
    if r is None:
        continue
    if AMD[off:off + size] == arena[r["arena_offset"]:r["arena_offset"] + size]:
        same += 1
    else:
        diff += 1
print(f"AMD-container payload vs arena payload: identical={same} differing={diff}")

block31 = by_name.get("block31.layer31.layer")
print(f"block31.layer31.layer arena_offset={block31['arena_offset']} payload_size={block31['payload_size']}")
out = ROOT / ".qwen" / "tmp" / "weights-arena-rebuilt-20261030.bin"
out.write_bytes(arena)
print(f"wrote: {out} ({len(arena)} bytes)")
