import json, struct
from pathlib import Path

AMD = Path(r"G:\Steam\steamapps\common\Wuthering Waves\Client\Binaries\Win64\dlssnr_on_amd_weights.bin")
INDEX = Path(r"C:\kien\dlss5-on-amd-9070xt-porting\Development\weights-arena-index.json")

payload = AMD.read_bytes()
print("container size", len(payload), "magic", payload[:8], "u16@8", struct.unpack_from("<II", payload, 8))

# framing measured earlier: header 16 bytes, then [u8 name_len][name][u64 off][u64 size]
cur, recs = 16, []
while cur < len(payload):
    n = payload[cur]
    if not 1 <= n <= 64:
        print("stop: implausible name length", n, "at", hex(cur))
        break
    name = payload[cur + 1:cur + 1 + n]
    try:
        name = name.decode("ascii")
    except UnicodeDecodeError:
        print("stop: non-ascii name at", hex(cur))
        break
    off, size = struct.unpack_from("<QQ", payload, cur + 1 + n)
    recs.append({"index": len(recs), "name": name, "container_offset": off, "payload_size": size})
    cur = cur + 1 + n + 16
print("container records parsed", len(recs), "closed at EOF", cur == len(payload))

want = json.loads(INDEX.read_text(encoding="utf-8"))
wi = want["records"]
print("index records", len(wi), "alignment", want["alignment"], "arena_size", want["arena_size"])
print("total container payload", sum(r["payload_size"] for r in recs))
print("total index payload     ", sum(r["payload_size"] for r in wi))

names_match = [(i, a["name"], b["name"]) for i, (a, b) in enumerate(zip(recs, wi)) if a["name"] != b["name"]]
sizes_match = [(i, a["name"], a["payload_size"], b["payload_size"]) for i, (a, b) in enumerate(zip(recs, wi)) if a["payload_size"] != b["payload_size"]]
print("name order mismatches", len(names_match), names_match[:5])
print("payload_size mismatches", len(sizes_match), sizes_match[:5])

# rebuild the arena from the container and compare offsets to the index
arena, cursor, rebuilt = bytearray(), 0, []
for r in recs:
    if cursor % 512:
        pad = 512 - cursor % 512
        arena += b"\x00" * pad
        cursor += pad
    r_start = cursor
    arena += payload[r["container_offset"]:r["container_offset"] + r["payload_size"]]
    cursor += r["payload_size"]
    rebuilt.append({"index": r["index"], "name": r["name"], "arena_offset": r_start,
                    "payload_size": r["payload_size"], "aligned_size": (r["payload_size"] + 511) // 512 * 512})
print("rebuilt arena size", len(arena), "matches index arena_size", len(arena) == want["arena_size"])
off_diff = [(a["name"], a["arena_offset"], b["arena_offset"]) for a, b in zip(rebuilt, wi) if a["arena_offset"] != b["arena_offset"]]
sz_diff = [(a["name"], a["aligned_size"], b["aligned_size"]) for a, b in zip(rebuilt, wi) if a["aligned_size"] != b["aligned_size"]]
print("arena_offset mismatches", len(off_diff), off_diff[:5])
print("aligned_size mismatches", len(sz_diff), sz_diff[:5])

# the container is header(16) + a 5,673-byte name/offset TABLE + payload blob,
# so the table deliberately does not close at EOF.
if len(arena) == want["arena_size"] - 448:
    arena += b"\x00" * (want["arena_size"] - len(arena))
    out = Path(r"C:\kien\dlss5-on-amd-9070xt-porting\.qwen\tmp\weights-arena-from-container-20261030.bin")
    out.write_bytes(bytes(arena))
    print("WROTE", out, len(arena))
    for name in ("block31.layer31.layer", "block31.layer32.layer", "block38.layer38.layer"):
        r = next(x for x in rebuilt if x["name"] == name)
        o = r["arena_offset"]
        head = bytes(arena[o:o + 16])
        tail = bytes(arena[o + r["payload_size"] - 8:o + r["payload_size"]])
        print(f"  {name:<28} off={o:<10} size={r['payload_size']:<8} head={head.hex()} tail={tail.hex()}")
