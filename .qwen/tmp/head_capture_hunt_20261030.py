import os, struct, io, json

ROOT = r"C:\kien\dlss5-on-amd-9070xt-porting"
NEED = {65536: "65536 B (65536 x E4M3 | 16384 f32)",
        131072: "131072 B (65536 x f16)",
        262144: "262144 B (65536 x f32)",
        2097152: "2097152 B (2 MiB physical arena)"}

hits = []
for d, dirs, fs in os.walk(ROOT):
    if os.sep + ".git" + os.sep in d + os.sep:
        continue
    for f in fs:
        p = os.path.join(d, f)
        try:
            s = os.path.getsize(p)
        except OSError:
            continue
        if s in NEED:
            hits.append((s, p))

print("files whose size is exactly one of", sorted(NEED), ":", len(hits))
KEY = ("head", "pool", "block30", "block29", "rgb512", "vit", "split")
by_size = {}
for s, p in sorted(hits, key=lambda t: (-t[0], t[1])):
    by_size.setdefault(s, []).append(p)
for s, ps in sorted(by_size.items()):
    print()
    print("==", s, NEED[s], "->", len(ps), "files")
    named = [p for p in ps if any(k in os.path.basename(p).lower() for k in KEY)]
    print("  name-matched:", len(named))
    for p in named[:15]:
        print("   ", p.replace(ROOT, "."))
    for p in ps[:5]:
        print("   sample:", p.replace(ROOT, "."))
