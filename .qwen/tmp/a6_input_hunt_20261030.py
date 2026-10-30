import os, re
D = r"C:\kien\dlss5-on-amd-9070xt-porting\Development"
A = os.path.join(D, "assets")
names = []
for b in range(31, 39):
    names.append(f"block{b}-vit-expand-effective.f16")
    names.append(f"block{b}-qkv-main.fp8" if b != 31 else "block31-qkv-effective.fp8")
    names.append(f"block{b}-qkv-work.f16" if b != 31 else "block31-qkv-work-effective.f16")
print("=== required assets (assets dir =", A, "exists", os.path.isdir(A), ")")
for n in names:
    for base in (A, D):
        p = os.path.join(base, n)
        if os.path.exists(p):
            print("  FOUND", os.path.getsize(p), p)
            break
    else:
        print("  MISSING", n)
print("=== payload fallback availability")
idx = os.path.join(D, "weights-arena-index.json")
recs = {r["name"]: r for r in __import__("json").loads(open(idx, encoding="utf-8").read())["records"]}
for b in range(31, 39):
    print("  block", b, [recs.get(f"block{b}.layer{l}.layer", {}).get("payload_size") for l in range(5)])
print("=== every Development file sized 65536 or 2097152 (candidate physical blobs)")
for dirpath, dirnames, filenames in os.walk(D):
    for f in filenames:
        p = os.path.join(dirpath, f)
        try:
            s = os.path.getsize(p)
        except OSError:
            continue
        if s in (65536, 2097152):
            print("  ", s, os.path.relpath(p, D))
print("=== invocation lines naming vit_blocks31_38_reference or vit-input")
pat = re.compile(r"vit_blocks31_38_reference|vit-input|vit_output|vit-out", re.I)
for f in sorted(os.listdir(D)):
    p = os.path.join(D, f)
    if os.path.isfile(p) and os.path.getsize(p) < 3_000_000:
        try:
            txt = open(p, "r", encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        for n, line in enumerate(txt.splitlines(), 1):
            if pat.search(line):
                print("  ", f, n, line[:300].replace("\n", " "))
