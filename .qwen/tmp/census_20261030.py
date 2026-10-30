import os, time
ROOT = r"C:\kien\dlss5-on-amd-9070xt-porting"
t0 = time.time()
npz, rel_dirs, fp8, sz65536, dev_total, dev_py = [], [], [], [], 0, 0
skip = {".git", ".qwen", "node_modules"}
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in skip]
    if os.path.basename(dirpath) == "release":
        rel_dirs.append(dirpath)
    for f in filenames:
        p = os.path.join(dirpath, f)
        try:
            s = os.stat(p).st_size
        except OSError:
            continue
        if f.endswith(".npz"):
            npz.append(p)
        if f.endswith(".fp8"):
            fp8.append((p, s))
        if s == 65536:
            sz65536.append(p)
        if os.path.basename(dirpath) == "Development":
            dev_total += 1
        if f.endswith(".py"):
            dev_py += 1
print("elapsed_s %.1f" % (time.time() - t0))
print("dev_top_level_files", dev_total, "py_anywhere", dev_py)
print("npz_count", len(npz), npz[:3])
print("release_dirs", len(rel_dirs), rel_dirs[:3])
print("fp8_count", len(fp8))
for p, s in fp8[:12]:
    print("  fp8", s, os.path.relpath(p, ROOT))
print("size_65536_count", len(sz65536))
for p in sz65536[:12]:
    print("  64k", os.path.relpath(p, ROOT))
try:
    print("dev_files_recursive", sum(1 for _ in os.walk(os.path.join(ROOT, "Development"))),
          "total_files_recursive", sum(len(fs) for _, _, fs in os.walk(os.path.join(ROOT, "Development"))))
except Exception as exc:
    print("walk err", exc)
