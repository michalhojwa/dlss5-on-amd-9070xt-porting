"""Read the gfx1151 build state from DISK, not from memory or a transcript.

  python xbox/check.py gates      # attempted / PASS / FAIL, and how many must port
  python xbox/check.py manifest   # SHA256SUMS vs *.hsaco vs the runtime's >=24 floor
  python xbox/check.py fails      # classify each failing recipe by its real compiler error

A third argument picks the arch folder, so the legacy set stays checkable:
  python xbox/check.py gates gfx1103

Buckets are PLAN's A/B/C/D/E (see xbox/PLAN.md "FPGA gate"). Anything not matching a bucket
prints UNCLASSIFIED with its first error line, so a wrong guess here is visible rather than
silently folded into a bucket.

On-disk naming (this build script): outputs are <module>.hsaco, logs are <module>.compile-log.txt,
and SHA256SUMS lists the FULL filename including .hsaco.
"""
import os, re, sys

ROOT = r"C:\kien\dlss5-on-amd-9070xt-porting"
MOD = os.path.join(ROOT, "hip", "modules", "gfx1151")
HSACO_SUFFIX = ".hsaco"
LOG_SUFFIX = ".compile-log.txt"

# Buckets are checked in order and the FIRST match wins, so they must not overlap.
# Matched against the whole log, not the displayed line (the comgr temp path eats the
# first ~70 characters of every error line, so a line-level grep misses the payload).
# Patterns are the strings the gfx1103 logs ACTUALLY contain, not guesses. First match wins.
BUCKETS = [
    ("A fp8 convert builtin absent", r"__builtin_amdgcn_cvt_pk_fp8_f32|__builtin_amdgcn_cvt_f32_fp8"
                                     r"|cvt_f16x2_from_f32x2|fp8e4m3|__fp8"),
    ("B mfma builtin absent",        r"__builtin_amdgcn_mfma|__builtin_amdgcn_wgmdf"),
    ("C gfx12-only WMMA",            r"wmma\w*_gfx12|wmdf\w*_gfx12|gfx12 FP8|gfx12 wmma"),
    ("D wave1 tiling guard",         r"wave1|tiled"),
]


def culprit(text):
    """The quoted identifier in the first error, or the static_assert message."""
    m = re.search(r"error:\s*'([^']+)'", text)
    if m:
        return "builtin/type: " + m.group(1)
    m = re.search(r'error:\s*"([^"]{4,90})', text)
    if m:
        return "assert: " + m.group(1)
    return "(no quoted payload)"

# the 13 recipes RuntimeOptions (:471-:479) can turn off when the module is absent
DROPPABLE = {"c32-wave1", "c32-wave1-rtz", "c32-wave1-fast", "c32-wave1-fast-norm900",
             "c64-wave2", "c64-wave2-fast", "c512-m32-mh", "c512-m32-deep",
             "swin-persistent", "swin-persistent-fast", "vit-stream", "vit-stream-fast",
             "vit-wide-deep"}


def state():
    files = os.listdir(MOD)
    hsaco_full = {f for f in files if f.endswith(HSACO_SUFFIX)}
    done = {f[:-len(HSACO_SUFFIX)] for f in hsaco_full}
    attempted = {f[:-len(LOG_SUFFIX)] for f in files if f.endswith(LOG_SUFFIX)}
    return done, attempted, hsaco_full


def fails():
    done, attempted, _ = state()
    for name in sorted(attempted - done):
        text = open(os.path.join(MOD, name + LOG_SUFFIX), errors="replace").read()
        label = next((lbl for lbl, pat in BUCKETS if re.search(pat, text, re.I)), "UNCLASSIFIED")
        print("%-32s %-36s %s%s" % (name, label, culprit(text),
                                    "  [DROPPABLE]" if name in DROPPABLE else ""))


def manifest():
    done, _, hsaco_full = state()
    lines = [ln for ln in open(os.path.join(MOD, "SHA256SUMS"), errors="replace").read().splitlines()
             if ln.strip()]
    listed = {ln.split()[-1] for ln in lines}
    print("SHA256SUMS lines : %d" % len(lines))
    print("hsaco on disk    : %d" % len(hsaco_full))
    absent = sorted(f for f in listed if f not in hsaco_full)
    unlisted = sorted(hsaco_full - listed)
    print("listed but absent: %d %s" % (len(absent), absent))
    print("on disk not listed: %d %s" % (len(unlisted), unlisted))
    print("floor 24         : %s" % ("OK" if len(lines) >= 24 else "SHORT BY %d" % (24 - len(lines))))
    print("manifest bytes   : %d (runtime cap is 1<<20 at LmxxfNrRuntime.cpp:298; ValidateModuleSet:337 and "
          "the has() fallback :465 both resolve names FLAT under modulesDir, so deploy flat)"
          % sum(len(ln) + 1 for ln in lines))


def gates():
    done, attempted, _ = state()
    failing = attempted - done
    print("attempted %d | PASS %d | FAIL %d | droppable-and-failing %d | must-port-or-reconfigure %d"
          % (len(attempted), len(done), len(failing), len(failing & DROPPABLE), len(failing - DROPPABLE)))
    print("PASS: " + ", ".join(sorted(done)))
    must = sorted(failing - DROPPABLE)
    print("NOT droppable (%d): %s" % (len(must), ", ".join(must)))


if __name__ == "__main__":
    if len(sys.argv) > 2:
        MOD = os.path.join(ROOT, "hip", "modules", sys.argv[2])
    {"fails": fails, "manifest": manifest, "gates": gates}.get(
        sys.argv[1] if len(sys.argv) > 1 else "gates", gates)()
