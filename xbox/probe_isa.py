# Compile-only ISA availability battery. Answers "does this builtin/spelling survive
# COMGR for this ISA" - it does NOT prove device behaviour (no execution here).
#   python xbox/probe_isa.py [target] [name ...]
# Output is one line per probe; error text is reduced to the quoted builtin +
# the target-feature clause the front end reports.
import os, subprocess, sys, re

BOX = os.path.dirname(os.path.abspath(__file__))
HIP = os.path.join(BOX, os.pardir, 'hip', 'rtc_compile.exe')
EXE_TARGETS = {'gfx1103', 'gfx1200', 'gfx1201'}
# rtc_compile.exe is frozen at 16.10.2026 and cannot be rebuilt here (see probe_comgr.py),
# so gfx115x goes through ctypes COMGR instead. Both paths are compile-only.
CT_TARGETS = {'gfx1150', 'gfx1151', 'gfx1152', 'gfx1153', 'gfx115E'}
TARGETS = EXE_TARGETS | CT_TARGETS


def digest(text):
    m = re.search(r"error: '([^']*)' needs target feature ([^\n]*)", text)
    if m:
        return "needs [%s]" % m.group(2).strip()
    m = re.search(r"error: (.*)", text)
    if m:
        return m.group(1).strip()[:110]
    return "no error line"


def isa_ops(path_s):
    if not os.path.exists(path_s):
        return "no .s"
    ops = [ln.split()[0] for ln in open(path_s, 'r', errors='replace')
           if re.match(r"\s*(v_wmma|v_mfma|v_mma|v_dot|global_store_b32)\S*\s", ln)]
    tally = {}
    for o in ops:
        tally[o] = tally.get(o, 0) + 1
    if not tally:
        return "no matrix ops"
    return ",".join("%s x%d" % (k, v) for k, v in sorted(tally.items()))


def main():
    args = sys.argv[1:]
    target = args[0] if args and args[0] in TARGETS else 'gfx1103'
    names = [a for a in args if a not in TARGETS] or ['probe_fp8', 'p_wmma16', 'p_wmma32',
              'p_wmma_f16x8', 'probe_wmma_f16', 'probe_mfma_f16']
    for n in names:
        src = os.path.join(BOX, n + '.hip')
        if not os.path.exists(src):
            print("%-22s SKIP (no source)" % n)
            continue
        out = os.path.join(BOX, 'out_%s_%s.hsaco' % (n, target))
        if target in EXE_TARGETS:
            r = subprocess.run([HIP, out, src, 'comgr', target], capture_output=True, text=True,
                               errors='replace', timeout=180)
            text = (r.stdout or '') + (r.stderr or '')
            tag = 'PASS' if r.returncode == 0 else 'FAIL'
            note = digest(text) if tag == 'FAIL' else isa_ops(out + '.s')
        else:
            sys.path.insert(0, BOX)
            from probe_comgr import load, compile_hip
            c = load()
            status, logtext, hsaco, assembly = compile_hip(
                c, open(src, 'r', errors='replace').read(), n + '.hip', target)
            tag = 'PASS' if status == 0 else 'FAIL'
            if assembly:
                open(out + '.s', 'wb').write(assembly)
            if hsaco:
                open(out, 'wb').write(hsaco)
            note = digest(logtext) if tag == 'FAIL' else isa_ops(out + '.s')
        print("%-22s %s %s" % (n, tag, note))


if __name__ == '__main__':
    main()
