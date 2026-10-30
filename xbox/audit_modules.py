import glob, os, re, sys, json, hashlib
from collections import Counter

def audit(folder):
    hs = sorted(glob.glob(os.path.join(folder, '*.hsaco')))
    print('FOLDER', folder)
    print('hsaco', len(hs))
    tags = Counter()
    for f in hs:
        b = open(f, 'rb').read()
        m = re.findall(rb'gfx[0-9]{3}[0-9E]', b)
        tags[m[0].decode() if m else 'NONE'] += 1
    print('ISA tags', dict(tags))
    asm = glob.glob(os.path.join(folder, '*.hsaco.s'))
    print('assembly dumps', len(asm))
    print('generated.hip', len(glob.glob(os.path.join(folder, '*.generated.hip'))))
    sums = os.path.join(folder, 'SHA256SUMS')
    if os.path.exists(sums):
        lines = [l.split() for l in open(sums, encoding='utf-8', errors='replace').read().splitlines() if l.strip()]
        # SHA256SUMS is lowercase hex, modules.json uppercase: never compare hashes with .upper()
        bad = [l[-1] for l in lines if hashlib.sha256(open(os.path.join(folder, l[-1]), 'rb').read()).hexdigest() != l[0].lower()]
        print('SHA256SUMS lines', len(lines), 'hash mismatches', len(bad), bad[:5])
        names = set(l[-1] for l in lines)
        on = set(os.path.basename(f) for f in hs)
        print('in SPECT but no hsaco', sorted(names - on))
        print('hsaco not in SPECT', sorted(on - names))
    mj = os.path.join(folder, 'modules.json')
    if os.path.exists(mj):
        try:
            d = json.load(open(mj, encoding='utf-8', errors='replace'))
            print('modules.json entries', len(d) if isinstance(d, list) else sorted(d.keys()))
        except Exception as e:
            print('modules.json UNPARSEABLE', repr(e)[:120])
    bad = 0
    ok = 0
    err = 0
    for f in sorted(glob.glob(os.path.join(folder, '*.compile-log.txt'))):
        txt = open(f, encoding='utf-8', errors='replace').read()
        if 'OK hsaco=' in txt or 'bytes=' in txt:
            ok += 1
        if re.search(r'\berror\b|FAILED|undefined|invalid', txt, re.I):
            err += 1
            print('ERRORISH LOG', os.path.basename(f), repr(txt[:300]))
        if txt.strip():
            bad += 1
            print('LOG', os.path.basename(f), len(txt), repr(txt[:200]))
    print('non-empty compile logs', bad, 'compiler-OK logs', ok, 'error-ish logs', err)

audit(sys.argv[1])
if len(sys.argv) > 2:
    ref = sys.argv[2]
    print('--- size compare vs', ref)
    for f in sorted(glob.glob(os.path.join(sys.argv[1], '*.hsaco'))):
        g = os.path.join(ref, os.path.basename(f))
        a = os.path.getsize(f)
        b = os.path.getsize(g) if os.path.exists(g) else -1
        print('%-38s %8d  ref %8d  delta %+d' % (os.path.basename(f)[:-6], a, b, (a - b) if b >= 0 else 0))
