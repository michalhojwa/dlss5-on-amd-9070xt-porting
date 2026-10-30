"""Per-kernel ISA census for a HIP/ROCm .hsaco.s dump (LLVM asm syntax).

  python isa_report.py <path.hsaco.s>                # census + metadata per kernel
  python isa_report.py <path.hsaco.s> --grep PAT --ctx 4

Naming traps learned on gfx1103 (this COMGR 3.0 emits RDNA spelling):
  LDS ops are ds_load_* / ds_store_*   (NOT ds_read_* / ds_write_*)
  FMA is v_fmac_f32_e32 / v_dual_fmac_f32  (NOT v_fma_f32; v_dual_* = 2 MACs each)
  Kernel names ending in _wmma make `grep v_wmma` match labels and `.set <name>.num_vgpr`
"""
import re, sys, collections

KERNEL_RE = re.compile(r'^([A-Za-z_][A-Za-z0-9_\.]*):\s*(;.*)?$')
INST_RE = re.compile(r'^[\t ]+([a-z][a-z0-9_\.]*)[\t ]')
SET_RE = re.compile(r'^[\t ]*\.set\s+\S+\.(\w+),\s*(\d+)')
DIR_RE = re.compile(r'^[\t ]*\.(amdgcn_target|amdhsa_\w+|amdgcn_amdhsa_\w+)\s+(.*)')

FAMILIES = {
    'fma(MACs)': (re.compile(r'^(v_dual_fmac|v_fmac|v_dual_fma|v_fma)_'), 2),
    'ds(LDS)':   (re.compile(r'^ds_'), 1),
    'global':    (re.compile(r'^global_'), 1),
    'scratch':   (re.compile(r'^scratch_'), 1),
    'cvt_f16':   (re.compile(r'^v_cvt_f16|^v_cvt_f32_f16'), 1),
    'sbarrier':  (re.compile(r'^(barrier|s_barrier)$'), 1),
}
# the ~40 "noise" ops that are bookkeeping, not math
NOISE = re.compile(r'^(s_delay_alu|s_waitcnt|s_mov_b32|v_mov_b32|v_dual_mov_b32|'
                   r's_cbranch|barrier|v_cndmask|s_mov|v_or|s_or|s_xor|s_and|s_lshl)')


def main():
    path = sys.argv[1]
    grep, ctx = None, 4
    args = sys.argv[2:]
    while args:
        if args[0] == '--grep':
            grep, args = args[1], args[2:]
        elif args[0] == '--ctx':
            ctx, args = int(args[1]), args[2:]
        else:
            args = args[1:]

    lines = open(path, 'r', errors='replace', encoding='utf-8').read().splitlines()

    if grep:
        pat = re.compile(grep)
        hits = [i for i, ln in enumerate(lines) if pat.search(ln)]
        print('%d hits for %r' % (len(hits), grep))
        for shown, i in enumerate(hits):
            if shown >= 12:
                print('  ... (%d more)' % (len(hits) - shown)); break
            lo, hi = max(0, i - ctx), min(len(lines), i + ctx + 1)
            print('--- line %d ---' % (i + 1))
            for j in range(lo, hi):
                print('%s%6d| %s' % ('>' if j == i else ' ', j + 1, lines[j]))
        return

    starts = [i for i, ln in enumerate(lines)
              if KERNEL_RE.match(ln) and not ln.startswith('.L')]
    starts = [i for i in starts if 'hip' not in lines[i]]
    print('file lines: %d   kernels: %d' % (len(lines), len(starts)))
    bounds = starts + [len(lines)]
    for k, s in enumerate(starts):
        seg = lines[s:bounds[k + 1]]
        name = lines[s].split(':')[0]
        c = collections.Counter()
        meta = {}
        for ln in seg:
            m = INST_RE.match(ln)
            if m:
                c[m.group(1)] += 1
            st = SET_RE.match(ln)
            if st:
                meta[st.group(1)] = int(st.group(2))
            d = DIR_RE.match(ln)
            if d and 'segment' in d.group(1):
                meta[d.group(1)] = d.group(2)
        print('\n=== %s  lines %d-%d ===' % (name, s + 1, bounds[k + 1]))
        print('  %-24s %-20s %-18s flat_scratch=%s'
              % ('num_vgpr=%s' % meta.get('num_vgpr', '?'),
                 'num_sgpr=%s' % meta.get('numbered_sgpr', '?'),
                 'priv_seg=%s' % meta.get('private_seg_size', '?'),
                 meta.get('uses_flat_scratch', '?')))
        print('  LDS: ' + ', '.join('%s x%d' % (n, v) for n, v in
                                    sorted((n, v) for n, v in c.items() if n.startswith('ds_'))))
        fam = []
        for f, (pat, mult) in FAMILIES.items():
            tot = sum(v * mult for n, v in c.items() if pat.match(n))
            if tot:
                fam.append('%s=%d' % (f, tot))
        print('  ' + '  '.join(fam))
        mathops = {n: v for n, v in c.items() if not NOISE.match(n)}
        top = sorted(mathops.items(), key=lambda kv: (-kv[1], kv[0]))
        print('  non-bookkeeping: ' + ', '.join('%s x%d' % (n, v) for n, v in top[:18]))


if __name__ == '__main__':
    main()
