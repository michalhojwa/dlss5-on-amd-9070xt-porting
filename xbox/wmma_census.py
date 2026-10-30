"""Which top-level HIP sources call which WMMA builtin, with what fragment types.

  python xbox/wmma_census.py            # spelling tally + per-file table
  python xbox/wmma_census.py --files    # only files that call a _gfx12 spelling

Scans hip/ top level only (*.hip *.inc *.h); hip/modules/** are GENERATED and are
derived from these, so counting them twice would inflate every total.
"""
import os, re, sys, collections

HIP = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'hip')
EXTS = ('.hip', '.inc', '.h')
SPLIT = re.compile(r'__builtin_amdgcn_wmma_\w+')


def main():
    args = sys.argv[1:]
    files_only = '--files' in args
    tally = collections.Counter()
    rows = []
    for fn in sorted(os.listdir(HIP)):
        p = os.path.join(HIP, fn)
        if not os.path.isfile(p):
            continue
        if os.path.splitext(fn)[1] not in EXTS:
            continue
        text = open(p, errors='replace').read()
        spellings = collections.Counter(m.group(0) for m in SPLIT.finditer(text))
        if not spellings:
            continue
        for k, v in spellings.items():
            tally[k] += v
        types = {}
        for m in re.finditer(r'using\s+(\w+)\s*=\s*([^;]+);', text):
            types[m.group(1)] = m.group(2).strip()
        rows.append((fn, spellings, types))
    print('distinct spellings: %d' % len(tally))
    for k, v in tally.most_common():
        print('%5d  %s' % (v, k))
    print('\nfiles: %d' % len(rows))
    for fn, spellings, types in rows:
        if files_only:
            if any('_gfx12' in k for k in spellings):
                print(fn)
            continue
        print('\n%s' % fn)
        for k, v in spellings.items():
            print('  %4d %s' % (v, k))
        frag = {n: t for n, t in types.items() if re.search(r'_Float16|__fp16|float|int', t)
                and re.search(r'vextor_type|vector_size', t)}
        if frag:
            print('  fragments: ' + ', '.join('%s = %s' % (n, t) for n, t in sorted(frag.items())))


if __name__ == '__main__':
    main()
