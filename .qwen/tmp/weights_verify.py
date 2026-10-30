import json, struct, pathlib, math, hashlib, os

IDX = pathlib.Path(r'C:\kien\dlss5-on-amd-9070xt-porting\Development\weights-index.json')
BIN = pathlib.Path(r'G:\Steam\steamapps\common\Wuthering Waves\Client\Binaries\Win64\dlssnr_on_amd_weights.bin')
ROOT = pathlib.Path(r'C:\kien\dlss5-on-amd-9070xt-porting')

idx = json.loads(IDX.read_text(encoding='utf-8'))
f = BIN.open('rb')
f.read(8)
count, hlen = struct.unpack('<II', f.read(8))
recs = []
while f.tell() < hlen:
    L = f.read(1)[0]
    name = f.read(L).decode('ascii', 'replace')
    off, size = struct.unpack('<QQ', f.read(16))
    recs.append((name, off, size))
ps = f.tell()

def half(u):
    s = -1.0 if u & 0x8000 else 1.0
    e = (u >> 10) & 0x1F
    m = u & 0x3FF
    if e == 0:
        return s * math.ldexp(float(m), -24)
    if e == 0x1F:
        return s * (float('inf') if m == 0 else float('nan'))
    return s * math.ldexp(float(m + 1024), e - 25)

print('--- float16 sanity (corrected exponent) ---')
for i in (0, 1, 14, 51, len(recs) - 1):
    name, off, size = recs[i]
    f.seek(ps + off)
    sl = f.read(min(size, 4096))
    vals = [half(u) for u in struct.unpack('<%dH' % (len(sl) // 2), sl)]
    fin = [v for v in vals if v == v and math.isfinite(v)]
    absmax = max(abs(v) for v in fin)
    print(i, name, size, 'sampled', len(vals),
          'min', round(min(fin), 5), 'max', round(max(fin), 5), 'absmax', round(absmax, 5),
          'nan', sum(1 for v in vals if v != v), 'zeros', sum(1 for v in vals if v == 0.0))

print('--- sha256 of each blob payload vs repo .f16/.f32/.fp8 files ---')
payload_hashes = {}
for name, off, size in recs:
    f.seek(ps + off)
    h = hashlib.sha256()
    remaining = size
    while remaining:
        chunk = f.read(min(remaining, 1 << 20))
        h.update(chunk)
        remaining -= len(chunk)
    payload_hashes[(name, size)] = h.hexdigest()

hits = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    for fn in filenames:
        ext = os.path.splitext(fn)[1].lower()
        if ext in ('.f16', '.f32', '.fp8', '.bin'):
            p = pathlib.Path(dirpath) / fn
            if p.stat().st_size not in {s for _, s in payload_hashes}:
                continue
            hh = hashlib.sha256(p.read_bytes()).hexdigest()
            for (name, size), ph in payload_hashes.items():
                if ph == hh:
                    hits.append((str(p), name, size))
print('byte_identical_matches', len(hits))
for t in hits[:20]:
    print('  ', t[0], '==', t[1], t[2])

big = sorted(payload_hashes.items(), key=lambda kv: -kv[1])[:3]
print('largest_three_hashes', [(n, s, h[:16]) for (n, s), h in big])
