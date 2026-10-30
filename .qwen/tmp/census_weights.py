import json, struct, collections, pathlib, array, math

IDX = pathlib.Path(r'C:\kien\dlss5-on-amd-9070xt-porting\Development\weights-index.json')
BIN = pathlib.Path(r'G:\Steam\steamapps\common\Wuthering Waves\Client\Binaries\Win64\dlssnr_on_amd_weights.bin')

idx = json.loads(IDX.read_text(encoding='utf-8'))
print('index_records', len(idx))
print('dtype_code_x_storage', dict(collections.Counter((x.get('dtype_code'), x.get('inferred_storage')) for x in idx)))
print('bytes_per_element', dict(collections.Counter(x['payload_size'] // x['element_count'] for x in idx if x.get('element_count'))))
print('total_payload_bytes', sum(x['payload_size'] for x in idx))
print('element_count_total', sum(x.get('element_count', 0) for x in idx))
print('names_sample', [x['name'] for x in idx[:6]])
print('largest', sorted(idx, key=lambda x: -x['payload_size'])[:4])

f = BIN.open('rb')
magic = f.read(8)
count, hlen = struct.unpack('<II', f.read(8))
recs = []
while f.tell() < hlen:
    L = f.read(1)[0]
    name = f.read(L).decode('ascii', 'replace')
    off, size = struct.unpack('<QQ', f.read(16))
    recs.append((name, off, size))
ps = f.tell()
print('bin', magic, count, hlen, ps)
print('order_match', [r[0] for r in recs] == [x['name'] for x in idx])
print('size_match', [r[2] for r in recs] == [x['payload_size'] for x in idx])

def rec(i):
    name, off, size = recs[i]
    f.seek(ps + off)
    return name, f.read(size)

def half(u):
    s = -1.0 if u & 0x8000 else 1.0
    e = (u >> 10) & 0x1F
    m = u & 0x3FF
    if e == 0:
        return s * math.ldexp(float(m), -24)
    if e == 0x1F:
        return s * (float('inf') if m == 0 else float('nan'))
    return s * math.ldexp(float(m + 1024), e - 1025)

for i in (0, 1, 14, len(recs) - 1):
    name, raw = rec(i)
    sl = raw[:4096]
    n = len(sl) // 2
    vals = [half(u) for u in struct.unpack('<%dH' % n, sl[:n * 2])]
    finite = [v for v in vals if v == v and math.isfinite(v)]
    print(i, name, len(raw), 'halves', len(raw) // 2,
          'sampled', len(vals),
          'min', round(min(finite), 5), 'max', round(max(finite), 5),
          'mean', round(sum(finite) / len(finite), 6),
          'nan', sum(1 for v in vals if v != v), 'inf', sum(1 for v in vals if math.isinf(v)),
          'zeros', sum(1 for v in vals if v == 0.0))
    print('   head16hex', raw[:16].hex())
