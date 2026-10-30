import json, struct, pathlib

BIN = pathlib.Path(r'G:\Steam\steamapps\common\Wuthering Waves\Client\Binaries\Win64\dlssnr_on_amd_weights.bin')
IDX = pathlib.Path(r'C:\kien\dlss5-on-amd-9070xt-porting\Development\weights-index.json')

idx = json.loads(IDX.read_text(encoding='utf-8'))
print('index_top_type', type(idx).__name__)
if isinstance(idx, dict):
    print('index_keys', sorted(idx.keys())[:8])
    recs_i = idx.get('records') or idx.get('entries')
    base = idx.get('base') or idx.get('archive_offset')
else:
    recs_i = idx
    base = None
print('index_records', len(recs_i))
print('index_first', json.dumps(recs_i[0]))
print('index_base_field', base)

f = BIN.open('rb')
magic = f.read(8)
count, hlen = struct.unpack('<II', f.read(8))
recs_b = []
while f.tell() < hlen:
    L = f.read(1)[0]
    name = f.read(L).decode('ascii', 'replace')
    a, b = struct.unpack('<QQ', f.read(16))
    recs_b.append((name, a, b))
payload_start = f.tell()
size = BIN.stat().st_size
print('bin_magic', magic)
print('bin_count', count, 'bin_header_len', hlen, 'bin_parsed', len(recs_b), 'payload_start', payload_start, 'bin_size', size, 'payload_bytes', size - payload_start)

print('field_shape_check', [(n, a, b) for n, a, b in recs_b[:3]])

def keyof(r):
    for k in ('name', 'record', 'id'):
        if k in r:
            return r[k]
    return None

def off(r):
    for k in ('payload_offset', 'offset'):
        if k in r:
            return r[k]
    return None

def siz(r):
    for k in ('payload_size', 'size'):
        if k in r:
            return r[k]
    return None

names_i = [keyof(r) for r in recs_i]
names_b = [r[0] for r in recs_b]
print('name_order_identical', names_i == names_b)
print('only_in_index', sorted(set(names_i) - set(names_b))[:10])
print('only_in_bin', sorted(set(names_b) - set(names_i))[:10])

BASE = 0x114a160
bad_off = bad_size = 0
off_deltas = set()
size_deltas = []
for r, (n, a, bb) in zip(recs_i, recs_b):
    oi, si = off(r), siz(r)
    off_deltas.add(a - (oi + BASE if BASE is not None else oi))
    if si != bb:
        bad_size += 1
    if oi + BASE != a:
        bad_off += 1
print('offset_field_deltas', sorted(off_deltas)[:6], 'bin_vs_index_offset_mismatches', bad_off, 'size_mismatches', bad_size)

# is bin payload contiguous in index order?
cum = 0
contig = True
for (n, a, bb) in recs_b:
    if a != BASE + cum and a != cum:
        pass
    cum += bb
print('sum_bin_sizes', cum, 'bin_payload_bytes', size - payload_start, 'equal', cum == size - payload_start)
