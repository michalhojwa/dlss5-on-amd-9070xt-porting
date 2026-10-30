# Oracle for hip/gpu_matrix_op.h's gmx_e4m3_enc. Independent of the header's bit tricks:
# the reference is the FORMAT DEFINITION - nearest representable finite e4m3 value under
# round-half-to-even, saturating at 448. Any byte the header picks must equal this scan's pick.
# The scan runs over MAGNITUDES ONLY (bytes 0x00..0x7e); the sign is applied afterwards, so
# -0.0 cannot tie with +0.0 and flip the sign spuriously.
# Run: python xbox/oracle_e4m3.py
import struct, random, sys


def f32(x): return struct.unpack('<f', struct.pack('<f', x))[0]      # the header runs in float32
def bits(x): return struct.unpack('<I', struct.pack('<f', x))[0]


def decode(byte):                       # e4m3: sign, 4-bit exp bias 7, 3-bit mantissa
    s = -1.0 if byte & 0x80 else 1.0
    e, m = (byte >> 3) & 0xF, byte & 0x7
    # normalized (1+m/8)*2^(e-7) == (8+m)*2^(e-10); subnormal m*2^-9
    return s * (m * 2.0 ** -9 if e == 0 else (8 + m) * 2.0 ** (e - 10))


MAG = [b for b in range(0x80) if b != 0x7F]                       # magnitudes only; 0x7f = NaN


def reference(x):                       # nearest fp8, ties to even mantissa, saturate at 448
    if x != x: return 0x7F
    ax, best, bd = abs(x), None, float('inf')
    for b in MAG:
        d = abs(decode(b) - ax)
        if d < bd or (d == bd and (b & 1) == 0):   # tie -> even mantissa bit
            best, bd = b, d
    return best | (0x80 if bits(x) & 0x80000000 else 0)


# --- faithful transcription of the header's gmx_e4m3_enc --------------------------------
def header(x):
    b, a = bits(x), bits(x) & 0x7FFFFFFF
    s = (b >> 24) & 0x80
    if a >= 0x43E00000: return s | 0x7E
    if a < 0x3C800000:
        sc = f32(abs(x) * 512.0)
        q = int(sc)
        r = f32(sc - q)
        q += 1 if (r > 0.5 or (r == 0.5 and (q & 1))) else 0
        return s | (q & 31)
    r = (a + 0x7FFFF + ((a >> 20) & 1)) & 0xFFF00000
    e = ((r >> 23) & 0xFF) - 127
    return s | ((e + 7) << 3) | ((r >> 20) & 7)


def main():
    random.seed(20261028)
    vals = [random.uniform(-600, 600) for _ in range(12000)]
    vals += [random.uniform(-1e-3, 1e-3) for _ in range(6000)]        # subnormal band
    vals += [random.uniform(-1e6, 1e6) for _ in range(2000)]          # saturation band
    vals += [0.0, -0.0, 448.0, -448.0, 447.9, 449.0, 2.0 ** -9, 2.0 ** -10,
             2.0 ** -6, 1.0 / 512 * 15, 0.001953125]
    vals += [float(b) / 512.0 for b in range(0, 33)]                 # exact subnormal grid
    vals += [d for b in MAG for d in (decode(b), -decode(b))]       # every fp8 value, both signs
    vals += [(decode(b) + decode(b + 1)) / 2.0 for b in MAG[:120]]   # exact half-way ties
    bad = []
    for v in vals:
        h, r = header(v), reference(v)
        if h != r: bad.append((v, h, r))
    print("probes=%d mismatches=%d" % (len(vals), len(bad)))
    for v, h, r in bad[:12]:
        print("  x=%r header=0x%02x reference=0x%02x (decoded %.9g vs %.9g)" % (v, h, r, decode(h), decode(r)))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
