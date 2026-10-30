import os

NEEDLE = ('nvngx', 'dlssnr', 'dlss')
for root_dir in (r'G:\Steam\steamapps\common\Wuthering Waves', r'C:\kien\dlss5-on-amd-9070xt-porting'):
    print('=== walk', root_dir)
    for r, ds, fs in os.walk(root_dir, onerror=lambda e: None):
        for f in fs:
            p = os.path.join(r, f)
            try:
                s = os.path.getsize(p)
            except OSError:
                continue
            if s > 100_000_000 or any(k in f.lower() for k in NEEDLE):
                print(f'{s:>12} {p}')
