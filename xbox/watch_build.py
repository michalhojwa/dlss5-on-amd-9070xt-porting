import glob, os, sys, time

p = sys.argv[1]
limit = float(sys.argv[2]) if len(sys.argv) > 2 else 1800.0
t0 = time.time()
last = -1
while True:
    n = len(glob.glob(os.path.join(p, '*.hsaco')))
    if n != last:
        newest = max((os.path.getmtime(f) for f in glob.glob(os.path.join(p, '*'))), default=0)
        print('%3d hsaco  newest %5.0fs ago  elapsed %5.0fs' % (n, time.time() - newest, time.time() - t0), flush=True)
        last = n
    if time.time() - t0 > limit:
        print('WATCH_TIMEOUT hsaco=%d' % n, flush=True)
        break
    time.sleep(15)
