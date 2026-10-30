import re
P = r"G:\Steam\steamapps\common\Wuthering Waves\Client\Binaries\Win64\dlssnr_amd_pass1.dll"
d = open(P, "rb").read()
for tag in (b"DLSSNRW1", b"block0.layer0.layer", b"WEIGHTS_HT"):
    ps = [m.start() for m in re.finditer(re.escape(tag), d, re.DOTALL)]
    print(tag, len(ps), [hex(p) for p in ps])
    for p in ps:
        print("  ctx", hex(p), "BEFORE", d[max(0, p - 24):p].hex(), "AFTER", d[p:p + 34].hex())
print("dll size", len(d))
