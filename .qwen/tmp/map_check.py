import json, re
from pathlib import Path
idx = json.loads(Path(r"C:\kien\dlss5-on-amd-9070xt-porting\Development\weights-index.json").read_text(encoding="utf-8"))
print("records", len(idx))
sizes = {}
for r in idx:
    sizes[r["payload_size"]] = sizes.get(r["payload_size"], 0) + 1
for s in (20672, 22720, 21696):
    print("payload_size", s, "-> count", sizes.get(s, 0))
print("--- names with size 20672 or 22720 ---")
for r in idx:
    if r["payload_size"] in (20672, 22720):
        print(r["index"], r["name"], r["payload_size"])
print("--- full name list (index, name, size) ---")
for r in idx:
    print(r["index"], r["name"], r["payload_size"])
