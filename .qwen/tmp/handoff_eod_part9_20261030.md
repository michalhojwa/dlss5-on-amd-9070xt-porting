
## END OF DAY 2026-10-30 (part 9 - A6 ARENA PREREQUISITE CLOSED ON THIS BOX; THE `input` ACTIVATION AND NUMPY REMAIN THE ONLY BLOCKERS)

Re-derived at 10:01 with `python C:\kien\dlss5-on-amd-9070xt-porting\xbox\check.py gates && python C:\kien\dlss5-on-amd-9070xt-porting\xbox\check.py manifest`:
`attempted 41 | PASS 41 | FAIL 0 | droppable-and-failing 0 | must-port-or-reconfigure 0`,
`SHA256SUMS lines : 41`, `hsaco on disk : 41`, listed-but-absent 0, on-disk-not-listed 0,
`floor 24 : OK`, `manifest bytes : 3638`. NO SOURCE IS NEWER THAN THE OUTPUTS, SO NO REBUILD WAS RUN.

### What was new: the aligned ARENA is reproducible on this box, from the container the runtime already uses

`Development/vit_blocks31_38_reference.py:20` takes POSITIONAL `arena index input output`, and
`Development/vit_blocks31_38_reference.py:21` resolves payloads as
`arena[arena_offset : arena_offset + payload_size]` for `block{N}.layer{L}.layer`.
The committed `Development/weights-arena-index.json` keys are ONLY
`records` (153) / `alignment` = 512 / `arena_size` = 147,719,680.

`Development/parse_weights_archive.py:111 build_aligned_arena`, with `:141 parser.add_argument("dll")`,
`:143 --arena` (help: "write the 512-byte-aligned FP16 arena") and the `--arena-json` flag, is the
project's OWN writer of that index. BUT `parse_weights_archive.py:19 def load_pe_helpers` ->
`:20 helper_path = Path(__file__).parents[1] / "296" / "analyze_dlssnr.py"`, and that file DOES NOT EXIST
ANYWHERE (`glob **/analyze_dlssnr.py` = 0 hits; there is no `296/` in the repo). So the project's
own arena BUILDER cannot run here as-is. The module IMPORTS FINE though (numpy-free; the PE
helper is only loaded inside `main()`), which is what I used.

MEASURED RECONSTRUCTION (scratch `.qwen/tmp/a6_arena_from_container_20261030.py`, run twice, verbatim):
```
container size 147689451 magic b'DLSSNRW1' u16@8 (153, 5673)
container records parsed 153 closed at EOF False
index records 153 alignment 512 arena_size 147719680
total container payload 147683778
total index payload     147683778
name order mismatches 0 []
payload_size mismatches 0 []
rebuilt arena size 147719232 matches index arena_size False
arena_offset mismatches 0 []
aligned_size mismatches 0 []
WROTE C:\kien\dlss5-on-amd-9070xt-porting\.qwen\tmp\weights-arena-from-container-20261030.bin 147719680
```
So walking `dlssnr_on_amd_weights.bin` with the framing
header(16 B) + a 5,673-byte name/offset TABLE (`[u8 name_len][name][u64 off][u64 size]`) + the payload blob,
then 512-aligning the 153 payloads in archive order, reproduces `weights-arena-index.json`
with ZERO mismatches on name order, payload_size, arena_offset and aligned_size for all 153 records.
The only delta is a TRAILING PAD: the arena is 147,719,232 after the last payload and the index's
`arena_size` is 147,719,680, i.e. 448 zero bytes of end-padding. Pad to that and the file is
exactly `arena_size`. (`closed at EOF False` is CORRECT, not a parse failure: the TABLE ends around
0x1629 = 5,673 and the payload blob follows, so the table is what the framing closes, not the file.)

SLICE SIZES CONFIRM THE INDEX IS THE RIGHT INPUT (from the index itself):
`block31.layer1.layer` payload_size 4,196,352 = 4096*1024 Contract + the 2,048-byte f16 skip tail EXACTLY;
`block31.layer4.layer` 1,050,624 = 1024*1024 Projection + 2,048 skip tail EXACTLY;
`block31.layer0.layer` 4,194,320 >= the `[:1024*4096]` Expand slice. Same pattern at block38
(layer0/1/2/3/4 = 4,194,320 / 4,196,352 / 3,145,856 / 2 / 1,050,624).

A DEAD END WORTH RECORDING: the DLL the project's arena builder expects
(`parse_weights_archive.py:141 parser.add_argument("dll")`) needs an embedded WEIGHTS_HT PE RESOURCE,
and I checked whether any on-box binary carries one.
`WEIGHTS_HT` occurs 2x in `dxgi.dll` and 2x in each of `dlssnr_amd_pass1/2/3.dll`, but dumping
context shows those are LOADER STRINGS, not an archive: `WEIGHTS_HT resource not found`,
`cannot read the DLL`, `WEIGHTS_HT resource %.1f MB`, and a neighbouring
`DLSSNR_CHAIN=1 (the Phi...`. `dxgi.dll` has ZERO occurrences of `block0.layer0.layer`;
the one occurrence inside `dlssnr_amd_pass1.dll` sits in an ASCII NAME TABLE
(`...er11.layer0.layer\0block0.layer0.layer\0block70.layer0...`), not a weight archive.
=> There is NO embedded WEIGHTS_HT ARCHIVE on this box. The arena comes from the AMD CONTAINER.

### The two blockers that are still real

1. `input` is now NAMED. `Development/vit_block31_reference.py:12 unswizzle` does
   `np.frombuffer(...,np.uint8)` -> `e4()` -> a 64-token x 1024-channel FP32 array, and the reference
   reads `a.input.read_bytes()[:65536]` with token bits `[2,6,7,8,14,15]` / channel bits
   `[0,1,3,4,5,9,10,11,12,13]`. So `input` is a 65,536-byte PHYSICAL E4M3 activation. `Development/`
   has NO such file: scanning every Development file >= 65,536 B for a non-zero extent inside
   65,000..66,000 returned only `block14-downsample-matrix.bin` (exactly 65,536 B, a WEIGHT not an
   activation) and two TEXT files under `Development/results/history-contract-20261006/`.
   `Development/block30-enter-512x1024.bin` is 2,097,152 B with 972,460 non-zero bytes spread to
   EOF, so it is a DENSE activation and NOT the physical blob. The name the project itself uses for
   it is `release/native-rgb512/vit-input.fp8` (Development/history/porting-worklog.md:2936), which
   is absent along with the rest of the `release/**` scatter maps.
2. NUMPY is still a HARD BLOCKER for both `vit_blocks31_38_reference.py:4` and
   `unpack_vit_matrices.py`. Interpreter `C:\Users\micha\AppData\Local\Python\pythoncore-3.14-64\python.exe`
   3.14.7. NOT installed - that is an environment change and the standing rule is to ASK first.

### Mined lead, now corroborated rather than speculative

`Development/history/porting-worklog.md` 1,244-1,256 read verbatim (relevant sentences):
`vit_blocks31_38_reference.py` "将八层串联后, block31->38 std 从 1.95 平滑降至 1.10, all block finite
no NaN/saturation"; Attention's semantic form is "32 main tokens + 32 zero padding keys, logit scale
1", kernel correlation 0.9627453 / MAE 0.280474, RX9070XT submit->fence 1.557 ms; a controlled
Attention impulse showed 64 exact `1/64` outputs and a per-bit on/off matrix proving Q/K low-5 dim
bits and high-5 head bits are identity, with K->V key-token bit map `[1,0,3,4,2]` and V physical
token bits `[1,0,4,5,2]`; "work FP16 is internal scratch, not another 32 tokens". It also states
the FINAL STREAKS are NOT a precision criterion: exact block31-34 prefix replacement left them
almost unchanged, and final acceptance requires the original block70 post. Use these as the
baseline to compare against, not as device validation.

### Net A6 verdict (honest)

`arena` NOW EXISTS ON THIS BOX (rebuilt, byte-exact to the committed index on all 153 records).
`index` EXISTS. All `--assets` inputs EXIST with sizes matching the code's slices.
`input` DOES NOT EXIST (65,536-B physical E4M3 activation; the project's own name is
`release/native-rgb512/vit-input.fp8`). NUMPY DOES NOT EXIST. So the Vi-T ORACLE is
BLOCKED ON EXACTLY TWO THINGS - a captured activation and numpy - and NOT on the weights at all.
Still UNVERIFIED: any Vi-T numeric output on this box. Do not state otherwise.

RULE REINFORCED: import the repo's OWN parser instead of hand-grepping. `build_aligned_arena`
reproduced the committed index with 0 mismatches; my earlier hand-derived claim that the arena
"cannot be produced on this box" was WRONG in a useful direction - it CAN, and now HAS BEEN.
