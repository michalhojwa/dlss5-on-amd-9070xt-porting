# Xbox Ally X (gfx1103 = legacy build, gfx1151 = chosen tag; NEITHER is the measured device arch) - DLSS5 NR enhancement port, ALU-first strategy
# FOR CURRENT STATE READ xbox/STATE.md (~100 lines, overwritten daily). This file is the DURABLE
# DESIGN CONTRACT + GATES ONLY; read the named section, not the whole file.
# Goal: apply Kien's DLSS5-NR network to the FINISHED (post-tonemap) frame of a
# running game as an enhancement (not upscaling), on the Ally X GPU, at 720p
# output, network at 1-in-2 frames, keeping the game at >= 30 fps.

## Which silicon the "ALU" work runs on (settled, not a preference)
  - hip/* compiles through amd_comgr_3.dll into ISA `amdgcn-amd-amdhsa--gfx1103` (52 ISAs
    listed, none of them XDNA) and runs on the APU RDNA-class shader cores under D3D12/HIP.
  - Measured gfx1103 (xbox/p_*.hip -> xbox/out_*.hsaco[.s]):
      wave32; v_fma_f32; v_pack_b32_f16 + v_pk_fma_f16; v_wmma_f32_16x16x16_f16.
      ABSENT: mai-insts (no MFMA), fp8-conversion-insts, gfx12-insts (no *_gfx12 WMMA),
      dot4a/vdot4a/usdot4a in this COMGR clang.
  - XDNA1/XDNA2 is the Copilot-class "Series 2" AI-engine (NPU) IP line: not addressable
    from HIP/D3D12 compute, not a rtc_compile target, so no .hip can land on it. If a real
    XDNA path is ever required it leaves this repo (ONNX/NPU toolchain), not hip/ + hsaco.
  - So "ALU" here means: **ALU-first kernels** - no matrix instruction at all, only the
    general-purpose vector ALU (v_fma_f32, optional 2-way v_pk_fma_f16).
  - 2026-10-29 ARCH CORRECTION (do not read "gfx1103" above or below as the device's arch): the Ally X APU is Ryzen AI
    Z2 Extreme = RDNA 3.5, which this COMGR spells gfx1150/1151/1152/1153/115E (AMD names gfx1151 for the Ryzen AI Max
    series); gfx1103 is a gfx110x/RDNA3 id, and RDNA4 (gfx1200/1201) is RULED OUT by that mapping. MEASURED compile-only
    on this box: all gfx115x are CAPABILITY-IDENTICAL to gfx1103 - fp8 converts rejected, the UNSUFFIXED
    v_wmma_f32_16x16x16_f16 the only accepted WMMA op, inline asm rejected identically - so the shim and the CW_PACK_MODE_MASK
    0 flip REMAIN REQUIRED and the w32 f16 slot-layout risk is NOT resolved by the arch swap. hip\modules\ now holds
    BOTH builds; which gfx115x the silicon reports is STILL UNESTABLISHED and neither set has run on a device. That
    mapping is DOCUMENT INFERENCE (AMD ROCm's arch->ISA table), not a device measurement - A0 stays open.

## Gates
[!] A0. Confirm the device's real arch string - BLOCKED 2026-10-26, not answerable on this box.
    native_hip_network.h:65 LogDevice writes adapter_name + architecture to logs/native-hip-device.txt
    only in the process that creates the HIP bridge (the host running the network); no such log exists
    in the repo or in the reference install, and amd_comgr_3.dll here can emit gfx1103 but not run it.
    If the Ally X APU is RDNA4 (gfx1200/gfx1201) the in-tree fast kernels need no port at all - capture
    that line from the device the next time anything runs there. (RULED OUT 2026-10-29 by AMD's RDNA3.5->gfx115x
    mapping, which is DOCUMENT INFERENCE - only a device run can settle it.)
    2026-10-29: what changed is the ASSUMED side only. USER chose the gfx1151 TAG; hip\build-modules.ps1:13's ValidateSet
    now accepts gfx1150/1151/1152/1153/115E, the gfx115*-scoped shim inline (:117-119) and __has_builtin guard rewrite
    (:141-143) fire, and hip\modules\gfx1151 holds a 41-module build that xbox\check.py:20 gates BY DEFAULT (3rd arg =
    gfx1103); both folders print 41 attempted / 41 PASS / 0 FAIL in xbox\gates_run.txt. Gates moved, NOTHING about the
    device did: WHICH SET TO DEPLOY is an OPEN DECISION (FLAT either way - ValidateModuleSet resolves SHA256SUMS names
    FLAT at src/LmxxfNrRuntime.cpp:337, floor found<24 at :345), a wrong tag failing at LOAD is UNTESTABLE on this box,
    and the device's exact gfx115x id is UNMEASURED - ASK BEFORE CREATING OR RUNNING ANY DEVICE PROBE.
[x] A1. gfx1103 ISA capability map - DONE (see measured list above; HANDOFF 2026-10-23)
[x] A2. Module survey - 2026-10-27 recount from disk (hip/modules/gfx1103): 41 recipes attempted,
    8 PASS (c32_prefix_reference, multihead-reference, deep_reference, boundary_reference,
    wave-pointwise, multi-pass-skin, multi-pass-predict, c32_alu) / 33 FAIL. The earlier
    "37 / 7 / 30" was a per-file count, not the on-disk truth; count *.compile-log.txt vs *.hsaco.
    2026-10-29 SUPERSEDED: with -ExtraDefines 'CW_PACK_MODE_MASK 0' the same 41 recipes all COMPILE - 41 attempted /
    41 PASS / 0 FAIL in hip\modules\gfx1103 AND hip\modules\gfx1151. The 8/33 split is the PRE-flip state, and a PASS
    is still ONLY a compile - no output has ever been read back.
[ ] A3. Runtime floor: >= 24 hsaco listed in SHA256SUMS AND on disk (src/LmxxfNrRuntime.cpp:333
    per-line existence check, :345 found<24). 2026-10-27: the 24 is NOT the real constraint -
    hip_reference_network.h:762 LoadModules() requests modules BY OPTION, and
    src/LmxxfProductionOptions.h forces o.wmma=o.wave=o.tiled=o.pooled=true plus the whole
    fast_*/fused_*/fp8_* set. RuntimeOptions (:471-:479) only turns OFF 5 optional families
    when their module is absent (wave_owned, c512_m32, swin, vit_stream, vit_proj_n64 = 13 recipes).
    So c32_wmma + multihead-wmma + deep_wmma are UNCONDITIONAL under opt.wmma and MUST port, and
    every other forced-true flag needs either its fast twin ported or the option re-profiled to the
    reference fallback that RunGraph already has (e.g. Run("c32","dlss5_prefix_reference",...)).
    Also: hip/modules/gfx1103/SHA256SUMS currently lists ONLY c32_alu.hsaco - the last -Only build
    rewrote it while 8 hsaco remain on disk. Regenerate before any device test.
    2026-10-29: that drift is GONE and the floor is MET x2 - hip\modules\gfx1103 AND hip\modules\gfx1151 each list
    41 SHA256SUMS lines = 41 .hsaco on disk, 0 absent / 0 unlisted, floor 24 OK, manifest 3638 bytes
    (xbox\check.py manifest / xbox\audit_modules.py re-derive it). STILL TRUE: deploy FLAT - an arch SUBFOLDER breaks
    the flat name resolution, so do NOT add arch folders to the runtime's module list.
[~] A4. ALU tile primitive + pilot module - 2026-10-26: hip/c32_alu.hip written to this design,
    registered as a gfx1103-only row, compiles clean to hip/modules/gfx1103/c32_alu.hsaco
    (sha D50F1CE9...). 2026-10-27: the ISA IS ON-CONTRACT; the 2026-10-26 off-contract reading was
    a measurement error from wrong opcode names (this COMGR spells LDS as ds_load_*/ds_store_* and
    FMA as v_fmac_f32_e32; see HANDOFF). Per-kernel MAC census now 262/1027/262/267/526/268 vs the
    256/1024/256/256/512/256 contract, and ds_store counts 32/128/32/32/64/32 = exactly the staged
    tile per chunk. What is LEFT of A4: num_vgpr=256 (the cap) + uses_flat_scratch=1 + private_seg
    48/132/144/160 on expand/qkv/scores/project (contract=129 and av=111 are clean), and 143
    ds_load_2addr_b32 in expand = per-lane LDS addressing that failed to vectorise into b128.
    2026-10-29: numbers RE-MEASURED with xbox\isa_report.py and reproduce; the escalated reading is that the vgpr cap is
    SPILLING TO GLOBAL SCRATCH (flat_scratch=1 plus scratch ops 15/16/44/20 on the SAME four kernels, while contract and
    av have none). Re-census only if hip\c32_alu.hip changes.
[ ] A5. Roll the twins out across the required set (prefix_fast, c32_fast, c32_fast_attention,
    multihead_wmma, multihead-fast*, boundary-fast, deep_fast*, c512-m32-*, vit-*,
    swin-persistent, c32-wave1*, c64-wave2*), or drop the droppable fast twins.
    2026-10-29: SATISFIED AT COMPILE LEVEL ONLY - under the shim + CW_PACK_MODE_MASK 0 all 41 recipes build in BOTH
    folders. No output has been verified. Reaching the reference twins is NOT an env var: none of the 71 DLSS5_HIP_* flags
    in src\native_hip_env_options.h touches wmma/wave/tiled/pooled/packed_weights/fp8_*, selection is BY NAME at
    Development\HIP\hip_reference_network.h:424-425 (an `accelerated` set of 20 kernel names; opt.wmma true appends
    _wmma) and src\LmxxfProductionOptions.h:18 FORCES o.wmma=true, so a twin run needs a SOURCE EDIT there. The USER
    chose DOCUMENT-ONLY, so the defaults are UNTOUCHED - if that changes, say so explicitly.
[ ] A6. Weights: the ALU path consumes plain block*-*.f16 / .f32 + noise.f32, NOT the
    HIP_PREPACKED_WEIGHTS fp8 blobs (gfx12 fast-path asset). Source: HANDOFF G4 / WW install.
    2026-10-29: SOURCE FOUND on this box - G:\Steam\steamapps\common\Wuthering Waves\Client\Binaries\Win64\dlssnr_on_amd_weights.bin
    (147,689,451 B, READ ONLY), self-describing: b'DLSSNRW1' + u32le 153 + header_len 5673, per record [u8 name_len][name][u64
    payload_offset][u64 payload_size]. Same 153 names/sizes as Development\weights-index.json, but that JSON's payload_offset
    is ARCHIVE-ABSOLUTE with gaps - slice with the container's OWN header, not the JSON. DTYPE IS MIXED (E4M3 matrix bodies
    + f16 bias/skip/slots + one f4 scale), so NEVER decode a whole payload as halves. Gate stays [ ] because the release/**
    .npz slot maps are still MISSING (zero .npz in the tree) and c512 / split-head / post70 provenance past block22 is untraced
    - do NOT invent mappings.

## Numerics rules for the ALU path (fixed, do not relitigate)
  Inputs stay E4M3-quantized (F()), and every E4M3 value is exact in F16. Products reach
  448^2 = 2.0e5 > f16 max 65504, so the accumulator MUST be f32 (v_fma_f32).
  v_pk_fma_f16 is only allowed behind an explicit scale + measurement, never as default.
  Integer dot products are out (no DP4A in this compiler). Reference kernels remain the
  bit-exact oracle; ALU kernels are validated ON DEVICE, not on this box.

## Cost model
  One 16x16 tile at K=32 = 8192 MACs = 256 v_fma_f32 per lane in a 32-lane wave - the same
  MAC count the WMMA path performs, moved onto the ALU. Network every other frame at the
  720 tier (~2 x 16.7 ms per networked frame); DLSS5_SKIP_BLOCKS (lossy) only if
  measurement says so.

## Pilot design (A4) - decided: NEW TWIN MODULE, not an in-file #if
  hip/c32_wmma.hip is wave-pure (no barriers, no LDS). Adding LDS there means changing a
  shipping gfx12 recipe, so the ALU path goes in a NEW FILE hip/c32_alu.hip exporting the
  SAME 6 kernel names and the SAME grid/work-group contract, registered as a gfx1103-only
  module in hip/build-modules.ps1 ($modules row, sources = ('c32_alu.hip')). gfx1200/gfx1201
  recipes stay byte-identical.
    Helper: DEV float8 alu_matrix32(a,b,a0,astr,akstr,rowcap,b0,bstr,bkstr)
      A(row,k) = a[a0 + row*astr + k*akstr]  (zero when row >= rowcap)
      B(col,k) = b[b0 + col*bstr + k*bkstr]
      acc[e] = dot over k=0..31, output slot = (row=group*8+e, col=rc), rc=lane&15, group=lane>>4
      LDS: __attribute__((shared)) float sa[16*32], sb[32*16]  (4096 B), staged cooperatively
      (32 lanes x 16 elements each), sync with the file idiom at c32_fast.hip:56
      (fence(3,"workgroup"); __builtin_amdgcn_s_barrier(); fence(2,"workgroup")) or the
      LDS-scoped WG_FENCE pair from c32_fused_ffn_attention.hip:90.
    Site -> argument table (all 6 kernels of c32_wmma.hip):
      c32_ffn_expand_wmma   alu(input,   weights, first*32,            32,1, tokens-first, 512+col*32,   32,1)
      c32_ffn_contract_wmma 4x chunk: alu(hidden, weights, first*128+chunk*32, 128,1, tokens-first, 4608+col*128+chunk*32, 128,1); residual seed path unchanged
      c32_attn_qkv_wmma     alu(input,   weights, first*32,            32,1, -1, part*1024+col*32, 32,1)
      c32_attn_scores_wmma  alu(normalized, normalized, (w*64+qr)*32,  32,1, -1, (tokens+w*64+kr)*32, 32,1)
      c32_attn_av_wmma      2x chunk: alu(prob, qkv, (w*64+qr)*64+chunk*32, 64,1, -1, (2*tokens+w*64+chunk*32)*32+col, 32,32); acc[e]=H(acc[e]+sum[e])
      c32_attn_project_wmma alu(av,      weights, first*32,            32,1, -1, 3072+col*32, 32,1)

## Host / runtime (apply to finished frame)
[ ] H1. Rebuild LmxxfNrRuntime.dll for gfx1103 (assets dir -> gfx1103 hsaco, arch gate)
[ ] H2. Host hook: finished post-tonemap frame -> codec -> network (history OFF) -> blend
[ ] H3. 1-in-2 frame skip (network every other frame, hold/blend on the other)
[ ] H4. Reduced model_scale (540/720 tier) to fit the 30 fps budget

## Perf + verification
[ ] P1. Per-frame network GPU ms on gfx1103 at 720 tier; target < ~16 ms avg with skip
[ ] P2. If short: DLSS5_SKIP_BLOCKS (lossy) / tile shrink, then re-measure
[ ] P3. In-game A/B: 720p output, 1-in-2, strength control, confirm >= 30 fps

## Deliverable
[ ] D1. Installable layout: dlss5-amd.addon64 (or injector) + LmxxfNrRuntime.dll
    + DLSS5-AMD assets (gfx1103 dir: SHA256SUMS, >= 24 hsaco, plain weights, noise.f32)
[ ] D2. Config: model_scale, frame-skip, strength, skip-blocks; per-game deploy
