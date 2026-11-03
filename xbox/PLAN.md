# Xbox Ally X (Ryzen AI Z2 Extreme + NPU Strix [0065:00:01.1]) - DLSS5 NR enhancement port, ONNX / Windows ML route ONLY
# FOR CURRENT STATE READ xbox/STATE.md (overwritten daily). This file is the DURABLE DESIGN CONTRACT + GATES ONLY.
# Goal (UNCHANGED): apply Kien's DLSS5-NR network to the FINISHED (post-tonemap) frame of a running game as an
# enhancement (not upscaling), 720p output, network at 1-in-2 frames, game stays >= 30 fps.
# ROUTE (SETTLED 2026-11-04 BY USER): ONNX / Windows ML / XDNA2 IP engine. The hip + hsaco route is PARKED:
# read xbox/archive/HIP-HSACO-PARKED-2026-11-04.md if ever un-parked. Gate IDs A0-A6 / H1-H4 / P1-P4 / D1-D2 from
# that file are NOT current gates and must not be re-litigated here.

## Which silicon the work runs on (MEASURED 2026-11-04, not a preference)
  - `C:\Windows\System32\AMD\xrt-smi.exe examine --batch` -> Distribution Windows 11 Home, OS Release 26300.9818,
    Processor `AMD Ryzen AI Z2 Extreme`, Model `ROG Xbox Ally X RC73XA_RC73XA`, BIOS Version `RC73XA.319`,
    `Device(s) Present | [0065:00:01.1] | NPU Strix`.
  - `xrt-smi.exe validate --batch` -> `Validate Device [0065:00:01.1] / Platform NPU Strix / Power Mode Performance`,
    Test 1 gemm `TOPS: 51.3` [PASSED]; Test 2 latency `Average latency: 94.0 us` [PASSED]; Test 3
    `Average throughput: 51625.0 op/s` [PASSED]. `xrt-smi.exe --version` -> `Version 2.21.0`,
    `NPU Driver Version 32.0.20102.4480`, `NPU Firmware Version 1.1.2.64`.
  => An IP-engine (XDNA-class) target IS PRESENT AND PASSES AMD's own validation ON THIS BOX. Any older doc saying
    "XDNA is unreachable" is OVERTURNED: what was unreachable was ssh to amd9070 / rtx5090.
  - `C:\Windows\System32\AMD\` = 34 files: `xrt-smi.exe` (2,292,072 B), `pyxrt.pyd` (533,864 B), AIE2P kernel blobs
    (`1x4_*`, `4x4_*`, `5x4_*`, `_ipu_2`, `AMD_AIE2P_4x4_Overlay_*`), `amdafd/amdfendr/amdkmpfd`.
  - The hip/COMGR ISA space has 52 ISAs and NONE of them is XDNA (measured 2026-10-23 in this repo). That is the
    reason for the route change, not a style choice.

## Gates
[ ] N0. **Prove a graph actually executes on [0065:00:01.1].** This is the root gate; everything downstream is
    worthless until a run leaves a DEVICE-SIDE trace (an IP/AIE kernel report, `xrt-smi`-style log, or an ONNX
    Runtime profiler line naming an IP kernel). "It returned a tensor" is NOT proof - CPU fallback is silent.
    Candidate surfaces, each MEASURED as far as this box allows:
      * `pyxrt.pyd` + `xrt_coreutil.dll` (direct XRT, bypasses ONNX entirely): byte scan shows it links
        `python310.dll`; `import pyxrt` -> `ImportError: DLL load failed while importing pyxrt: Nie
        können odnaleźc określonego modułu.` Interpreter inventory: `NO_UV`, `NO_PY310_LAUNCHER`,
        `py --list` -> `-V:3.14[-64] * Python 3.14.7` only. BLOCKED on a CPython 3.10. INSTALLING ONE IS YOURS TO
        ASK FOR.
      * `xrt_coreutil.dll` C API from our own C++ (the repo already builds C++ DLLs): NO PYTHON DEPENDENCY;
        headers/SDK are NOT in `C:\Windows\System32\AMD` (only the runtime DLLs), so the import lib must be
        obtained - UNESTABLISHED, needs a download decision.
      * ONNX Runtime in-box: `C:\Windows\System32\onnxruntime.dll` 10,675,424 B; string scan finds
        `EP_COUNT 32` and a list with `DmlExecutionProvider`, `VitisAIExecutionProvider`, `OpenVINOExecutionProvider`,
        `CPUExecutionProvider`, `CUDAExecutionProvider`, `ROCMExecutionProvider`, `MIGraphXExecutionProvider`,
        `QNNExecutionProvider`; version context `Current ORT Version is: %s` + `1.17.1` and `ORT Build Info:
        git-branch=HEAD, git-commit-id=0`. Counter-evidence: `VitisAI`/`Vitis` occur only x2 each and the VitisAI
        config keys `config_file_path` / `perf_profile` occur 0 times, and the same scan shows no
        `vitis-ai-runtime.dll` import string - SO whether that EP is COMPILED IN IS UNESTABLISHED. A name in an EP
        list is NOT availability.
      * **RadeonML C API (`RadeonML*.dll` in System32) - STRONGEST LOCAL CANDIDATE, MEASURED 2026-11-04 by byte-scan
        (`dumpbin` is NOT installed here, so these are STRING hits, not a symbol table).** `RadeonML.dll` 633,704 B:
        error surface `RML_ERROR_BAD_MODEL` / `RML_ERROR_MODEL_NOT_READY` / `RML_ERROR_UNSUPPORTED_DATA`, env-ish
        `RML_LOG_LEVEL` `RML_TL_ENABLE`, `### Failed to open amodel file "`, namespace `.?AVrmlAmodel@0amd@@`, and the
        BACKEND ENUM `TF_CPU  TF_CUDA  IPU  TVM  AKS` plus versioned tags `U1 IPU2 IPU3 IPU4 ONNX TENSORFLOW_PB TVM2
        TVM3 TVM4`. `RadeonML_aks.dll` 3,729,264 B: the MODEL CALL SURFACE `rmlInfer(model=` / `rmlPrepareModel` /
        `rmlSetModelInput` / `rmlGetModelInputNames` / `rmlGetModelQOS` / `rmlSetModelDynamicParam` /
        `rmlGetModelInputScaleFactor` / `rmlResetModelStates`, the FILE SUFFIXES `.amodel` `.xmodel` `.onnxtxt`, ONNX
        PROTOBUF INTAKE (`onnx-ml.proto3`, `.onnx.NodeProto`, `.onnx.TensorProto`, ~28 `onnx.*` descriptors, build
        path `W:\src\RadeonML\third_party\onnx\onnx\onnx-ml.pb.cc`), and the DEVICE-PLANNING MESSAGES `IPU device
        found` / `IPU not available` / `IPU Platform 1`..`IPU Platform 4` / `IPU Platform Unknown`. `RadeonML_ipu.dll`
        3,733,864 B: 38 `IPU` hits incl. `IPU.Context Index=` and `Unsupported ONNX data type:`. => AN ONNX-INTAKE
        backend WITH AN IP-engine target IS SHIPPED IN THIS IMAGE. UNESTABLISHED: whether those `rml*` names are EXPORTED
        (a string list is not a symbol table) and WHERE THE HEADERS COME FROM - AMD's RadeonML repo is a DOWNLOAD
        DECISION, ASK.
      * AMD IP-engine runtime in-box: `vitis-ai-runtime.dll` 7,378,792 B + `vitis-ai-runtime2.dll` 12,029,800 B. A
        scan of `vitis-ai-runtime.dll` finds the KERNEL PATH `S.y.s.t.m.3.2.\.A.M.D` + `AMD_AIE2P_4x4_Overlay_CFG0_3.5.0.0-2229_ipu_2.xclbin`
        and links `xrt_coreutil.dll` (x2, plus `XrtContext::~X`). => IT IS THE COMPONENT THAT FEEDS THE AIE2P KERNEL
        BLOBS ALREADY SITTING IN `C:\Windows\System32\AMD`. It contains NO `Vitis` and NO `onnxruntime` string.
      * Windows ML: `Windows.AI.MachineLearning.dll` 1,196,848 B + `winml.dll` 65,536 B + `Windows.AI.MachineLearning.Preview.dll`
        + `directml.dll` 10,239,288 B. The 2026-10-29 "a THIRD-PARTY MODEL MUST BE SIGNED/ATTESTED" claim is STILL
        UNVERIFIED, and today's scan of `Windows.AI.MachineLearning.dll` found NO `attestation` / `Attestation` /
        `signature` / `EnableModelSignatureValidation` / `XVA` / `VitisAI` marker (its 4 `Signature` hits are D3D *root
        signature* strings, `winml::D3DDeviceCache::GetTensorizeRootSignature` - UNRELATED). A 0-hit scan is NOT A
        VERDICT: TEST IT BY LOADING AN UNSIGNED MODEL.
    FIRST ACTION FOR N0 (no new installs): `C:\Windows\System32\AMD\xrt-smi.exe` has only `configure|examine|validate`
    (`xrt-smi --help` captured 2026-11-04 - NO sample-model command), so N0 cannot be closed by xrt-smi alone.
    The minimum self-contained test is a tiny hand-authored ONNX graph (a MatMul + Add with f32/f16 initializers)
    run through `C:\Windows\System32\onnxruntime.dll` via its C API (ctypes or a C++ exe), with the provider list read
    BACK FROM the session (`GetAllRegisteredEpList` / `Session::GetProviders`) - that settles the VitisAI question
    above definitively, which a string scan cannot. Then repeat against the RadeonML surface if the ORT build has no
    IP EP. If a pip install (`onnx`, `onnxruntime`) is wanted: PyPI REACHES (`PYPI_STATUS 200 BYTES 645387`) but that is
    a SYSTEM CHANGE - ASK.
    PRIOR ART FOUND 2026-11-04: `C:\Windows\System32\amd_taau_ops.rml_plugin.dll` 521,576 B carries the debug path
    `W:\src\RadeonML\build_out\x64\Release\Release\amd_taau_ops.rml_plugin.pdb` and namespace symbol
    `.?AVException@details@taau@rml@@`. => AMD SHIPS A TAAU (temporal-AA) OP PLUGIN IN THE RadeonML PLUGIN NAMESPACE ON
    THIS IMAGE, i.e. a GAME-ENHANCEMENT NEURAL PIPELINE ALREADY RUNS THROUGH THIS STACK. Read it as the TEMPLATE for
    D1's plugin layout and for N6's hook shape; it is NOT NR and gives no weights.
[ ] N1. **Weight initializers.** In-box asset (READ ONLY): `G:\Steam\steamapps\common\Wuthering Waves\Client\Binaries\
    Win64\dlssnr_on_amd_weights.bin` = 147,689,451 B, magic `b'DLSSNRW1'`, u32le `0x99` = 153 records, `header_len
    0x1629` = 5673, per record `[u8 name_len][name][u64 payload_offset][u64 payload_size]`. Slice with the container's
    OWN header - `Development/weights-index.json` offsets are ARCHIVE-ABSOLUTE with growing drift. DTYPE MIXED
    (E4M3 matrix bodies + f16 bias/skip/slots + one f4 scale): NEVER decode a whole payload as halves.
    What the container does NOT give: tensor SHAPES. Shapes must come from the architecture model (N2) - deriving
    them by dividing byte counts is INVENTING MAPPINGS. `Development/weights-arena-index.json` (25,809 B) is a dict
    under key `records`. Container is NOT in the repo (single copy in the game install): keep it out of git;
    record size + a checksum instead.
[ ] N2. **Architecture -> ONNX graph.** Source of truth = `Development/network-graph.json` (39,952 B, 1,682 lines: `blocks[]`
    with `block_type`/`layer_type`/`width`/`inputs`/`weight_records` + top-level `sample_sha256`; its own `boundary`
    key says verbatim "Block types, order, and block-to-block edges are recovered; block 0 external texture bindings
    remain unresolved") + `Development/HIP/hip_reference_network.h` (177,256 B, 1,093 lines) +
    `Development/history/reverse-engineering-notes.md` (13,169 B, 241 lines) + the 41-module composition and its 6-kernel
    contract (expand/qkv/scores/av/project/ffn, MACs 256/1024/256/256/512/256 per tile as the contract, on-disk ALU
    census 262/1027/262/267/526/268 — those numbers are in `xbox/archive/HANDOFF-2026-10-23_to_27.md:404/406/488`
    and `xbox/archive/HIP-HSACO-PARKED-2026-11-04.md:48/122`). The graph must be authored FROM THOSE NOTES - NVIDIA
    ships NO ONNX for NR (`nvngx_dlssnr.dll` = 15 `sm_120` .cubin + the weight container above). In-repo ONNX PRIOR
    ART is the AGENT's ViT back chain + the mask survey, NOT NR: `Development/history/amd-port-plan.md:200` and
    `:242`, `Development/results/character-mask-survey-20261011/skin-and-parsing.md:13` (BiSeNet-R18: author ships PT
    AND ONNX, ~11.2M params/43MB, fixed 512x512, dynamic batch, opset20) and `:23` ("do not infer full-GPU coverage
    from a model loading"), `full-person.md:19/29/31/39` — DOCUMENT-ONLY, chain CLOSED by user decision.
    CORRECTION (self-inflicted, fixed 2026-11-04): an earlier draft of this file cited
    `Development/results/2026-10-29-preblock-ffn-reverse-engineering/DESIGN-NOTES.md` and
    `Development/hip-reference-network.md` and quoted "2,840,120 params". MEASURED: none of those paths is in
    `git ls-files` (`git grep 840,120` hits only our own drafts) and `skin-and-parsing.md` is 39 lines, so the line
    cites were impossible. Do not re-propagate those three citations.
[ ] N3. **Op partition + fallback census.** Which ONNX ops the IP path actually takes, which fall back. Report per-op
    from the runtime's own profiler output; a silent CPU fallback is a FAILURE, not a result.
[ ] N4. **Numerical parity.** Reference outputs still come from NVIDIA silicon (the .cubin oracle), so the pivot does
    NOT remove the CUDA-host dependency: `dir /b /s *.cubin` -> `NO_CUBIN`; the 21 `CUDA-ORACLE` front-chain maps
    and the 39 referenced release/** .npz (0 on disk, grade_tally {NO-PRODUCER 14, CUDA-ORACLE 21, ANALYTIC 4,
    NEEDS-BYTE-BLOBS 1}, the 14 OVERSTATES - argparse `a.folder` producers are parser misses; the third item this file
    previously named, `Development/build_c32_oracle_artifacts.py:91`, IS NOT A PATH IN THIS REPO (see N2 CORRECTION) and
    the real `view/mapping.npz` producers are `Development/recover_native_c64_view.py:50` and
    `Development/recover_native_split_view.py:36`) remain CUDA-oracle-bound. Do NOT hand-write a mapping - "do not invent mappings".
[ ] N5. **Perf model on the NPU.** Measured ceiling inputs: gemm `TOPS: 51.3`, `51625.0 op/s`, latency `94.0 us`
    (Power Mode Performance). Convert the NR MAC count through that and compare against 1-in-2 frames at the 720 tier.
    Power Mode on battery is a DIFFERENT number - measure both before quoting a budget.
[ ] N6. **Host hook (was H1-H4).** Swap the hip-loader path in `src/LmxxfNrRuntime.cpp` for an ONNX/Windows-ML session:
    module selection is BY NAME (`Development/HIP/hip_reference_network.h:424-425`, `src/LmxxfProductionOptions.h:18`
    forces `o.wmma=true`; none of the 71 `DLSS5_HIP_*` flags in `src/native_hip_env_options.h` touches wmma/wave/tiled/
    pooled/packed_weights/fp8), so wiring the new backend is a SOURCE EDIT there, not an env var. Then: finished frame ->
    network (history OFF) -> blend; 1-in-2 skip; reduced model_scale tier.
[ ] D1. Installable layout (addon64 or injector + the runtime DLL + the .onnx + weight initializers + noise).
[ ] D2. Config: model_scale, frame-skip, strength, skip-blocks; per-game deploy.

## Non-negotiables (fixed, do not relitigate)
  - Inputs stay E4M3-quantized (F()), and every E4M3 value is exact in F16. Products reach 448^2 = 2.0e5 > f16
    max 65504, so the ACCUMULATOR MUST BE f32. f16 accumulation is a correctness bug, and an IP-engine kernel type
    does NOT change this rule - verify the ONNX graph's accumulation dtype explicitly.
  - A PASSING COMPILE / exit code 0 / "the session was created" is NOT proof the design landed. Verification claims
    need the EXACT COMMAND LINE + VERBATIM OUTPUT, otherwise write NOT RUN / UNVERIFIED.
  - No device probe (a process touching the NPU) without an explicit go-ahead - ASK FIRST.
  - Never `git add .`; never revert / `git restore` / `git clean` uncommitted or untracked artifacts; the tree holds
    one-of-a-kind build outputs.

## Host facts to re-MEASURE, not trust (this file is a contract, not a log)
  - `xrt-smi.exe` on this box: `--version` `Version 2.21.0`; commands are ONLY `configure | examine | validate`
    (captured help 2026-11-04). No sample/model runner.
  - Interpreter: `py --list` -> `-V:3.14[-64] * Python 3.14.7`; `py -3.10` -> `NO_PY310_LAUNCHER`; `NO_UV`;
    packages `onnxruntime False, onnx False, torch False, numpy True, sklearn False`. numpy 2.5.4 lives at
    `C:\Users\micha\AppData\Local\Python\pythoncore-3.14-64\python.exe`.
  - `dumpbin` is NOT installed (`'dumpbin' is not recognized as an internal or external command`) - read DLL deps by
    byte-scanning for `*.dll` names instead.
  - Network reaches PyPI (`PYPI_STATUS 200 BYTES 645387`); installs are YOUR decision.
  - No reachable ssh host: `C:\Users\micha\.ssh\config` absent, amd9070 100.83.122.75 times out, rtx5090
    100.102.178.207 refused. A CUDA host is required for N4 (previously used: RTX 5090, DGX Spark `spark-3a10`).

## Git (history is the archive, not appended prose)
  - Repo root = this directory. `origin` = https://github.com/michalhojwa/dlss5-on-amd-9070xt-porting.git ; branch
    `xdna2_poc` tracks `origin/xdna2_poc`. Run everything from the repo root; NO HARDCODED DRIVE PATHS (the tree
    moved C:\kien -> C:\kien2; check.py derives ROOT from __file__, build_target.ps1 walks up for hip\build-modules.ps1,
    rtc_compile_cmd.cmd uses %~dp0).
  - **2026-11-04 POLICY CHANGE (USER):** commit AND PUSH REGULARLY - at least once a day, or after each bigger unit of
    finished work, whichever comes first. Check remote state first with `git ls-remote origin xdna2_poc` (measured
    2026-11-04: remote at `5bc57b02`, local ahead 1). This SUPERSEDES the older "push only when asked".
  - `git add <the exact paths you touched>` then `git commit`. NEVER `git add .` / `-A`: build logs, hip/modules/**,
    rtc_compile.exe are already TRACKED, so a blanket add sweeps unrelated churn into the commit. Do NOT add new large
    binaries (history already carries hip/rtc_compile.exe at 2,682,656 B and hip/modules/gfx1103/*.hsaco.s at
    10,830,841 B; `git count-objects -vH` -> `size-pack: 831.25 MiB`).
  - NEVER `git clean -xfd` and NEVER `git checkout --`: the .hsaco / logs ARE deliverables and are not re-creatable
    cheaply. core.autocrlf=input: index LF, worktree CRLF - that display is normal, status stays clean.
  - SEARCH WITH `git grep -n` - the Grep tool SILENTLY RETURNS 0 MATCHES on this repo. Trackedness: `git ls-files` /
    `git check-ignore --no-index -v <path>` (plain check-ignore reports NOTHING for an already-TRACKED path).
  - Superseded text lives in history + `xbox/archive/`: pre-2026-11-04 hip contract is filed at
    `xbox/archive/HIP-HSACO-PARKED-2026-11-04.md`; days older than 2026-10-30 are under `xbox/archive/`
    (HANDOFF-2026-10-28_to_29.md, pre-git/STATE.md). Old `HANDOFF.md:NNN` citations point into those archive files.
