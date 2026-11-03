# DLSS5-NR ON XBOX ALLY X - STATE CARD (end of 2026-11-04, ONNX / WINDOWS ML ROUTE)
READ FIRST EVERY SESSION. OVERWRITTEN daily, never appended to, kept near 100 lines.
xbox/RESUME.txt = the paste block; xbox/HANDOFF.md = HISTORY ONLY (newest END OF DAY = part 13 below the 2026-11-04
banner; everything pre-2026-11-04 there is HIP-ERA and PARKED); xbox/PLAN.md = the DURABLE CONTRACT + GATE LADDER N0..N6/D1/D2.
THE HIP + hsaco ROUTE IS PARKED BY USER DECISION: its full last-known contract is filed verbatim at
xbox/archive/HIP-HSACO-PARKED-2026-11-04.md (19,512 B). Gate IDs A0-A6 / H1-H4 / P1-P4 from that file are NOT current.
Today's machine/build-options report: xbox/REPORT-2026-11-04-machines-build-options-xdna2.md.

STATUS: THE PIVOT IS DOCUMENTED AND THE ROUTE IS NOW MEASURED FEASIBLE ON THIS BOX. No model has run on the NPU yet:
gate N0 is still OPEN, so NO CLAIM below says "works on device". What changed vs. the hip era is decisive and measured
(2026-11-04): THIS BOX HAS AN IP-ENGINE (XDNA-class) TARGET AND AMD'S WHOLE IP RUNTIME STACK LOCALLY. The hip route was
dead-ended because COMGR's 52 ISAs contain NO XDNA ISA; the ONNX route does not need COMGR at all.

SILICON (MEASURED 2026-11-04, OVERTURNS older "XDNA unreachable" text):
  `C:\Windows\System32\AMD\xrt-smi.exe examine --batch` -> Processor `AMD Ryzen AI Z2 Extreme`, Model `ROG Xbox Ally X
  RC73XA_RC73XA`, BIOS `RC73XA.319`, `Device(s) Present | [0065:00:01.1] | NPU Strix`.
  `xrt-smi.exe validate --batch` -> `Validate Device [0065:00:01.1] / Platform NPU Strix / Power Mode Performance`:
  Test 1 gemm `TOPS: 51.3` [PASSED]; Test 2 `Average latency: 94.0 us` [PASSED]; Test 3 `Average throughput: 51625.0
  op/s` [PASSED]. `xrt-smi.exe --version` -> `Version 2.21.0`, `NPU Driver Version 32.0.20102.4480`, `NPU Firmware
  Version 1.1.2.64`. `xrt-smi --help` -> commands are ONLY `configure | examine | validate` (NO sample-model runner),
  so N0 CANNOT be closed by xrt-smi. "unreachable" was ever only ssh: C:\Users\micha\.ssh\config ABSENT, amd9070
  100.83.122.75 TIMES OUT, rtx5090 100.102.178.207 REFUSED.

GATE LADDER STATUS (see PLAN.md for the definitions; one line each here)
  [ ] N0 PROVE A GRAPH RUNS ON [0065:00:01.1] with a DEVICE-SIDE TRACE. OPEN - root gate, do everything else after it.
      CPU FALLBACK IS SILENT, so "it returned a tensor" proves nothing; a real close needs an IP/AIE kernel line or
      the session's provider list read back. Four surfaces MEASURED as far as this box allows (details in PLAN.md):
        * RadeonML (System32) = STRONGEST LOCAL CANDIDATE. `RadeonML.dll` 633,704 B holds the BACKEND ENUM `TF_CPU
          TF_CUDA IPU TVM AKS` + tags `IPU2 IPU3 IPU4 ONNX TENSORFLOW_PB`; `RadeonML_aks.dll` 3,729,264 B holds the
          MODEL CALL SURFACE (`rmlInfer(model=`, `rmlPrepareModel`, `rmlSetModelInput`, `rmlGetModelInputNames`,
          `rmlGetModelQOS`, `rmlSetModelDynamicParam`, `rmlResetModelStates`), the suffixes `.amodel` `.xmodel`
          `.onnxtxt`, ONNX protobuf intake (`onnx-ml.proto3`, ~28 `onnx.*` descriptors, `W:\src\RadeonML\third_party\onnx\onnx\onnx-ml.pb.cc`)
          and the device-planning messages `IPU device found` / `IPU not available` / `IPU Platform 1..4`;
          `RadeonML_ipu.dll` 3,733,864 B has 38 `IPU` hits incl. `Unsupported ONNX data type:`. => AN ONNX-INTAKE
          BACKEND WITH AN IP target IS SHIPPED IN THE IMAGE. UNESTABLISHED: exported-ness (these are STRING hits;
          `dumpbin` is NOT installed here) and where the headers come from (AMD's RadeonML repo = a download, ASK).
        * ONNX Runtime in-box: `C:\Windows\System32\onnxruntime.dll` 10,675,424 B; `EP_COUNT 32` and an EP name list
          containing `DmlExecutionProvider`, `VitisAIExecutionProvider`, `OpenVINOExecutionProvider`, `CPUExecutionProvider`,
          `CUDAExecutionProvider`, `ROCMExecutionProvider`, `MIGraphXExecutionProvider`, `QNNExecutionProvider`;
          `Current ORT Version is: %s` + `1.17.1`. COUNTER-EVIDENCE (measured 2026-11-04): `AppendExecutionProvider_VitisAI`
          0 HITS while `AppendExecutionProvider_DML` is PRESENT, and `config_file_path` / `perf_profile` /
          `provider_options` / `vitis-ai-runtime` all 0 hits. => VitisAI EP availability UNESTABLISHED, likely absent in
          THIS build; ONLY `GetAllRegisteredEpList()` / `Session::GetProviders()` AT RUNTIME settles it.
        * pyxrt direct XRT: BLOCKED. `import pyxrt` (sys.path += C:\Windows\System32\AMD) ->
          `ImportError: DLL load failed while importing pyxrt: Nie kann...` (Czech locale for "module not found"); it
          links `python310.dll`; `py --list` -> `-V:3.14[-64] * Python 3.14.7` ONLY, `NO_PY310_LAUNCHER`, `NO_UV`.
          Installing 3.10 is a SYSTEM CHANGE - ASK before doing it.
        * Windows ML: `Windows.AI.MachineLearning.dll` 1,196,848 B + `winml.dll` 65,536 B + `directml.dll` 10,239,288 B.
          The 2026-10-29 "THIRD-PARTY MODELS MUST BE SIGNED/ATTESTED" claim is STILL UNVERIFIED; today's scan found NO
          `attestation` / `signature` / `EnableModelSignatureValidation` / `XVA` marker in that DLL (4 `Signature` hits
          are D3D *root signature* strings - UNRELATED). 0-HIT IS NOT A VERDICT: TEST BY LOADING AN UNSIGNED MODEL.
  [ ] N1 weight initializers from the container (147,689,451 B, magic `b'DLSSNRW1'`, 153 records) - UNCHANGED by the pivot.
  [ ] N2 author the ONNX graph from the reverse-engineered architecture (NVIDIA SHIPS NO ONNX FOR NR) - OPEN.
  [ ] N3 op partition + fallback census - OPEN, needs N0 closed.
  [ ] N4 numerical parity - OPEN and STILL CUDA-ORACLE-BOUND: `dir /b /s *.cubin` -> `NO_CUBIN`, no reachable CUDA host
      from this box (RTX 5090 / DGX Spark `spark-3a10` were the only oracle hosts). The pivot did NOT remove this
      dependency; it only removed the hsaco-compile job.
  [ ] N5 perf model vs `TOPS: 51.3` / `51625.0 op/s` / `94.0 us` - OPEN; measure Power Mode on BATTERY separately.
  [ ] N6 host hook = a SOURCE EDIT (module selection is BY NAME; none of the 71 `DLSS5_HIP_*` env flags touches it).
  [ ] D1 installable layout, [ ] D2 config - OPEN.

NEW ASSET FOUND 2026-11-04 (PRIOR ART, NOT NR, NO WEIGHTS): `C:\Windows\System32\amd_taau_ops.rml_plugin.dll` 521,576 B
  carries debug path `W:\src\RadeonML\build_out\x64\Release\Release\amd_taau_ops.rml_plugin.pdb` and namespace
  `.?AVException@details@taau@rml@@`. => AMD ALREADY RUNS A TAAU (temporal-AA) NEURAL OP PLUGIN THROUGH THIS STACK ON
  THIS IMAGE. Read it as the TEMPLATE for D1's plugin layout and N6's hook shape.
COMPONENT MAP (sizes measured 2026-11-04, all `C:\Windows\System32`): `vitis-ai-runtime.dll` 7,378,792 B + `vitis-ai-runtime2.dll`
  12,029,800 B (the first one strings the KERNEL PATH `S.y.s.t.m.3.2.\.A.M.D` + `AMD_AIE2P_4x4_Overlay_CFG0_3.5.0.0-2229_ipu_2.xclbin`
  and links `xrt_coreutil.dll` => it is what FEEDS the AIE2P kernel blobs already in System32\AMD; it contains NO `Vitis`
  and NO `onnxruntime` string), `RadeonML_AKS_TF.dll` 22,385,528 B, `RadeonML_AKS_TVM.dll` 12,979,912 B,
  `RadeonML_DirectML.dll` 1,795,440 B, `Windows.AI.Agents.dll` 384,320 B, `Windows.Internal.AI.PlatformCapability.dll`
  195,584 B, `WSAIFabricHost.dll` 839,680 B, `MLEngineStub.exe` 167,200 B, `AIComponentMgmt.dll` 179,288 B.
  System32\AMD holds the AIE2P kernel blobs (`1x4_*`, `4x4_*`, `5x4_*`, `_ipu_2`, `AMD_AIE2P_4x4_Overlay_*`) +
  `xrt-smi.exe` 2,292,072 B + `xrt_coreutil.dll` + `pyxrt.pyd` 533,864 B.

HOST / TOOLING STATE (MEASURED 2026-11-04): `python` = 3.14.2 `C:\Program Files\python314\python.exe`; packages
  numpy 2.5.4 TRUE, onnxruntime FALSE, onnx FALSE, torch FALSE, sklearn FALSE. `pip --version` -> NOT RECOGNIZED (use
  `python -m pip` if ever needed, ASK first). `gh --version` -> NOT RECOGNIZED, so a NEW GitHub repo cannot be created
  from here. `dumpbin` NOT INSTALLED => read DLL deps by byte-scanning for `*.dll`. Scripts: multi-line `python -c`
  returns EMPTY EXIT 0 under cmd.exe - write real .py files under C:\Users\micha\.qwen\tmp\.

GIT (MEASURED 2026-11-04): repo `C:\kien2\dlss5-on-amd-9070xt-porting`, branch `xdna2_poc` tracks
  `origin/xdna2_poc`; HEAD `3edf7aca` "docs(xbox): 2026-11-04 report; hip+hsaco parked to archive; PLAN.md rewritten
  for ONNX/Windows ML"; remote `git ls-remote origin xdna2_poc` -> `5bc57b02`, LOCAL AHEAD 1.
  POLICY CHANGED 2026-11-04 BY USER: COMMIT AND PUSH REGULARLY (>= once a day / after each bigger unit). Still: NEVER
  `git add .` or `-A` (build outputs are TRACKED), NEVER `git clean -xfd` / `git checkout --` (the .hsaco files and logs
  are one-of-a-kind deliverables), SEARCH WITH `git grep -n` (the Grep tool FALSE-ZEROES on this repo), old git REJECTS
  `:(x)` pathspec magic, NO `tail`/`head`/`more` (redirect to a file and read_file it). History size: `git count-objects
  -vH` -> `size-pack: 831.25 MiB`.

STANDING RULES (unchanged): a PASSING COMPILE / exit 0 / "session created" is NOT PROOF; verification claims carry the
  EXACT COMMAND LINE + VERBATIM OUTPUT else write NOT RUN / UNVERIFIED; NEVER hand-write a weight→shape mapping ("do
  not invent mappings"); the ACCUMULATOR MUST BE f32 (448^2 = 2.0e5 > f16 max 65504) regardless of kernel type; NO
  DEVICE PROBE (any process touching the NPU) and NO INSTALL without an explicit go-ahead - ASK FIRST; never claim
  "validated", "shipped", or "works on console".

NEXT MOVE: close N0 with the smallest self-contained test - a hand-authored ONNX graph run against
`C:\Windows\System32\onnxruntime.dll` through its C API (ctypes or a C++ exe), printing the REGISTERED PROVIDER LIST
back from the session, then the same graph through the RadeonML surface if ORT has no IP EP. No installs needed for that
first step; nothing touches the device until you give a go-ahead.
