# REPORT — 2026-11-04 · machines, build options, and the XDNA2 question

Purpose: this is the previous day's report, saved as a standalone artifact and updated with what was
MEASURED on 2026-11-04. It is a REPORT, not a contract — `xbox/PLAN.md` is the contract.

## 1. What this box actually is (MEASURED 2026-11-04, `xrt-smi examine --batch`)

```
C:\Windows\System32\AMD\xrt-smi.exe examine --batch
  OS Release              26300.9818
  Distribution            Windows 11 Home
  Machine               x86_64, CPU Cores 16, Memory 15993 MB
  Model                 ROG Xbox Ally X RC73XA_RC73XA
  BIOS Vendor           American Megatrends …
  BIOS Version          RC73XA.319
  Processor             AMD Ryzen AI Z2 Extreme
  Device(s) Present     [0065:00:01.1]
  Device Name           NPU Strix
```

`xrt-smi --version` → `Version 2.21.0 / Branch HEAD / Hash 15e6319be8… / NPU Driver Version 32.0.20102.4480 / NPU Firmware Version 1.1.2.64`.

`xrt-smi validate --batch` → `Validate Device [0065:00:01.1] / Platform NPU Strix / Power Mode Performance`, then
Test 1 gemm `TOPS: 51.3` `[PASSED]`, Test 2 latency `Average latency: 94.0 us` `[PASSED]`, Test 3 throughput
`Average throughput: 51625.0 op/s` `[PASSED]`.

**Consequence, and it overturns an older assumption in this repo's docs:** an XDNA2-class IP engine
(Strix/AIE2P) is PRESENT and WORKING on this machine. "No device reachable" in older handoff text meant
*ssh to amd9070 / rtx5090 times out* — it never meant "no accelerator here".

## 2. "Cannot build" — what that claim really meant

Three different jobs were being conflated under the single word "build":

1. **Compile `.hip` → `.hsaco`** — needs only COMGR (`hip/rtc_compile.exe`, `amd_comgr_3.dll`). CPU-only, runs here.
   ISA target is a *string*: `--arch gfx1201` / `gfx1151` etc. Nothing about the physical GPU is consulted.
2. **Generate the front-chain `.npz` maps** — the `CUDA-ORACLE` producers. These compile `.cpp` with
   `#include <cuda.h>` and run it on NVIDIA silicon. THIS is the job that genuinely has no host here.
3. **Run + measure on device** (P1–P3, H1–H4) — needs the target GPU to be present and reachable.

`xbox/check.py` gate A2 measures job 1 and PASSes, so "cannot build" was never about the compiler.

## 3. Machine options, one line each

- **This Xbox Ally X (Ryzen AI Z2 Extreme + Strix NPU)** — XRT/XDNA2 stack installed and validated above; no
  RDNA3 gfx1201 compile target here; no CUDA. Best host for the ONNX / Windows ML / IP-engine route.
- **Xbox Ally X iGPU only** — RDNA 3.5, ISA family `gfx115x`/`gfx115E`; it is NOT gfx1201.
- **RX 9070 XT (gfx1201)** — the original ship target. Needs ssh to reach it; measured DEAD from here
  (`ssh amd9070 100.83.122.75` timed out, `rtx5090 100.102.178.207` connection refused).
- **RTX 5090 / DGX Spark (`spark-3a10`)** — the only known hosts for job 2 (CUDA oracle), and the only places
  the NVIDIA `.cubin` oracle can run.
- **Bazzite box, 7900 XT 20 GB + 32 GB RAM** — a normal ROCm/HIP host: fine for jobs 1 and 3 on `gfx94c`-class
  RDNA3 (the 7900 XT is NOT gfx1201 either), no XDNA IP engine, no CUDA.
- **CPU-only on any of them** — compiles job 1, and can host an ONNX CPU-EP reference run. Cannot produce
  device timings (P1–P3) and cannot run a CUDA oracle.

The one thing no local box can do: run the CUDA oracle that the 21 `CUDA-ORACLE` front-chain maps depend on.
That job survives this pivot, because reference outputs for NR are NVIDIA's.

## 4. "The whole point is XDNA2 — can NR run on it?" (re-answered with today's evidence)

Yes, hardware-wise, and it was always hardware-feasible; the blocker is the ENTRY path, not the silicon:

- The IP engine speaks **IP layouts** (`1x4`, `4x4`, `5x4`, `_ipu_2`) and consumes compiled artifacts, exposed
  through `xrt_coreutil.dll` / `pyxrt.pyd` / AIE (`vitis-ai-runtime.dll`) — not through HIP ISA strings.
  `C:\Windows\System32\AMD` holds the AIE2P kernels (`AMD_AIE2P_4x4_Overlay*`, `…_ipu_2`): 34 files including
  `pyxrt.pyd` (533,864 B) and `xrt-smi.exe` (2,292,072 B).
- The NR network is a transformer family (41 modules, `preblock`, `inpview`, `ffn`, `attn`), i.e. matmul/softmax
  composition — expressible as an ONNX graph. So the route is: author ONNX → run it through the VitisAI/Windows
  ML IP pipeline.
- What is NOT true and must not be claimed: that XDNA runs our `.hip`/`.hsaco` (it doesn't — COMGR has no XDNA
  ISA), or that a `.cubin` is portable to AMD (it is NVIDIA bitcode).
- What is UNVERIFIED after today and stays OPEN: whether third-party (unsigned) ONNX models may use the IP path
  from a normal user-mode process on Windows 11 (XVA attestation). A 6-day-old claim said attestation is required;
  today's DLL string scan found `XVA` in `onnxruntime.dll` but **no** `attest`/`Attest`/`isSigned`/`AllowedOrigins`
  markers in `Windows.AI.MachineLearning.dll`. Treat as OPEN, not settled.

## 5. Local ONNX / Windows ML inventory (MEASURED 2026-11-04)

`C:\Windows\System32`: `onnxruntime.dll` 10,675,424 B, `directml.dll` 10,239,288 B, `Windows.AI.MachineLearning.dll`
1,196,848 B, `Windows.AI.MachineLearning.Preview.dll`, `winml.dll`, `RadeonML_DirectML.dll`, `RadeonML.dll`,
`RadeonML_aks.dll`, `RadeonML_ipu.dll`, `amd_taau_ops.rml_plugin.dll`, `vitis-ai-runtime.dll`, `vitis-ai-runtime2.dll`,
`xrt_coreutil.dll`, `Windows.AI.Agents.dll`, `Windows.Internal.AI.PlatformCapability.dll`.

`onnxruntime.dll` strings: `Current ORT Version is: %s` with `1.17.1` nearby; `ORT Build Info: git-branch=HEAD,
git-commit-id=0`; `EP_COUNT 32` and an EP list containing `DmlExecutionProvider`, `VitisAIExecutionProvider`,
`OpenVINOExecutionProvider`, `CPUExecutionProvider`, `CUDAExecutionProvider`, `ROCMExecutionProvider`,
`MIGraphXExecutionProvider`, `QNNExecutionProvider`. Counting exact strings inside it: `VitisAI` ×2, `Vitis` ×2,
`XRT` ×1, `execution_provider` ×5, but `config_file_path` ×0 and `perf_profile` ×0 — i.e. the EP *name* is in the
table while the VitisAI config keys are absent. Whether the VitisAI EP is actually compiled into that build is
**NOT SETTLED** (a name in the list is not proof of availability); see `PLAN.md` gate N3.

Python side: `py --list` → `-V:3.14[-64] * Python 3.14.7` only; `py -3.10 …` → `NO_PY310_LAUNCHER`; `uv` → `NO_UV`;
packages: `numpy True`, `onnxruntime False`, `onnx False`, `torch False`. `pyxrt.pyd` byte-scan shows it links
`python310.dll` + `xrt_coreutil.dll`, and `import pyxrt` → `ImportError: DLL load failed while importing pyxrt:
Nie można odnaleźć określonego modułu.` (that string is cmd.exe's localisation of the error, not a Windows ML verdict).
`dumpbin` is not installed (`'dumpbin' is not recognized …`), which is why the dependency list came from a byte scan.

Network: `PYPI_STATUS 200 BYTES 645387` — so installing `onnx`/`onnxruntime` is possible but is a SYSTEM change and
needs your go-ahead.

## 6. Repo-side facts for the new route

- NR is **not** shipped as ONNX. `nvngx_dlssnr.dll` ships 15 `sm_120` `.cubin` blobs plus a proprietary weight
  container: magic `b'DLSSNRW1'`, `record_count 0x99` = 153, `header_len 0x1629` = 5673, per-record
  `[u8 name_len][name][u64 payload_offset][u64 payload_size]`. dtypes are MIXED: E4M3 matrix bodies, f16 bias/skip/slots,
  f32 scale. Container size 147,689,451 B and it is NOT in the repo — the only copy lives in the game install.
- So the ONNX route = author our own graph from the reverse-engineered architecture and inject those 153 named
  tensors as initializers. No NVIDIA graph to convert.
- In-repo ONNX prior art is the AGENT's perception nets, not NR: `Development/history/amd-port-plan.md:200`
  ("ONNX Runtime"), `:242` ("ONNX Runtime / DirectML: fastest way to verify the AMD card can produce output"),
  `Development/results/character-mask-survey-20261011/skin-and-parsing.md:118/123/135/142` (ViT back chain re-implemented
  in Python, 2,840,120 params, "can be exported to ONNX"), `Development/results/face-detection-probe-20261011/README.md:3`.
- Index files that the new route will consume: `Development/weights-index.json` (offsets are ARCHIVE-ABSOLUTE with
  growing drift — do not trust them as container-relative) and `Development/weights-arena-index.json` (25,809 B, a dict
  under key `records`).
