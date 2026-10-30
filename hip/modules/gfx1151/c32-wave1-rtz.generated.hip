#define HIP_ISA_HALF 1
#define CW_PACK_MODE_MASK 0
#define CW_UP_FUSED 1
#define CW_ACT_FMED3 1
#define HIP_PREPACKED_WEIGHTS 1
#define CW_ROLL_HIDDEN 1
#define CW_ROLL_WINDOW 1
#define CW_VEC_INPUT 1
#define CW_PREFIX_SPLIT 1
#define CW_PACK8 1
#define HIP_FP8_SAT_MODE 3
#define CW_DIRECT_OUT 1
#define CW_RTZ_PAIR 1
#define CW_PREFIX_DIRECT_OUT 1
#define CW_PREFIX_FULL_TILE 1
#define CW_FINISH_FULL_TILE 1
#define CW_SKIP_BYTE 1
#define CW_DIAG_ONLY 1
#define CW_INPUT_HALF 7
#define CW_PREFIX_HALF_SOURCE 1
#define CW_PREPOST_BYTE 1
#define CW_PREFIX_TAIL_VEC 1
#define CW_FINISH_TAIL_VEC 1
#define CW_HOIST_UP 3
#define HIP_C32_RTZ_ISA 2
// gfx1103 (RDNA3) has no fp8-conversion-insts, so the two convert builtins used by the gfx12
// fast sources are emulated here. build-modules.ps1 prepends this header ONLY for -Targets
// gfx1103, so gfx1200/gfx1201 generated.hip text and their SHA256SUMS stay untouched.
//
// Semantics here are DERIVED, not measured on this box:
//  - the packing convention comes from the gfx12 call sites (see gmx_cvt_pk_fp8_f32 below);
//  - the rounding rule is the e4m3 RNE-with-saturation-at-448 already implemented as F() in
//    c32_wmma.hip / deep_wmma.hip / multihead_*.hip, reproduced by construction.
// Nothing here is executed on this box: COMGR emits gfx1103 but cannot run it. Validation is
// a device job (PLAN A4: "ALU kernels are validated ON DEVICE, not on this box").

typedef unsigned uint;
#define GMX_DEV __attribute__((device)) __attribute__((always_inline))

GMX_DEV uint gmx_bits(float x){union{float f;uint u;}v;v.f=x;return v.u;}
GMX_DEV float gmx_value(uint x){union{float f;uint u;}v;v.u=x;return v.f;}
GMX_DEV float gmx_abs(float x){return gmx_value(gmx_bits(x)&0x7fffffffu);}

// One float -> one e4m3 byte, round-to-nearest-even, saturating at 448 (= 0x7e).
// Matches F() bit-for-bit: same 0x43e00000 (448) saturation boundary, same 2^-6 subnormal
// branch, same +0x7ffff/lsb rounding to 3 mantissa bits.
GMX_DEV uint gmx_e4m3_enc(float x){
  uint b=gmx_bits(x), a=b&0x7fffffffu, s=(b>>24)&0x80u;
  if(a>=0x43e00000u)return s|0x7eu;                        // 448, incl. inf/nan overflow
  if(a<0x3c800000u){ // |x| < 2^-6: the whole subnormal range is an integer count of 2^-9
    float sc=gmx_abs(x)*512.f;uint q=(uint)sc;float r=sc-(float)q;
    q+=uint(r>0.5f||(r==0.5f&&(q&1u)));
    return s|(q&31u);}
  uint r=(a+0x7ffffu+((a>>20)&1u))&0xfff00000u;            // keep 3 mantissa bits
  uint e=((r>>23)&0xffu)-127u;
  return s|((e+7u)<<3)|((r>>20)&7u);
}

// One e4m3 byte -> float. Exact in f32 (and in f16, per PLAN's numerics rule).
GMX_DEV float gmx_e4m3_dec(uint byte){
  uint e=(byte>>3)&0xfu, m=byte&7u;
  float v=e?(float)(8u|m)*(e>=7u?(float)(1u<<(e-7u)):1.f/(float)(1u<<(7u-e))):(float)m*0.001953125f;
  return (byte&0x80u)?-v:v;
}

// cvt_pk_fp8_f32(a,b,c,d): a,b pack into the LOW 16 bits when d is false, or into the HIGH 16
// bits merged with c when d is true. Derived from deep_fast.hip:129 / c512_m32_deep.inc:150,
// which build one 4x8-bit group as (t0,t1,0,false) then (t2,t3,lo,true).
GMX_DEV int gmx_cvt_pk_fp8_f32(float a,float b,int c,bool d){
  uint lo=gmx_e4m3_enc(a)|(gmx_e4m3_enc(b)<<8);
  return (int)(d?(lo<<16)|((uint)c&0x0000ffffu):((uint)c&0xffff0000u)|lo);
}
GMX_DEV float gmx_cvt_f32_fp8(int packed,int idx){return gmx_e4m3_dec(((uint)packed>>(8*idx))&0xffu);}

// Shadowing a builtin name with a function-like macro compiles for gfx1103 (verified by
// xbox/probe_isa.py -> p_macro_shim PASS), which is what makes one header cover every call site.
#define __builtin_amdgcn_cvt_pk_fp8_f32(a,b,c,d) gmx_cvt_pk_fp8_f32((float)(a),(float)(b),(int)(c),(bool)(d))
#define __builtin_amdgcn_cvt_f32_fp8(p,i) gmx_cvt_f32_fp8((int)(p),(int)(i))

// The *_gfx12 WMMA builtins ARE macro-mappable after all. On gfx1103 they report
// "needs target feature gfx12-insts,wavefrontsize32", and the only form that survives is the
// UNSUFFIXED __builtin_amdgcn_wmma_f32_16x16x16_f16_w32. Its operand vectors are wider than the
// sources' (evidence: xbox/out_p_wmma16_gfx1103.hsaco.s emits
// `v_wmma_f32_16x16x16_f16 v[1:8], v[9:16], v[17:24], v[1:8]` - A and B each span 8 registers,
// i.e. 16 __fp16 per lane, while the accumulator is 8 f32), and probe_isa.py's p_wmma_f16x8 FAILS
// with "cannot initialize a parameter of type ... vector_size(16*sizeof(__fp16)) __fp16".
// That is a TYPE mismatch, not an arithmetic one: clang vector types are structural, so one helper
// per fragment shape takes the sources' typedefs (h8 = _Float16 x8, i2 = int x2, f8 = float x8)
// and widens into the x16 operand. f16 storage per PLAN (halves LDS traffic vs f32); e4m3 -> f16
// is exact, so the fp8 path stays numerically faithful to the gfx12 sources.
#if !defined(__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12)
#define HIP_WMMA_SHIM 1
typedef _Float16 gmx_h8 __attribute__((ext_vector_type(8)));
typedef int      gmx_i2 __attribute__((ext_vector_type(2)));
typedef float    gmx_f8 __attribute__((ext_vector_type(8)));
typedef __fp16   gmx_p16 __attribute__((ext_vector_type(16)));

// WHICH of the 16 slots carry the tile data is NOT derivable offline: no HIP SDK tree beside the
// repo (checked C:\kien), and COMGR emits gfx1103 but cannot run it. Slots 0..7 is the working
// hypothesis; resolve it ON DEVICE with xbox/p_wmma_layout.hip before trusting any numerics from
// the modules this shim unblocks. It is ONE function, so a measured map is a one-place edit.
GMX_DEV gmx_p16 gmx_widen(gmx_h8 v){
  gmx_p16 r{};
  for(uint i=0u;i<8u;i++)r[i]=(__fp16)v[i];
  return r;
}
GMX_DEV gmx_h8 gmx_fp8_row(gmx_i2 p){
  gmx_h8 r{};
  for(uint e=0u;e<8u;e++)r[e]=(_Float16)gmx_e4m3_dec(((uint)p[e>>2]>>(8*(e&3u)))&0xffu);
  return r;
}
GMX_DEV gmx_f8 gmx_wmma_f16(gmx_h8 a,gmx_h8 b,gmx_f8 c){
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(gmx_widen(a),gmx_widen(b),c);
}
GMX_DEV gmx_f8 gmx_wmma_fp8(gmx_i2 a,gmx_i2 b,gmx_f8 c){
  return gmx_wmma_f16(gmx_fp8_row(a),gmx_fp8_row(b),c);
}
#define __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,b,c) gmx_wmma_f16(a,b,c)
#define __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,c) gmx_wmma_fp8(a,b,c)
#endif

#ifndef HIP_C32_ALIAS_FFN_V
#define HIP_C32_ALIAS_FFN_V 1
#endif
// Production C32 FAST4 attention precision, separate from exact reference.
// All kernels flat grid ceil(work_items/256), block32; token count divisible64.
#ifndef HIP_C32_LANE_STAGE
#define HIP_C32_LANE_STAGE 1 /* 2026-09-16: lane-broadcast input staging, -0.14ms, bit-exact */
#endif
#ifndef HIP_C32_DIAG_WEIGHTS
#define HIP_C32_DIAG_WEIGHTS 1 /* chain residual diagonal B tiles prepacked after the FFN weights (byte 34944); 2026-09-16 -0.05 (noise) on the old kernel, 2026-09-17 -0.16 once the kernel was no longer issue-bound, bit-exact */
#endif
#ifndef HIP_C32_NO_UNROLL
#define HIP_C32_NO_UNROLL 0 /* 1: keep the FFN/QKV/attention/projection loops rolled (fewer live accumulators -> lower VGPR count -> more waves per SIMD; HLSL fast4 ~128 VGPR) */
#endif
#if HIP_C32_NO_UNROLL
#define C32_LOOP _Pragma("clang loop unroll(disable)")
#else
#define C32_LOOP
#endif
#ifndef HIP_C32_WAVES_PER_EU
#define HIP_C32_WAVES_PER_EU 0 /* >0: amdgpu_waves_per_eu occupancy target for the fused kernels (HLSL fast4 runs ~12 waves/SIMD at ~128 VGPRs; HIP chain is 190) */
#endif
#if HIP_C32_WAVES_PER_EU
#define C32_OCC __attribute__((amdgpu_waves_per_eu(HIP_C32_WAVES_PER_EU,HIP_C32_WAVES_PER_EU)))
#else
#define C32_OCC
#endif
#ifndef HIP_C32_BRANCHLESS_F
#define HIP_C32_BRANCHLESS_F 1 /* 2026-09-17: -0.19ms alone, bit-exact. F()/fp8() as clamp+cvt+cndmask (ISA showed every F() call site as a divergent branch pair with exec-mask save/restore: 16 per token stage + 16 per output; only NaN input differs, sign of the saturated value) */
#endif
#ifndef HIP_C32_STAGE_PREFETCH
#define HIP_C32_STAGE_PREFETCH 1 /* 2026-09-17: -0.68ms together with BRANCHLESS_F, bit-exact. lane staging issues all 16 token loads before any conversion (ISA showed load -> s_wait_loadcnt 0 -> branchy F per token, 16 serialized latencies per wave) */
#endif
#ifndef HIP_C32_ABLATE
#define HIP_C32_ABLATE 0 /* timing-only ablation, never production: 1 skip FFN expand+contract, 2 skip scores/softmax/AV, 3 skip projection+output */
#endif
#ifndef HIP_C32_VT
#define HIP_C32_VT 0 /* 1: V stored transposed in its own LDS array so the AV B fragments load as 8 contiguous bytes; same bytes */
#endif
#ifndef HIP_C32_HOIST_PW
#define HIP_C32_HOIST_PW 1 /* 2026-09-17: -0.05ms, bit-exact. projection scale w[8193+col] read once per lane instead of 16 aliasing-forced reloads in the output loop */
#endif
#ifndef HIP_C32_TRANSPOSED_TAIL
#define HIP_C32_TRANSPOSED_TAIL 0 /* 2026-09-24 experiment: FFN contraction, prefix matrix and projection computed transposed (WMMA operand swap
 = exact transpose, as in the folded expand and mh_fast's tail). A lane then holds one token row with 8 consecutive channels, so the
 unrolled narrow stores widen: ffn8 bytes 16 -> 2 (8-byte), scratch.ex halves 16 -> 2 (16-byte), the raw FP8 out bytes 16 -> 2 per wave,
 pre16 halves likewise; the residual init reads 8 consecutive in16 halves per lane. Projection scale needs 16 hoisted loads instead of 2.
 Requires FOLDED_FFN + REGISTER_FFN (saved_ffn/ffn8 layout). Same values at the same addresses -> bit-exact. */
#endif
#ifndef HIP_C32_LOCAL_ATTN_SYNC
#define HIP_C32_LOCAL_ATTN_SYNC 0 /* probabilities in the wave's own ex rows; attention-phase barriers become wave-local fences (HLSL fast4: two group syncs) */
#endif
#ifndef HIP_C32_LOCAL_FFN_SYNC
#define HIP_C32_LOCAL_FFN_SYNC 0 /* staging/hidden rows are wave-private: wave-local fences instead of s_barrier */
#endif
#ifndef HIP_C32_INPUT_DWORD
#define HIP_C32_INPUT_DWORD 1
#endif
#ifndef HIP_C32_DIRECT_WEIGHT8
#define HIP_C32_DIRECT_WEIGHT8 1
#endif
#ifndef HIP_C32_WEIGHT_DWORD
#define HIP_C32_WEIGHT_DWORD 1
#endif
#ifndef HIP_C32_LOCAL_QKV_SYNC
#define HIP_C32_LOCAL_QKV_SYNC 1
#endif
#ifndef HIP_C32_BYTE_CHAIN
#define HIP_C32_BYTE_CHAIN 1 /* 2026-09-23: raw chain carries FP8 bytes; also enables the vectorised 128-bit staging for the f32-row Mapped kernels (mapped/post). mapped/chain producers (HalfOutput, raw_output=1) store fp8(F(float((_Float16)v))) -- exactly the byte the f16 consumer used to compute -- and RawMapped consumers load one byte per element with a vectorised 128-bit staging path, skipping F(). Bit-exact; whole network -0.08/-0.06ms. The raw buffer is then bytes, not f16: boundary_fast c32_finish_crop_half (SkipChainFinish, C32 skip blocks) still expects f16 and must not be combined with this flag. */
#endif
#ifndef HIP_C32_FOLDED_FFN
#define HIP_C32_FOLDED_FFN 1 /* 2026-09-22: expand->contract without the LDS round trip (transposed WMMA operands); bit-exact, -0.10/-0.06ms whole network */
#endif
#ifndef HIP_C32_IN16_ALIAS
#define HIP_C32_IN16_ALIAS 1 /* 2026-09-23: the NeedIn16 kernels (half_mapped, post_merge_head/fused) keep the staged f16 rows in the
 Scratch union instead of their own 4352-byte array. in16 is last read by the mode-0 residual init before the FFN; Scratch is first
 written in the QKV normalisation, two sync_windows later, and scratch.hidden is compiled out (FOLDED_FFN). LDS 19712 -> 15360 per
 group, CU-mode residency 3 -> 4 groups per CU (6 -> 8 waves/SIMD, HW_ID-verified). Bit-exact; -0.09/-0.06 ms (1080/900). */
#endif
#ifndef HIP_C32_REGISTER_FFN
#define HIP_C32_REGISTER_FFN 1
#endif
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
typedef unsigned uint;
#define DEV __attribute__((device)) __attribute__((always_inline))
#ifndef HIP_LDS_FENCE
#define HIP_LDS_FENCE 1 /* 2026-09-22 fence scope: workgroup fences here only publish LDS; the default "workgroup" scope made gfx12 emit global_inv (vector L0 invalidate) and wait for all outstanding global loads at every barrier. Restricting the fence to the LDS address space removes both; bit-exact, see Development/results/fence-scope-all-20260922 */
#endif
#if HIP_LDS_FENCE
#define WG_FENCE(k) __builtin_amdgcn_fence(k,"workgroup","local")
#else
#define WG_FENCE(k) __builtin_amdgcn_fence(k,"workgroup")
#endif
#ifndef HIP_C32_CU_MODE
#define HIP_C32_CU_MODE 1 /* 2026-09-22: 128-thread C32 workgroups run faster with all four waves on one CU (-0.13/-0.17ms whole network on top of the fence change, bit-exact); larger MH/deep workgroups get slower in CU mode, so this is C32-only */
#endif
#ifndef HIP_KERNEL_WPE
#define HIP_KERNEL_WPE 0 /* 2026-10-01 compiler sweep (diagnostic, default 0 = no attribute): amdgpu_waves_per_eu lower bound applied to every kernel of the module */
#endif
#ifndef HIP_KERNEL_ATTR
#if HIP_KERNEL_WPE
#define HIP_KERNEL_ATTR __attribute__((amdgpu_waves_per_eu(HIP_KERNEL_WPE)))
#else
#define HIP_KERNEL_ATTR
#endif
#endif
#if HIP_C32_CU_MODE
#define KERNEL extern "C" __attribute__((global)) __attribute__((target("cumode"))) HIP_KERNEL_ATTR
#else
#define KERNEL extern "C" __attribute__((global)) HIP_KERNEL_ATTR
#endif
#if HIP_C32_TRANSPOSED_TAIL && !(HIP_C32_FOLDED_FFN && HIP_C32_REGISTER_FFN)
#error "HIP_C32_TRANSPOSED_TAIL needs HIP_C32_FOLDED_FFN and HIP_C32_REGISTER_FFN"
#endif
/* FFN-contraction family accumulations (residual diagonal, expand->contract): operand order swapped under the transposed tail */
#if HIP_C32_TRANSPOSED_TAIL
#define C32_WMMA_ACC(a,b,c) __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(b,a,c)
#else
#define C32_WMMA_ACC(a,b,c) __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,c)
#endif
// DLSS5_STYLE (2026-10-01): NGX Style control as preprocess feature 6 = Style/128. Default 1/128 (Style=1, the value captured
// from the original runtime on 09-06); the host overwrites it per module with hipModuleGetGlobal when DLSS5_STYLE is set.
#ifndef DLSS5_STYLE_FEATURE_DEFINED
#define DLSS5_STYLE_FEATURE_DEFINED 1
__attribute__((constant)) float dlss5_style_feature=.0078125f;
#endif
DEV uint bits(float x){union{float f;uint u;}v;v.f=x;return v.u;}
DEV float value(uint x){union{float f;uint u;}v;v.u=x;return v.f;}
DEV float absf(float x){return value(bits(x)&0x7fffffffu);}
DEV float maxf(float a,float b){return a>b?a:b;}
DEV float minf(float a,float b){return a<b?a:b;}
DEV float clampf(float a,float lo,float hi){return minf(maxf(a,lo),hi);}
DEV uint shift_rne(uint x,uint n){if(n>31)return 0;uint q=x>>n,mask=(1u<<n)-1u,r=x&mask,mid=1u<<(n-1);return q+uint(r>mid||(r==mid&&(q&1u)));}
DEV uint half_bits(float x){uint b=bits(x),a=b&0x7fffffffu,s=(b>>16)&0x8000u;int e=int(a>>23)-127;
 if(a>=0x7f800000u)return s|0x7c00u|((a&0x7fffffu)?0x200u:0u);
 if(e<-25)return s;
 if(e<-14)return s|shift_rne((a&0x7fffffu)|0x800000u,uint(-e-1));
 if(e>15)return s|0x7c00u;
 uint h=((a+0xfffu+((a>>13)&1u))>>13)-0x1c000u;return s|(h>=0x7c00u?0x7c00u:h);
}
DEV float from_half(uint h){uint s=(h&0x8000u)<<16,e=(h>>10)&31u,m=h&1023u;if(e==31)return value(s|0x7f800000u|(m<<13));if(!e){if(!m)return value(s);float f=float(m)*5.9604644775390625e-8f;return s?-f:f;}return value(s|((e+112)<<23)|(m<<13));}
#ifdef HIP_ISA_HALF
// Keep f32 expression evaluation and both rounding boundaries opaque to fusion.
DEV float H(float x){uint h;float result;asm("v_cvt_f16_f32 %0, %1" : "=v"(h) : "v"(x));asm("v_cvt_f32_f16 %0, %1" : "=v"(result) : "v"(h));return result;}
#else
DEV float H(float x){return from_half(half_bits(x));}
#endif
#ifndef HIP_NATIVE_FP8_F
#define HIP_NATIVE_FP8_F 1
#endif
// HIP_FP8_SAT_MODE (experiment fp8-sat-mode, default 0): the kernel sets MODE.FP16_OVFL (hwreg MODE bit 23) at entry, which
// makes v_cvt_pk_fp8_f32 saturate finite overflow to +-448 by itself, so the explicit clamp before each cast is dropped.
// Measured on gfx1201: identical bytes for every finite input; +-Inf and NaN become the E4M3 NaN byte instead of +-448 /
// -448; every f16 conversion in the kernel also saturates to +-65504 instead of +-Inf. 2 = MODE bit only, clamps kept (diagnostic).
#ifndef HIP_FP8_SAT_MODE
#define HIP_FP8_SAT_MODE 0
#endif
DEV void fp8_sat_mode(){
#if HIP_FP8_SAT_MODE==1||HIP_FP8_SAT_MODE==2
 __builtin_amdgcn_s_setreg(1|(23<<6),1);
#endif
}
#if HIP_NATIVE_FP8_F && HIP_FP8_SAT_MODE==1
DEV float F(float x){uint a=bits(x)&0x7fffffffu;float r=__builtin_amdgcn_cvt_f32_fp8(__builtin_amdgcn_cvt_pk_fp8_f32(x,0.f,0,false),0);return a==0?0.f:r;}
#elif HIP_NATIVE_FP8_F && HIP_FP8_SAT_MODE==3
// 3 = no MODE change, the clamp as one v_med3_f32 instead of v_max_num(x,x)+v_maxmin_num; probe: same byte as the
// fminf/fmaxf form for every finite, infinite and NaN input (NaN -> -448).
DEV float F(float x){uint a=bits(x)&0x7fffffffu;float r=__builtin_amdgcn_cvt_f32_fp8(__builtin_amdgcn_cvt_pk_fp8_f32(__builtin_amdgcn_fmed3f(x,-448.f,448.f),0.f,0,false),0);return a==0?0.f:r;}
#elif HIP_NATIVE_FP8_F && HIP_C32_BRANCHLESS_F
// Same values as the branchy form for every finite and infinite input (clamp to +-448 then RNE cast; +-448 converts exactly,
// values in (447.75,448] already round to 448 in both), zero keeps the +0 result; a NaN input yields -448 instead of +-448.
DEV float F(float x){uint a=bits(x)&0x7fffffffu;float c=__builtin_fminf(__builtin_fmaxf(x,-448.f),448.f);float r=__builtin_amdgcn_cvt_f32_fp8(__builtin_amdgcn_cvt_pk_fp8_f32(c,0.f,0,false),0);return a==0?0.f:r;}
#elif HIP_NATIVE_FP8_F
DEV float F(float x){
 uint b=bits(x),a=b&0x7fffffffu;
 if(a==0)return 0.f;
 // Preserve the existing signed saturation, including nonfinite inputs.
 if(a>=0x43e00000u)return value((b&0x80000000u)|0x43e00000u);
 int packed=__builtin_amdgcn_cvt_pk_fp8_f32(x,0.f,0,false);
 return __builtin_amdgcn_cvt_f32_fp8(packed,0);
}
#else
DEV float F(float x){uint b=bits(x),a=b&0x7fffffffu,s=b&0x80000000u;if(a==0)return 0.f;
 if(a>=0x43e00000u)return value(s|0x43e00000u); // finite E4M3 saturation, including overflow
 if(a<0x3c800000u){float scaled=absf(x)*512.f;uint q=uint(scaled);float r=scaled-float(q);q+=uint(r>.5f||(r==.5f&&(q&1u)));float f=float(q)/512.f;return s?-f:f;}
 uint r=(a+0x7ffffu+((a>>20)&1u))&0xfff00000u;return value(s|(r>0x43e00000u?0x43e00000u:r));}
#endif

using i2=int __attribute__((ext_vector_type(2)));
using i4=int __attribute__((ext_vector_type(4)));
using f4=float __attribute__((ext_vector_type(4)));
using f8=float __attribute__((ext_vector_type(8)));
using h8=_Float16 __attribute__((ext_vector_type(8)));
#define WAVE KERNEL __attribute__((amdgpu_flat_work_group_size(32,32)))
DEV uint rc(){return __builtin_amdgcn_workitem_id_x()&15;}
DEV uint gr(){return (__builtin_amdgcn_workitem_id_x()&31)>>4;}
DEV uint bid(){return __builtin_amdgcn_workgroup_id_x();}
// D3D explicit f32tof16 maps to RTZ here; matrix Cast<F16> remains RNE.
#ifndef HIP_C32_RTZ_ISA
#define HIP_C32_RTZ_ISA 1
#endif
// FAST TIER (lossy, 2026-10-01 fast-tier): CW_FAST_NUM bit0 = f32 activation path (Hrtz is the identity: no RTZ
// round trip to half between f32 ops; half stores still convert), bit1 = softmax 1/sum as a single rcp (no Newton steps).
// Norm WMMA rows are replicated; optional900 fast recipe reuses one rsqrt perlane.
#ifndef CW_NORM_HOIST
#define CW_NORM_HOIST 0
#endif
#ifndef CW_FAST_NUM
#define CW_FAST_NUM 0
#endif
#if CW_FAST_NUM&1
DEV float Hrtz(float x){return x;}
#elif HIP_C32_RTZ_ISA==2
// 2026-10-02 ideas-yami-ikaruga: the same two instructions as builtins (v_cvt_pkrtz_f16_f32, then an exact f16->f32 widen) so
// LLVM may fold the widen into the consumer (v_fma_mix_f32 with a -0 addend == the f32 mul/add of the widened half). Value
// per input identical to the asm form (GPU 2^32 check in results/ideas-yami-ikaruga-20261002).
DEV float Hrtz(float x){auto h=__builtin_amdgcn_cvt_pkrtz(x,x);return float(h[0]);}
#elif HIP_C32_RTZ_ISA
DEV float Hrtz(float x){uint h;float result;
 asm("v_cvt_pkrtz_f16_f32 %0, %1, %1" : "=v"(h) : "v"(x));
 asm("v_cvt_f32_f16 %0, %1" : "=v"(result) : "v"(h));
 return result;
}
#else
DEV float Hrtz(float x){uint b=bits(x),a=b&0x7fffffffu,sg=(b>>16)&0x8000u;int e=int(a>>23)-127;
 if(a>=0x7f800000u)return from_half(sg|0x7c00u|((a&0x7fffffu)?0x200u:0u));
 if(e>15)return from_half(sg|0x7bffu);
 if(e<-24)return from_half(sg);
 if(e<-14)return from_half(sg|(((a&0x7fffffu)|0x800000u)>>uint(-e-1)));
 return from_half(sg|((a>>13)-0x1c000u));
}
#endif
#if HIP_FP8_SAT_MODE==1
DEV uint fp8(float f){return uint(__builtin_amdgcn_cvt_pk_fp8_f32(f,0.f,0,false))&255u;}
#elif HIP_FP8_SAT_MODE==3
DEV uint fp8(float f){return uint(__builtin_amdgcn_cvt_pk_fp8_f32(__builtin_amdgcn_fmed3f(f,-448.f,448.f),0.f,0,false))&255u;}
#elif HIP_C32_BRANCHLESS_F
DEV uint fp8(float f){return uint(__builtin_amdgcn_cvt_pk_fp8_f32(__builtin_fminf(__builtin_fmaxf(f,-448.f),448.f),0.f,0,false))&255u;}
#else
DEV uint fp8(float f){return uint(__builtin_amdgcn_cvt_pk_fp8_f32(clampf(f,-448.f,448.f),0.f,0,false))&255u;}
#endif
// Bounded reciprocal: the 64 positive exponent values sum to
// [1/256, 624], so no zero/subnormal/infinite divisor scaling is needed.
// gfx1201 exhaustive test: all 144441345 float divisors in this interval
// match the original 1.f/x bit-for-bit (2026-09-19).
DEV float norm_inverse(float x){float r=__builtin_amdgcn_rcpf(x);
#if CW_FAST_NUM&2
 return r;
#endif
 r=__builtin_fmaf(r,__builtin_fmaf(-x,r,1.f),r);
 return __builtin_fmaf(r,__builtin_fmaf(-x,r,1.f),r);
}
DEV void put(i2&v,uint e,float x){v[e/4]=int(uint(v[e/4])|(fp8(x)<<(8*(e%4))));}

DEV void sync_window(){WG_FENCE(3);__builtin_amdgcn_s_barrier();WG_FENCE(2);}

// Only for wave-private LDS rows: order memory without waiting for other waves.
DEV void sync_owned_rows(){
#if HIP_C32_LOCAL_QKV_SYNC
 WG_FENCE(3);WG_FENCE(2);
#else
 sync_window();
#endif
}

DEV float scale_piece(float v){float m=absf(v);if(m<.015625f){float x=m*512.f;uint q=uint(x);float r=x-float(q);q+=uint(r>.5f||(r==.5f&&(q&1u)));if(q>7)q=7;return value((bits(v)&0x80000000u)|bits(float(q)/512.f));}return F(v);}
// Packed attention: shared raw/exp scratch 8448 bytes + normalized/prob/AV bytes 6912.
using ushort=unsigned short;
DEV void put_bits(i2&v,uint e,uint b){v[e/4]=int(uint(v[e/4])|((b&255u)<<(8*(e%4))));}
DEV uint matrix_byte(const float*w,uint start,uint index){
#ifdef HIP_PREPACKED_WEIGHTS
 return reinterpret_cast<const unsigned char*>(w+start)[index];
#else
 return fp8(w[start+index]);
#endif
}
DEV i2 matrix8(const float*w,uint index){
#if HIP_C32_DIRECT_WEIGHT8 && defined(HIP_PREPACKED_WEIGHTS)
 i2 result;__builtin_memcpy(&result,reinterpret_cast<const unsigned char*>(w)+index,8);return result;
#else
 i2 result{};for(uint e=0;e<8;e++)put_bits(result,e,matrix_byte(w,0,index+e));return result;
#endif
}
// Byte-preserving packed LDS reads; memcpy does not impose pointer alignment.
DEV i2 load8(const unsigned char*p){
#ifdef HIP_C32_LDS_VECTOR
 i2 result;__builtin_memcpy(&result,p,8);return result;
#else
 i2 result{};for(uint e=0;e<8;e++)put_bits(result,e,p[e]);return result;
#endif
}
// Inline prefix (host option prefix_inline, block 0 only): dlss5_prefix_fast_fused_raster's per-token features and the
// 16->32 f16 MMA projection run inside the input staging (HLSL DLSS5_INLINE_PREFIX shape): same RNG, roundings and MMA
// operand order (both K tiles, the second all-zero, so signed zeros match), no W*H*32 f32 prefix buffer written or read.
DEV uint production_pcg(uint s){uint w=((s>>((s>>28)+4))^s)*0x108ef2d9u;return (w>>22)^w;}
DEV float production_uniform(uint s){uint w=((s>>((s>>28)+4))^s)*0x108ef2d9u;return float(((w>>30)^(w>>8))+1)*5.9604644775390625e-8f;}
DEV void fused_prefix_values(const float*rgba,const float*history,uint p,uint width,uint seed,uint temporal,float*f){
 uint tile=p/64,x=(tile%(width/8))*8+p%8,y=(tile/(width/8))*8+(p%64)/8,rgbp=y*width+x;
 uint h=production_pcg((x*0x8da6b343u)^(y*0xd8163841u)^(seed*0x9e3779b9u)^0x243f6a88u);
 float a=production_uniform(h*0xcaa5b80du+0x21dd796bu),b=production_uniform(h*0x2c9277b5u+0xac564b05u),c=production_uniform(h*0x83232c31u+0x3463e0acu),d=production_uniform(h*0xfa6dc5f9u+0x4712a88eu);
 float r0=__builtin_amdgcn_sqrtf(-2.f*(__builtin_amdgcn_logf(a)*.6931471805599453f)),r1=__builtin_amdgcn_sqrtf(-2.f*(__builtin_amdgcn_logf(b)*.6931471805599453f));
 float ct=(6.283185482025146f*c)*.15915494309189535f,dt=(6.283185482025146f*d)*.15915494309189535f;
 float g0=Hrtz(r0*__builtin_amdgcn_cosf(ct)),g1=Hrtz(r1*__builtin_amdgcn_cosf(dt)),g2=Hrtz(r1*__builtin_amdgcn_sinf(dt));
 float red=Hrtz(Hrtz(Hrtz(rgba[rgbp*4])-.5f)*.125f),green=Hrtz(Hrtz(Hrtz(rgba[rgbp*4+1])-.5f)*.125f),blue=Hrtz(Hrtz(Hrtz(rgba[rgbp*4+2])-.5f)*.125f);
 float values[16]={g1,g2,green,blue,g0,1,dlss5_style_feature,1,red,green,1,1,blue,red,1,0};
 if(temporal){uint q=(y*width+x)*4;values[13]=Hrtz(Hrtz(Hrtz(history[q])-.5f)*.125f);values[2]=Hrtz(Hrtz(Hrtz(history[q+1])-.5f)*.125f);values[3]=Hrtz(Hrtz(Hrtz(history[q+2])-.5f)*.125f);}
 for(uint i=0;i<16;i++)f[i]=values[i];
}
template<bool Raw=false,bool Merge=false>
DEV float mapped_c32_input(const float*in,uint window,uint token,uint channel,uint width,uint height,uint sx,uint sy,uint prevw=0,uint prevsx=0,uint prevsy=0,const float*skip=nullptr,const float*scales=nullptr){uint ww=width+2*sx;int x=int((window%(ww/8))*8+token%8)-int(sx),y=int((window/(ww/8))*8+token/8)-int(sy);if(x<0||y<0||x>=int(width)||y>=int(height))return 0.f;
 if constexpr(Merge){uint i=(uint(y)*width+uint(x))*32+channel,q=((uint(y)/2)*(width/2)+uint(x)/2)*32+channel;float high=__builtin_amdgcn_cvt_f32_fp8(int(reinterpret_cast<const unsigned char*>(skip)[i]),0);return Hrtz(Hrtz(in[q]*scales[channel])+high*scales[32+channel]);}
 if constexpr(Raw){uint px=uint(x)+prevsx,py=uint(y)+prevsy,tile=(py/8)*(prevw/8)+px/8,token0=(py%8)*8+px%8;return F(float(reinterpret_cast<const _Float16*>(in)[(tile*64+token0)*32+channel]));}
 return in[(uint(y)*width+uint(x))*32+channel];}

template<bool HalfOutput,bool Mapped=false,bool RawMapped=false,bool Merge=false,bool RgbHead=false,bool Finish=false,bool Main8=false,bool DownCrop=false,bool Prefix=false>
DEV void c32_fused_body(const float*raw_input,const float*fw,const float*w,float*out,uint windows,uint residual_mode,uint raw_output,uint sourcew=0,uint sourceh=0,uint sx=0,uint sy=0,uint prevw=0,uint prevsx=0,uint prevsy=0,const float*skip=nullptr,const float*scales=nullptr,const float*color=nullptr,const float*headw=nullptr,float*rgb=nullptr,float rgb_scale=0.f,float*main=nullptr,float*down=nullptr,const float*rgba=nullptr,const float*hist=nullptr,uint seed=0,uint temporal=0){
 uint window=bid();if(window>=windows)return;uint tid=__builtin_amdgcn_workitem_id_x(),wave=tid/32,first=wave*16,base=window*64;
 union Scratch {float raw[64*33];_Float16 ex[64*66];unsigned char hidden[64*132];};
 __attribute__((shared)) Scratch scratch;
 __attribute__((shared)) unsigned char packed[3*64*36];
#if HIP_C32_VT
 __attribute__((shared)) unsigned char vt[32*68]; // V transposed [col][key]: the AV B fragment is 8 contiguous bytes (the [key][col] rows needed 8 byte gathers per fragment)
#endif
#if HIP_C32_REGISTER_FFN
 h8 saved_ffn[2];
#if HIP_C32_ALIAS_FFN_V
 // Q/K writes are disjoint; V overwrites only after this wave finishes its FFN reads.
 // The final residual remains in saved_ffn registers.
 unsigned char*ffn8=packed+128*36;
#else
 __attribute__((shared)) unsigned char ffn8[64*36];
#endif
#else
 __attribute__((shared)) _Float16 ffn[64*34];
#endif
 if(residual_mode!=0&&residual_mode!=3)return;
 // Stage input64x36 only; FFN matrices are loaded directly from global weights.
#if HIP_C32_LANE_STAGE
 // Lane staging (HLSL ffn_fused shape): each wave stages its own 16 tokens; lanes 0..15 compute the mapped source
 // index of one token once, then for every token the 32 lanes load channel l of that token (one contiguous 128-byte
 // row) with the index broadcast by readlane. Same values as mapped_c32_input, 64 index computations per window
 // instead of 2048. The f16 copy (exact: every mapped source is H()-rounded) feeds the mode-0 residual.
 constexpr bool NeedIn16=Mapped&&!RawMapped;
#if HIP_C32_IN16_ALIAS && HIP_C32_FOLDED_FFN && HIP_C32_REGISTER_FFN
 _Float16*in16=reinterpret_cast<_Float16*>(scratch.ex); // aliased: in16 dies at the residual init, scratch is born in QKV
#else
 __attribute__((shared)) _Float16 in16[NeedIn16?64*34:1];
#endif
 f8 pre_acc[2];pre_acc[0]=f8{};pre_acc[1]=f8{};
 if constexpr(Prefix){uint l=tid%32;float*values=scratch.raw+wave*256;_Float16*pre16=reinterpret_cast<_Float16*>(scratch.raw+1024)+wave*512; // scratch is free until the FFN hidden writes
  if(l<16){float f[16]{};fused_prefix_values(rgba,hist,base+first+l,sourcew,seed,temporal,f);for(uint k=0;k<16;k++)values[l*16+k]=f[k];}
  sync_window();
  for(uint tile=0;tile<2;tile++){h8 a{};for(uint e=0;e<8;e++){uint k=tile*16+gr()*8+e;a[e]=(_Float16)(k<16?values[rc()*16+k]:0.f);}
   for(uint col=0;col<2;col++){h8 b{};for(uint e=0;e<8;e++){uint k=tile*16+gr()*8+e;b[e]=(_Float16)(k<16?fw[(col*16+rc())*16+k]:0.f);}
#if HIP_C32_TRANSPOSED_TAIL
    pre_acc[col]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b,a,pre_acc[col]);}} // transposed: lane = token row, e = channel
  for(uint col=0;col<2;col++){h8 hv{};for(uint e=0;e<8;e++){float v=Hrtz(pre_acc[col][e]);pre_acc[col][e]=v;hv[e]=(_Float16)v;}__builtin_memcpy(pre16+rc()*32+col*16+gr()*8,&hv,16);}
#else
    pre_acc[col]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,b,pre_acc[col]);}}
  for(uint col=0;col<2;col++)for(uint e=0;e<8;e++){float v=Hrtz(pre_acc[col][e]);pre_acc[col][e]=v;pre16[(gr()*8+e)*32+col*16+rc()]=(_Float16)v;}
#endif
  sync_window();
  #pragma unroll
  for(uint j=0;j<16;j++)packed[(first+j)*36+l]=static_cast<unsigned char>(fp8(float(pre16[j*32+l])));
 }else
 {uint l=tid%32;int mine=-1,mine2=-1;
  if constexpr(Mapped){if(l<16){uint token=first+l,ww=sourcew+2*sx;int x=int((window%(ww/8))*8+token%8)-int(sx),y=int((window/(ww/8))*8+token/8)-int(sy);
   if(x>=0&&y>=0&&x<int(sourcew)&&y<int(sourceh)){if constexpr(Merge){mine=int((uint(y)*sourcew+uint(x))*32);mine2=int(((uint(y)/2)*(sourcew/2)+uint(x)/2)*32);}else if constexpr(RawMapped){uint px=uint(x)+prevsx,py=uint(y)+prevsy,tile=(py/8)*(prevw/8)+px/8,token0=(py%8)*8+px%8;mine=int((tile*64+token0)*32);}else mine=int((uint(y)*sourcew+uint(x))*32);}}}
#if HIP_C32_STAGE_PREFETCH
#if HIP_C32_BYTE_CHAIN
  if constexpr(RawMapped){
   // Byte rows: lane l stages half of token l>>1 (16 bytes) with one 128-bit load; the source row comes from lanes 0..15
   // via ds_bpermute. 16 load requests + 16 byte LDS stores per wave become 1 + 4 (2026-09-23, -0.08/-0.06ms, bit-exact).
   int src=__builtin_amdgcn_ds_bpermute(int((l>>1)*4),mine);uint idx=src>=0?uint(src)+(l&1)*16:0u;
   i4 q;__builtin_memcpy(&q,reinterpret_cast<const unsigned char*>(raw_input)+idx,16);
   if(src<0)q=i4{0,0,0,0};
   __builtin_memcpy(packed+(first+(l>>1))*36+(l&1)*16,&q,16);
  }else if constexpr(Mapped){
   // f32 rows (128 B): lane l stages channels (l&7)*4..+3 of tokens (l>>3)+4k with four 128-bit loads (row indices via
   // ds_bpermute), one dword LDS store per token and two dword in16 stores; Merge also takes its 4 skip bytes as one b32.
   // 16 loads + 16 byte stores + 16 half stores per lane become 4 (+4) + 4 + 8. Per-element arithmetic unchanged.
   // 2026-09-23: -0.16/-0.10ms whole network, three replications, bit-exact (results/c32-vec-stage-20260923).
   const uint c0=(l&7)*4;float sc[4],sh[4];
   if constexpr(Merge){_Pragma("unroll") for(uint e=0;e<4;e++){sc[e]=scales[c0+e];sh[e]=scales[32+c0+e];}}
   _Pragma("unroll") for(uint k=0;k<4;k++){uint tok=k*4+(l>>3);int src=__builtin_amdgcn_ds_bpermute(int(tok*4),mine);uint idx=src>=0?uint(src):0u;
    f4 q;uint hb4=0;
    if constexpr(Merge){int src2=__builtin_amdgcn_ds_bpermute(int(tok*4),mine2);uint idx2=src2>=0?uint(src2):0u;__builtin_memcpy(&q,raw_input+idx2+c0,16);__builtin_memcpy(&hb4,reinterpret_cast<const unsigned char*>(skip)+idx+c0,4);}
    else __builtin_memcpy(&q,raw_input+idx+c0,16);
    float v[4];uint word=0;
    _Pragma("unroll") for(uint e=0;e<4;e++){float c;if constexpr(Merge){float high=__builtin_amdgcn_cvt_f32_fp8(int((hb4>>(8*e))&255u),0);c=Hrtz(Hrtz(q[e]*sc[e])+high*sh[e]);}else c=q[e];v[e]=src>=0?c:0.f;word|=(fp8(v[e])&255u)<<(8*e);}
    __builtin_memcpy(packed+(first+tok)*36+c0,&word,4);
    if constexpr(NeedIn16){_Float16 h[4];_Pragma("unroll") for(uint e=0;e<4;e++)h[e]=(_Float16)v[e];__builtin_memcpy(in16+(first+tok)*34+c0,h,8);}
   }
  }else{
#else
  {
#endif
  // Pass 1: every token's load is issued with a branch-free index (out-of-window tokens read element l of the buffer and
  // are zeroed below), so the 16 loads overlap; pass 2 converts. Values identical to the single-pass form.
  float lv[16];uint hb[16];int srcs[16];
  #pragma unroll
  for(uint j=0;j<16;j++){uint row=first+j;
   if constexpr(!Mapped){lv[j]=raw_input[(base+row)*32+l];srcs[j]=0;hb[j]=0;}
   else{int src=__builtin_amdgcn_readlane(mine,j);srcs[j]=src;uint idx=src>=0?uint(src):0u;hb[j]=0;
    if constexpr(Merge){int src2=__builtin_amdgcn_readlane(mine2,j);uint idx2=src2>=0?uint(src2):0u;hb[j]=reinterpret_cast<const unsigned char*>(skip)[idx+l];lv[j]=raw_input[idx2+l];}
#if HIP_C32_BYTE_CHAIN
    else if constexpr(RawMapped){hb[j]=reinterpret_cast<const unsigned char*>(raw_input)[idx+l];lv[j]=0.f;}
#else
    else if constexpr(RawMapped)lv[j]=float(reinterpret_cast<const _Float16*>(raw_input)[idx+l]);
#endif
    else lv[j]=raw_input[idx+l];}}
  #pragma unroll
  for(uint j=0;j<16;j++){uint row=first+j;float v;
   if constexpr(!Mapped)v=lv[j];
#if HIP_C32_BYTE_CHAIN
   else{float c;if constexpr(Merge){float high=__builtin_amdgcn_cvt_f32_fp8(int(hb[j]),0);c=Hrtz(Hrtz(lv[j]*scales[l])+high*scales[32+l]);}else if constexpr(RawMapped)c=0.f;else c=lv[j];v=srcs[j]>=0?c:0.f;}
   if constexpr(RawMapped)packed[row*36+l]=srcs[j]>=0?static_cast<unsigned char>(hb[j]):static_cast<unsigned char>(0);else packed[row*36+l]=static_cast<unsigned char>(fp8(v));
#else
   else{float c;if constexpr(Merge){float high=__builtin_amdgcn_cvt_f32_fp8(int(hb[j]),0);c=Hrtz(Hrtz(lv[j]*scales[l])+high*scales[32+l]);}else if constexpr(RawMapped)c=F(lv[j]);else c=lv[j];v=srcs[j]>=0?c:0.f;}
   packed[row*36+l]=static_cast<unsigned char>(fp8(v));
#endif
   if constexpr(NeedIn16)in16[row*34+l]=(_Float16)v;
  }  }}
#else
  #pragma unroll
  for(uint j=0;j<16;j++){uint row=first+j;float v=0.f;
   if constexpr(!Mapped)v=raw_input[(base+row)*32+l];
   else{int src=__builtin_amdgcn_readlane(mine,j);if(src>=0){if constexpr(Merge){int src2=__builtin_amdgcn_readlane(mine2,j);float high=__builtin_amdgcn_cvt_f32_fp8(int(reinterpret_cast<const unsigned char*>(skip)[uint(src)+l]),0);v=Hrtz(Hrtz(raw_input[uint(src2)+l]*scales[l])+high*scales[32+l]);}else if constexpr(RawMapped)v=F(float(reinterpret_cast<const _Float16*>(raw_input)[uint(src)+l]));else v=raw_input[uint(src)+l];}}
   packed[row*36+l]=static_cast<unsigned char>(fp8(v));
   if constexpr(NeedIn16)in16[row*34+l]=(_Float16)v;
  }}
#endif
#elif HIP_C32_INPUT_DWORD
 for(uint i=tid;i<64*9;i+=128){uint row=i/9,k=(i%9)*4,word=0;if(k<32)for(uint e=0;e<4;e++)word|=(fp8(Mapped?mapped_c32_input<RawMapped,Merge>(raw_input,window,row,k+e,sourcew,sourceh,sx,sy,prevw,prevsx,prevsy,skip,scales):raw_input[(base+row)*32+k+e])&255u)<<(e*8);__builtin_memcpy(packed+i*4,&word,4);}
#else
 for(uint i=tid;i<64*36;i+=128){uint row=i/36,k=i%36;packed[i]=k<32?static_cast<unsigned char>(fp8(Mapped?mapped_c32_input<RawMapped,Merge>(raw_input,window,row,k,sourcew,sourceh,sx,sy,prevw,prevsx,prevsy,skip,scales):raw_input[(base+row)*32+k])):0;}
#endif
#if HIP_C32_LOCAL_FFN_SYNC
 sync_owned_rows();
#else
 sync_window();
#endif
#if HIP_C32_FOLDED_FFN
 /* 2026-09-22 folded FFN: WMMA A and B operand registers share one layout, so swapping the expand operands yields the
    transposed product: each lane then holds one row with 8 consecutive hidden columns = exactly the A fragment the contract
    needs. The activated hidden block is packed to FP8 in registers and fed straight into the contract; scratch.hidden and the
    sync between expand and contract go away. Residual initialisation of acc stays before the contract, as before.
    Bit-exact; whole network about -0.10ms (1080) / -0.06ms (900), see Development/results/c32-transposed-ffn-20260922. */
 f8 acc[2];acc[0]=f8{};acc[1]=f8{};
#if HIP_C32_TRANSPOSED_TAIL
 if(residual_mode==0){for(uint ci=0;ci<2;ci++)for(uint e=0;e<8;e++){uint row=first+rc(),c=ci*16+gr()*8+e;float x; // transposed accumulator layout
#else
 if(residual_mode==0){for(uint ci=0;ci<2;ci++)for(uint e=0;e<8;e++){uint row=first+gr()*8+e,c=ci*16+rc();float x;
#endif
#if HIP_C32_LANE_STAGE
   if constexpr(Prefix)x=pre_acc[ci][e];else if constexpr(NeedIn16)x=float(in16[row*34+c]);else x=Mapped?mapped_c32_input<RawMapped,Merge>(raw_input,window,row,c,sourcew,sourceh,sx,sy,prevw,prevsx,prevsy,skip,scales):raw_input[(base+row)*32+c];
#else
   x=Mapped?mapped_c32_input<RawMapped,Merge>(raw_input,window,row,c,sourcew,sourceh,sx,sy,prevw,prevsx,prevsy,skip,scales):raw_input[(base+row)*32+c];
#endif
   acc[ci][e]=clampf(x,-448.f,448.f)*fw[8704+c];}}
#if HIP_C32_DIAG_WEIGHTS
 else{for(uint part=0;part<3;part++)for(uint ci=0;ci<2;ci++)for(uint kt=0;kt<2;kt++){i2 a=load8(packed+(first+rc())*36+kt*16+gr()*8),b=matrix8(fw,34944+((part*2+ci)*2+kt)*512+(gr()*16+rc())*8);acc[ci]=C32_WMMA_ACC(a,b,acc[ci]);}}
#else
 else{for(uint part=0;part<3;part++)for(uint ci=0;ci<2;ci++){uint c=ci*16+rc();float remaining=fw[8704+c],piece=0;for(uint j=0;j<=part;j++){piece=scale_piece(remaining);remaining-=piece;}
   for(uint kt=0;kt<2;kt++){i2 a{},b{};
#if HIP_C32_LOCAL_FFN_SYNC
    a=load8(packed+(first+rc())*36+kt*16+gr()*8);for(uint e=0;e<8;e++){uint k=kt*16+gr()*8+e;put_bits(b,e,k==c?fp8(piece):0u);}
#else
    for(uint e=0;e<8;e++){uint k=kt*16+gr()*8+e;put_bits(a,e,packed[(first+rc())*36+k]);put_bits(b,e,k==c?fp8(piece):0u);}
#endif
    acc[ci]=C32_WMMA_ACC(a,b,acc[ci]);}
  }}
#endif
#if HIP_C32_ABLATE!=1
 C32_LOOP for(uint t=0;t<8;t++){f8 sum{};uint col=t*16;
  for(uint kt=0;kt<2;kt++){i2 a{},b{};uint k=kt*16+gr()*8;a=load8(packed+(first+rc())*36+k);b=matrix8(fw+512,(col+rc())*32+k);sum=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(b,a,sum);}
  i2 h{};for(uint e=0;e<8;e++){float v=sum[e],g=clampf(v,-4.f,4.f),q=__builtin_fmaf(absf(g),-.055908203125f,.447265625f),p=__builtin_fmaf(g,q,.89453125f);put_bits(h,e,fp8(v*p));}
  {uint k=col+gr()*8;for(uint ci=0;ci<2;ci++){i2 b=matrix8(fw+4608,(ci*16+rc())*128+k);acc[ci]=C32_WMMA_ACC(h,b,acc[ci]);}}
 }
#endif
#else
#if HIP_C32_ABLATE!=1
 C32_LOOP for(uint col=0;col<128;col+=16){f8 sum{};
  for(uint kt=0;kt<2;kt++){i2 a{},b{};uint k=kt*16+gr()*8;a=load8(packed+(first+rc())*36+k);b=matrix8(fw+512,(col+rc())*32+k);sum=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,sum);}
  for(uint e=0;e<8;e++){float v=sum[e],g=clampf(v,-4.f,4.f),q=__builtin_fmaf(absf(g),-.055908203125f,.447265625f),p=__builtin_fmaf(g,q,.89453125f);scratch.hidden[(first+gr()*8+e)*132+col+rc()]=static_cast<unsigned char>(fp8(v*p));}
 }
#endif

#if HIP_C32_LOCAL_FFN_SYNC
 sync_owned_rows();
#else
 sync_window();
#endif
 f8 acc[2];acc[0]=f8{};acc[1]=f8{};
 if(residual_mode==0){for(uint ci=0;ci<2;ci++)for(uint e=0;e<8;e++){uint row=first+gr()*8+e,c=ci*16+rc();float x;
#if HIP_C32_LANE_STAGE
   if constexpr(Prefix)x=pre_acc[ci][e];else if constexpr(NeedIn16)x=float(in16[row*34+c]);else x=Mapped?mapped_c32_input<RawMapped,Merge>(raw_input,window,row,c,sourcew,sourceh,sx,sy,prevw,prevsx,prevsy,skip,scales):raw_input[(base+row)*32+c];
#else
   x=Mapped?mapped_c32_input<RawMapped,Merge>(raw_input,window,row,c,sourcew,sourceh,sx,sy,prevw,prevsx,prevsy,skip,scales):raw_input[(base+row)*32+c];
#endif
   acc[ci][e]=clampf(x,-448.f,448.f)*fw[8704+c];}}
#if HIP_C32_DIAG_WEIGHTS
 else{for(uint part=0;part<3;part++)for(uint ci=0;ci<2;ci++)for(uint kt=0;kt<2;kt++){i2 a=load8(packed+(first+rc())*36+kt*16+gr()*8),b=matrix8(fw,34944+((part*2+ci)*2+kt)*512+(gr()*16+rc())*8);acc[ci]=C32_WMMA_ACC(a,b,acc[ci]);}}
#else
 else{for(uint part=0;part<3;part++)for(uint ci=0;ci<2;ci++){uint c=ci*16+rc();float remaining=fw[8704+c],piece=0;for(uint j=0;j<=part;j++){piece=scale_piece(remaining);remaining-=piece;}
   for(uint kt=0;kt<2;kt++){i2 a{},b{};
#if HIP_C32_LOCAL_FFN_SYNC
    a=load8(packed+(first+rc())*36+kt*16+gr()*8);for(uint e=0;e<8;e++){uint k=kt*16+gr()*8+e;put_bits(b,e,k==c?fp8(piece):0u);}
#else
    for(uint e=0;e<8;e++){uint k=kt*16+gr()*8+e;put_bits(a,e,packed[(first+rc())*36+k]);put_bits(b,e,k==c?fp8(piece):0u);}
#endif
    acc[ci]=C32_WMMA_ACC(a,b,acc[ci]);}
  }}
#endif
#if HIP_C32_ABLATE!=1
 C32_LOOP for(uint chunk=0;chunk<4;chunk++)for(uint ci=0;ci<2;ci++){
  for(uint kt=0;kt<2;kt++){i2 a{},b{};uint k=chunk*32+kt*16+gr()*8;a=load8(scratch.hidden+(first+rc())*132+k);b=matrix8(fw+4608,(ci*16+rc())*128+k);acc[ci]=C32_WMMA_ACC(a,b,acc[ci]);}
 }
#endif
#endif

#if HIP_C32_ALIAS_FFN_V && HIP_C32_REGISTER_FFN
 // Preserve the validated contraction-to-FFN phase boundary.
 sync_window();
#endif
#if HIP_C32_TRANSPOSED_TAIL
 for(uint ci=0;ci<2;ci++){i2 fb{};for(uint e=0;e<8;e++){float v=Hrtz(acc[ci][e]);saved_ffn[ci][e]=(_Float16)v;put_bits(fb,e,fp8(v));}__builtin_memcpy(ffn8+(first+rc())*36+ci*16+gr()*8,&fb,8);} // 8 consecutive channels of the lane's token
#else
 for(uint ci=0;ci<2;ci++)for(uint e=0;e<8;e++){
#if HIP_C32_REGISTER_FFN
  float v=Hrtz(acc[ci][e]);saved_ffn[ci][e]=(_Float16)v;ffn8[(first+gr()*8+e)*36+ci*16+rc()]=static_cast<unsigned char>(fp8(v));
#else
  ffn[(first+gr()*8+e)*34+ci*16+rc()]=(_Float16)Hrtz(acc[ci][e]);
#endif
 }
#endif
 sync_window();
 // A single raw QKV plane is reused after its norm has been packed separately.
 C32_LOOP for(uint part=0;part<3;part++){
  for(uint col=0;col<32;col+=16){f8 acc{};
   for(uint kt=0;kt<2;kt++){i2 a{},b{};
#if HIP_C32_REGISTER_FFN
    a=load8(ffn8+(first+rc())*36+kt*16+gr()*8);
#endif
    for(uint e=0;e<8;e++){uint k=kt*16+gr()*8+e;
#if !HIP_C32_REGISTER_FFN
     put(a,e,float(ffn[(first+rc())*34+k]));
#endif
    }b=matrix8(w,part*1024+(col+rc())*32+kt*16+gr()*8);acc=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,acc);}
   for(uint e=0;e<8;e++)scratch.raw[(first+gr()*8+e)*33+col+rc()]=acc[e];
  }
  sync_owned_rows();f8 sum{};
  if(part<2)for(uint kt=0;kt<2;kt++){h8 a{},ones{};for(uint e=0;e<8;e++){uint k=kt*16+gr()*8+e;float v=scratch.raw[(first+rc())*33+k];a[e]=(_Float16)(v*v);ones[e]=(_Float16)1.f;}sum=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,ones,sum);}
  for(uint col=0;col<32;col+=16)for(uint e=0;e<8;e++){uint row=first+gr()*8+e;float v=scratch.raw[row*33+col+rc()];if(part<2)v*=__builtin_amdgcn_rsqf(maxf(sum[e],6.198883056640625e-5f))*(part==0?w[8192]:1.f);
#if HIP_C32_VT
   if(part==2)vt[(col+rc())*68+row]=static_cast<unsigned char>(fp8(v));else
#endif
   packed[(part*64+row)*36+col+rc()]=static_cast<unsigned char>(fp8(v));}
  sync_owned_rows();
 }
#if HIP_C32_LOCAL_QKV_SYNC
 sync_window(); // Attention reads K/V rows owned by all four waves.
#endif
 // Q/K packed rows: 36-byte stride. Exponent values stored exactly as half bits.
#if HIP_C32_LOCAL_ATTN_SYNC
 // Wave-private from here on: scores/ex rows, probability rows (in the wave's own ex slice) and the AV bytes (own Q rows)
 // are read only by the wave that wrote them; K/V rows were published by the group sync above.
 unsigned char*prob=reinterpret_cast<unsigned char*>(scratch.ex)+first*132;
#define ATTN_SYNC() sync_owned_rows()
#else
 unsigned char*prob=packed;
#define ATTN_SYNC() sync_window()
#endif
 // Keep the generated half exponent values for probability normalization.
 // Both fixed-index loops must unroll: dynamic vector-array indexing expands
 // into thousands of lane/select instructions on COMGR (2026-09-19).
 h8 saved_ex[4];
 f8 av[2];av[0]=f8{};av[1]=f8{};
#if HIP_C32_ABLATE!=2
 _Pragma("unroll 4") for(uint key=0;key<64;key+=16){f8 acc{};
  for(uint kt=0;kt<2;kt++){i2 a{},b{};uint k=kt*16+gr()*8;a=load8(packed+(first+rc())*36+k);b=load8(packed+(64+key+rc())*36+k);acc=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,acc);}
  _Pragma("unroll 8") for(uint e=0;e<8;e++){uint q=first+gr()*8+e,k=key+rc();float score=acc[e]+w[4096+q*64+k],affine=clampf(score*.044921875f+1.30078125f,1.03125f,1.5693359375f);uint ah=(bits(affine)>>13)-0x1c000u;ushort hb=static_cast<ushort>(((ah<<5)+0x8000u)&65535u);scratch.ex[q*66+k]=__builtin_bit_cast(_Float16,hb);saved_ex[key/16][e]=__builtin_bit_cast(_Float16,hb);}
 }
 ATTN_SYNC();
 // Scores no longer consume Q/K; their 4608 bytes become 64 probability rows of 68 bytes.
 {f8 sum{};for(uint kt=0;kt<4;kt++){h8 a{},ones{};for(uint e=0;e<8;e++){uint k=kt*16+gr()*8+e;a[e]=scratch.ex[(first+rc())*66+k];ones[e]=(_Float16)1.f;}sum=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,ones,sum);}
#if HIP_C32_LOCAL_ATTN_SYNC
  float pv[4][8];for(uint key=0;key<64;key+=16)for(uint e=0;e<8;e++){uint row=first+gr()*8+e,k=key+rc();pv[key/16][e]=float(scratch.ex[row*66+k])*norm_inverse(sum[e]);}
  ATTN_SYNC(); // every lane of this wave has read its ex values before the rows are reused for probabilities
  for(uint key=0;key<64;key+=16)for(uint e=0;e<8;e++){uint row=gr()*8+e,k=key+rc();prob[row*68+k]=static_cast<unsigned char>(fp8(pv[key/16][e]));}
#else
  _Pragma("unroll 4") for(uint key=0;key<64;key+=16) _Pragma("unroll 8") for(uint e=0;e<8;e++){uint row=first+gr()*8+e,k=key+rc();float v=float(saved_ex[key/16][e])*norm_inverse(sum[e]);prob[row*68+k]=static_cast<unsigned char>(fp8(v));}
#endif
 }
 ATTN_SYNC();
 // Both AV column fragments stay live until every wave has finished reading probability rows.
 for(uint ci=0;ci<2;ci++){f8 acc{};uint col=ci*16;
  for(uint kt=0;kt<4;kt++){i2 a{},b{};
#if HIP_C32_VT
   __builtin_memcpy(&a,prob+(first+rc())*68+kt*16+gr()*8,8);__builtin_memcpy(&b,vt+(col+rc())*68+kt*16+gr()*8,8);
#else
   for(uint e=0;e<8;e++){uint k=kt*16+gr()*8+e;
#if HIP_C32_LOCAL_ATTN_SYNC
   put_bits(a,e,prob[rc()*68+k]);
#else
   put_bits(a,e,prob[(first+rc())*68+k]);
#endif
   put_bits(b,e,packed[(128+k)*36+col+rc()]);}
#endif
   acc=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,acc);}av[ci]=acc;
 }
#endif

 ATTN_SYNC();
 for(uint ci=0;ci<2;ci++)for(uint e=0;e<8;e++)packed[(first+gr()*8+e)*36+ci*16+rc()]=static_cast<unsigned char>(fp8(av[ci][e]));
 ATTN_SYNC();
#if HIP_C32_TRANSPOSED_TAIL
#if HIP_C32_ABLATE!=3
 // Transposed projection: lane = token row, e = channel within an 8-run. The per-channel scale becomes 16 hoisted loads.
 float pwt[2][8];for(uint ci=0;ci<2;ci++)for(uint e=0;e<8;e++)pwt[ci][e]=w[8193+ci*16+gr()*8+e];
 C32_LOOP for(uint col=0;col<32;col+=16){f8 acc{};
  for(uint kt=0;kt<2;kt++){i2 a{},b{};for(uint e=0;e<8;e++){uint k=kt*16+gr()*8+e;put_bits(a,e,packed[(first+rc())*36+k]);}b=matrix8(w,3072+(col+rc())*32+kt*16+gr()*8);acc=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(b,a,acc);}
  const uint row=first+rc(),c0=col+gr()*8,i0=(base+row)*32+c0;h8 hv{},ho{};i2 fb{};f8 fv{};
  for(uint e=0;e<8;e++){float residual=float(saved_ffn[col/16][e]),prod=residual*pwt[col/16][e],v=Hrtz(acc[e]+prod);hv[e]=(_Float16)v;
   if constexpr(Finish||RgbHead){}
#if HIP_C32_BYTE_CHAIN
   else if constexpr(HalfOutput){put_bits(fb,e,fp8(F(float((_Float16)v))));ho[e]=(_Float16)F(v);}else fv[e]=raw_output?v:F(v);}
#else
   else if constexpr(HalfOutput)ho[e]=(_Float16)(raw_output?v:F(v));else fv[e]=raw_output?v:F(v);}
#endif
  if constexpr(RgbHead||Finish)__builtin_memcpy(scratch.ex+row*34+c0,&hv,16); // raw_output is 1 here: v is the half the finish/head would have read back
  if constexpr(Finish){if(out)__builtin_memcpy(reinterpret_cast<_Float16*>(out)+i0,&hv,16);}
  else if constexpr(RgbHead){}
#if HIP_C32_BYTE_CHAIN
  else if constexpr(HalfOutput){if(raw_output)__builtin_memcpy(reinterpret_cast<unsigned char*>(out)+i0,&fb,8);else __builtin_memcpy(reinterpret_cast<_Float16*>(out)+i0,&ho,16);}else __builtin_memcpy(out+i0,&fv,32);
#else
  else if constexpr(HalfOutput)__builtin_memcpy(reinterpret_cast<_Float16*>(out)+i0,&ho,16);else __builtin_memcpy(out+i0,&fv,32);
#endif
 }
#endif
#else
#if HIP_C32_HOIST_PW
 // The projection scale is per column: read the lane's two values once. Inside the loop the reads sat between the out[]
 // stores, so the compiler re-issued global_load + s_wait_loadcnt 0 for every one of the 16 outputs (ISA, 2026-09-17).
 const float pw0=w[8193+rc()],pw1=w[8193+16+rc()];
#endif
#if HIP_C32_ABLATE!=3
 C32_LOOP for(uint col=0;col<32;col+=16){f8 acc{};
  for(uint kt=0;kt<2;kt++){i2 a{},b{};for(uint e=0;e<8;e++){uint k=kt*16+gr()*8+e;put_bits(a,e,packed[(first+rc())*36+k]);}b=matrix8(w,3072+(col+rc())*32+kt*16+gr()*8);acc=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,acc);}
  for(uint e=0;e<8;e++){uint i=(base+first+gr()*8+e)*32+col+rc();float residual;
#if HIP_C32_REGISTER_FFN
   residual=float(saved_ffn[col/16][e]);
#else
   residual=float(ffn[(first+gr()*8+e)*34+col+rc()]);
#endif
#if HIP_C32_HOIST_PW
   float prod=residual*(col==0?pw0:pw1),v=Hrtz(acc[e]+prod);
#else
   float prod=residual*w[8193+col+rc()],v=Hrtz(acc[e]+prod);
#endif
   if constexpr(RgbHead||Finish)scratch.ex[(first+gr()*8+e)*34+col+rc()]=(_Float16)v; // raw_output is 1 here: v is the half the finish/head would have read back
   if constexpr(Finish){if(out)reinterpret_cast<_Float16*>(out)[i]=(_Float16)v;}
   else if constexpr(RgbHead){}
#if HIP_C32_BYTE_CHAIN
   else if constexpr(HalfOutput){if(raw_output)reinterpret_cast<unsigned char*>(out)[i]=static_cast<unsigned char>(fp8(F(float((_Float16)v))));else reinterpret_cast<_Float16*>(out)[i]=(_Float16)F(v);}else out[i]=raw_output?v:F(v);}
#else
   else if constexpr(HalfOutput)reinterpret_cast<_Float16*>(out)[i]=(_Float16)(raw_output?v:F(v));else out[i]=raw_output?v:F(v);}
#endif
 }
#endif
#endif /* HIP_C32_TRANSPOSED_TAIL */

 if constexpr(Finish){
  // c32_finish_crop_half / c32_finish_fast_half_main8 in the epilogue (HLSL DLSS5_C32_EPILOGUE finish shape): the wave's 16
  // tokens are two 8-pixel rows of the window, so both the cropped main (F of the half) and the 2x2 downsample (same
  // Hrtz order as the finish kernels) come from the wave's own LDS rows; the raw write is skipped when out is null.
  sync_owned_rows();uint l=tid%32,ww=sourcew+2*sx,tx=(window%(ww/8))*8,ty=(window/(ww/8))*8;
  if(main){
   #pragma unroll
   for(uint j=0;j<16;j++){uint tok=first+j;int x=int(tx+tok%8)-int(sx),y=int(ty+tok/8)-int(sy);if(x<0||y<0||x>=int(sourcew)||y>=int(sourceh))continue;uint i=(uint(y)*sourcew+uint(x))*32+l;float v=F(float(scratch.ex[tok*34+l]));
    if constexpr(Main8)reinterpret_cast<unsigned char*>(main)[i]=static_cast<unsigned char>(fp8(v));else main[i]=v;}}
  if(down){
   #pragma unroll
   for(uint k=0;k<4;k++){uint t0=first+2*k;float top=Hrtz(float(scratch.ex[t0*34+l])+float(scratch.ex[(t0+1)*34+l])),bottom=Hrtz(float(scratch.ex[(t0+8)*34+l])+float(scratch.ex[(t0+9)*34+l]));uint x=tx+t0%8,y=ty+t0/8;
    if constexpr(DownCrop){int cx=int(x/2)-int(sx/2),cy=int(y/2)-int(sy/2);if(cx<0||cy<0||cx>=int(sourcew/2)||cy>=int(sourceh/2))continue;down[(uint(cy)*(sourcew/2)+uint(cx))*32+l]=F(Hrtz(Hrtz(top+bottom)*.25f));} // mh_shift_crop of the down grid folded in
    else down[((y/2)*(ww/2)+x/2)*32+l]=F(Hrtz(Hrtz(top+bottom)*.25f));}}
 }
 if constexpr(RgbHead){
  // hip_post_head_fast_half in the epilogue (HLSL DLSS5_C32_EPILOGUE shape): no raw write/read-back, no head dispatch.
  // One lane per token shares feature loads across RGB; each 32-term dot retains its sequential f32 order.
  sync_owned_rows();uint l=tid%32,ww=sourcew+2*sx;
  if(l<16){uint tok=first+l;int x=int((window%(ww/8))*8+tok%8)-int(sx),y=int((window/(ww/8))*8+tok/8)-int(sy);
   if(x>=0&&y>=0&&x<int(sourcew)&&y<int(sourceh)){uint p=uint(y)*sourcew+uint(x);float a0=0.f,a1=0.f,a2=0.f;
    for(uint j=0;j<32;j++){float f=float(scratch.ex[tok*34+j]);a0+=f*headw[j];a1+=f*headw[32+j];a2+=f*headw[64+j];}
    float sums[3]={a0,a1,a2};for(uint row=0;row<3;row++){float acc=Hrtz(sums[row]),centered=color[p*4+row]*.125f-.0625f,v=(acc*rgb_scale+centered)*8.f+.5f;rgb[p*3+row]=clampf(v,0.f,1.f);}
   }
  }
 }
}

KERNEL __attribute__((amdgpu_flat_work_group_size(128,128))) C32_OCC
void c32_fast_ffn_attention_fused(const float*in,const float*fw,const float*w,float*out,uint windows,uint mode,uint raw){c32_fused_body<false>(in,fw,w,out,windows,mode,raw);}
KERNEL __attribute__((amdgpu_flat_work_group_size(128,128))) C32_OCC
void c32_fast_ffn_attention_fused_half(const float*in,const float*fw,const float*w,float*out,uint windows,uint mode,uint raw){c32_fused_body<true>(in,fw,w,out,windows,mode,raw);}

KERNEL __attribute__((amdgpu_flat_work_group_size(128,128))) C32_OCC
void c32_fast_ffn_attention_fused_half_mapped(const float*in,const float*fw,const float*w,float*out,uint windows,uint mode,uint raw,uint width,uint height,uint sx,uint sy){c32_fused_body<true,true>(in,fw,w,out,windows,mode,raw,width,height,sx,sy);}

KERNEL __attribute__((amdgpu_flat_work_group_size(128,128))) C32_OCC
void c32_fast_ffn_attention_fused_half_chain(const float*in,const float*fw,const float*w,float*out,uint windows,uint mode,uint raw,uint width,uint height,uint sx,uint sy,uint prevw,uint prevsx,uint prevsy){c32_fused_body<true,true,true>(in,fw,w,out,windows,mode,raw,width,height,sx,sy,prevw,prevsx,prevsy);}

// Finish in the epilogue (host option c32_finish_fused). Block 0: unmapped input, main8 bytes + down over the w x h grid
// (sourcew/sourceh carry w/h, no shift). Chain end: previous raw tiles in, cropped f32 main (+ down for block 4), no raw.
KERNEL __attribute__((amdgpu_flat_work_group_size(128,128))) C32_OCC
void c32_fast_ffn_attention_fused_half_finish_main8(const float*in,const float*fw,const float*w,float*main8,float*down,uint windows,uint mode,uint raw,uint width,uint height){c32_fused_body<true,false,false,false,false,true,true>(in,fw,w,nullptr,windows,mode,raw,width,height,0,0,0,0,0,nullptr,nullptr,nullptr,nullptr,nullptr,0.f,main8,down);}
KERNEL __attribute__((amdgpu_flat_work_group_size(128,128))) C32_OCC
void c32_fast_ffn_attention_fused_half_chain_finish(const float*in,const float*fw,const float*w,float*main,float*down,uint windows,uint mode,uint raw,uint width,uint height,uint sx,uint sy,uint prevw,uint prevsx,uint prevsy){c32_fused_body<true,true,true,false,false,true,false>(in,fw,w,nullptr,windows,mode,raw,width,height,sx,sy,prevw,prevsx,prevsy,nullptr,nullptr,nullptr,nullptr,nullptr,0.f,main,down);}
// Block 0 with the prefix inlined (host option prefix_inline): rgba raster + history in, main8 bytes + down out, no prefix buffer.
KERNEL __attribute__((amdgpu_flat_work_group_size(128,128))) C32_OCC
void c32_fast_ffn_attention_fused_half_prefix_finish_main8(const float*rgba,const float*hist,const float*fw,const float*w,float*main8,float*down,uint windows,uint mode,uint raw,uint width,uint height,uint seed,uint temporal){c32_fused_body<true,false,false,false,false,true,true,false,true>(nullptr,fw,w,nullptr,windows,mode,raw,width,height,0,0,0,0,0,nullptr,nullptr,nullptr,nullptr,nullptr,0.f,main8,down,rgba,hist,seed,temporal);}
// Chain end with the down grid already cropped to (w/2)x(h/2) (host option down_crop_fused): no mh_shift_crop launch.
KERNEL __attribute__((amdgpu_flat_work_group_size(128,128))) C32_OCC
void c32_fast_ffn_attention_fused_half_chain_finish_dcrop(const float*in,const float*fw,const float*w,float*main,float*down,uint windows,uint mode,uint raw,uint width,uint height,uint sx,uint sy,uint prevw,uint prevsx,uint prevsy){c32_fused_body<true,true,true,false,false,true,false,true>(in,fw,w,nullptr,windows,mode,raw,width,height,sx,sy,prevw,prevsx,prevsy,nullptr,nullptr,nullptr,nullptr,nullptr,0.f,main,down);}
// Post-merge C32 with the rgb head in the epilogue (host option post_head_fused): out unused; rgb = W*H*3 floats.
KERNEL __attribute__((amdgpu_flat_work_group_size(128,128))) C32_OCC
void c32_post_merge_head_half(const float*low,const float*skip,const float*scales,const float*fw,const float*w,const float*color,const float*headw,float*rgb,uint windows,uint width,uint height,uint sx,uint sy,float scale){c32_fused_body<true,true,false,true,true>(low,fw,w,nullptr,windows,0,1,width,height,sx,sy,0,0,0,skip,scales,color,headw,rgb,scale);}
KERNEL __attribute__((amdgpu_flat_work_group_size(128,128))) C32_OCC
void c32_post_merge_fused_half(const float*low,const float*skip,const float*scales,const float*fw,const float*w,float*out,uint windows,uint width,uint height,uint sx,uint sy){c32_fused_body<true,true,false,true>(low,fw,w,out,windows,0,1,width,height,sx,sy,0,0,0,skip,scales);}

// small-wins-20260930 (from mochizuki-022 I/P): exact representation changes only; every arithmetic rounding kept, its half bits reused.
#ifndef CW_INPUT_HALF
#define CW_INPUT_HALF 0 /* bit0: post Hrtz value; bit1: mapped input (half/FP8 producers); bit2: Up H() value -- skip float((_Float16)v) */
#endif
#ifndef CW_PREFIX_HALF_SOURCE
#define CW_PREFIX_HALF_SOURCE 0 /* prefix: keep the RTZ half bit pattern through the shuffle into the WMMA operand */
#endif
#if CW_PREFIX_HALF_SOURCE
using cw_pv_t=_Float16;
DEV cw_pv_t cw_prefix_rtz(float x){uint p;asm("v_cvt_pkrtz_f16_f32 %0, %1, %1":"=v"(p):"v"(x));return __builtin_bit_cast(_Float16,(unsigned short)p);}
DEV cw_pv_t cw_prefix_shuffle(cw_pv_t x,uint lane){uint p=uint(__builtin_bit_cast(unsigned short,x));p=uint(__builtin_amdgcn_ds_bpermute(lane*4,int(p)));return __builtin_bit_cast(_Float16,(unsigned short)p);}
#else
using cw_pv_t=float;
DEV cw_pv_t cw_prefix_rtz(float x){return Hrtz(x);}
DEV cw_pv_t cw_prefix_shuffle(cw_pv_t x,uint lane){return __builtin_bit_cast(float,__builtin_amdgcn_ds_bpermute(lane*4,__builtin_bit_cast(int,x)));}
#endif
// One complete 64-token C32 window per wave; chain FP8 input/output.
#ifndef CW_LAUNDER
#define CW_LAUNDER 0
#endif
#ifndef CW_ROLL_HIDDEN
#define CW_ROLL_HIDDEN 0
#endif
#ifndef CW_QKV_FENCE
#define CW_QKV_FENCE 0
#endif
#ifndef CW_ROLL_WINDOW
#define CW_ROLL_WINDOW 0
#endif
#ifndef CW_PREFIX_SPLIT
#define CW_PREFIX_SPLIT 0 /* 1: prefix features split across both half-waves (bit-exact; 1080 -0.03ms, results/vit-attn-c32-input-20260926) */
#endif
#ifndef CW_TAIL_QUAD
#define CW_TAIL_QUAD 0 /* 1: finish/prefix tail walks 2x2 quads once: 4 LDS reads feed both main and down (was 64+64 reads), 32-bit byte offsets for stores; arithmetic unchanged */
#endif
#ifndef CW_VEC_INPUT
#define CW_VEC_INPUT 0 /* 1: mapped/post input rows read as b128/b64 instead of per-element loads (bit-exact; results/c32-wave-phase-20260926) */
#endif
// C32 ACO audit (Development/results/c32-aco-20260927). Switches default off; the build recipe selects the accepted pair.
#ifndef CW_FMED3_CLAMP
#define CW_FMED3_CLAMP 0
#endif
#ifndef CW_DIRECT_OUT
#define CW_DIRECT_OUT 0
#endif
#ifndef CW_RTZ_PAIR
#define CW_RTZ_PAIR 0
#endif
DEV float cw_clampf(float x,float lo,float hi){
#if CW_FMED3_CLAMP
 return __builtin_amdgcn_fmed3f(x,lo,hi);
#else
 return clampf(x,lo,hi);
#endif
}
// Activation clamp only: the polynomial retains all original rounding
// boundaries. fmed3(x,-4,4) returns -4 for NaN just like the ternary clamp.
#ifndef CW_ACT_FMED3
#define CW_ACT_FMED3 0
#endif
DEV float cw_act_gate(float x){
#if CW_ACT_FMED3
 return __builtin_amdgcn_fmed3f(x,-4.f,4.f);
#else
 return cw_clampf(x,-4.f,4.f);
#endif
}
// Two independent RTZ conversions in the instruction we already use; no packed-half arithmetic,
// no MODE change, and no contraction across an arithmetic/rounding boundary.
#if CW_RTZ_PAIR
using cw_h2=_Float16 __attribute__((ext_vector_type(2)));
DEV h8 cw_rtz_half8(f8 x){h8 out;_Pragma("unroll") for(uint j=0;j<4;j++){
 uint packed;asm("v_cvt_pkrtz_f16_f32 %0, %1, %2" : "=v"(packed) : "v"(x[2*j]),"v"(x[2*j+1]));
 cw_h2 pair=__builtin_bit_cast(cw_h2,packed);out[2*j]=pair[0];out[2*j+1]=pair[1];}return out;}
#endif
DEV float cw_output_value(float v){
#if CW_DIRECT_OUT
 // fp8(F(v)) == fp8(v), except F canonicalizes an exact -0. Preserve underflow-generated -0.
 return v+0.f;
#else
 return F(float((_Float16)v));
#endif
}
#ifndef CW_PREFIX_DIRECT_OUT
#define CW_PREFIX_DIRECT_OUT 0 /* prefix byte tail: avoid fp8(F(half-value)); retain exact-zero canonicalization */
#endif
#ifndef CW_PREFIX_FULL_TILE
#define CW_PREFIX_FULL_TILE 0 /* prefix only: host emits complete 8x8 tiles, sx=sy=0, so its per-pixel bounds tests are redundant */
#endif
#ifndef CW_POST_HEAD_VEC
#define CW_POST_HEAD_VEC 0 /* RGB head: read each contiguous eight-half group together, retain channel reduction order */
#endif
#ifndef CW_POST_FULL_TILE
#define CW_POST_FULL_TILE 0 /* specialize interior RGB output windows once per wave */
#endif
#ifndef CW_FINISH_FULL_TILE
#define CW_FINISH_FULL_TILE 0 /* specialize interior main/down output windows once per wave */
#endif
DEV uint cw_prefix_byte(float v){
#if CW_PREFIX_DIRECT_OUT
 return fp8(v+0.f);
#else
 return fp8(F(v));
#endif
}
struct cw_true {constexpr operator bool()const{return true;}};
struct cw_false {constexpr operator bool()const{return false;}};
using cw_i16=int __attribute__((ext_vector_type(16)));
// CW_PACK8 (2026-09-27, results/pack8-20260927): the same bytes as put_bits(a,e,fp8(x)) -- the identical clamp, then
// v_cvt_pk_fp8_f32 on two values at once written straight into the fragment word instead of one cvt + and/shift/or per byte.
// fp8() has no -0 select in C32, so nothing else changes.
#ifndef CW_PACK8
#define CW_PACK8 0
#endif
#if CW_PACK8
#if HIP_FP8_SAT_MODE==3
DEV float cw_z(float x){return __builtin_amdgcn_fmed3f(x,-448.f,448.f);}
#elif HIP_C32_BRANCHLESS_F
DEV float cw_z(float x){return __builtin_fminf(__builtin_fmaxf(x,-448.f),448.f);}
#else
DEV float cw_z(float x){return clampf(x,-448.f,448.f);}
#endif
#ifndef CW_PACK_CENSUS
#define CW_PACK_CENSUS 0 /* diagnostic only: per-kind, per-site pack counts / nonfinite / >448 / >1 / maximum absolute finite value */
#endif
#ifndef CW_PACK_MODE_MASK
#define CW_PACK_MODE_MASK 0 /* bit per pack site: input, hidden, feature, QKV, probability, AV, output */
#endif
#if CW_PACK_CENSUS
extern "C" __attribute__((device)) uint cw_pack_census[6*7*5]={};
template<uint Site,uint Kind> DEV void cw_census(const float*t){
 uint lane=__builtin_amdgcn_workitem_id_x(),flags=0,mx=0;
 for(uint e=0;e<8;e++){uint ab=bits(t[e])&0x7fffffffu;flags|=(ab>=0x7f800000u?1u:0u)|(ab>0x43e00000u?2u:0u)|(ab>0x3f800000u?4u:0u);if(ab<0x7f800000u&&ab>mx)mx=ab;}
 for(uint sh=16;sh;sh>>=1){uint other=uint(__builtin_amdgcn_ds_bpermute((lane^sh)*4,int(mx)));if(other>mx)mx=other;flags|=uint(__builtin_amdgcn_ds_bpermute((lane^sh)*4,int(flags)));}
 if(lane==0){uint*p=cw_pack_census+(Kind*7+Site)*5;__hip_atomic_fetch_add(p,1u,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);
  if(flags&1)__hip_atomic_fetch_add(p+1,1u,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);
  if(flags&2)__hip_atomic_fetch_add(p+2,1u,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);
  if(flags&4)__hip_atomic_fetch_add(p+3,1u,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);
  __hip_atomic_fetch_max(p+4,mx,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);}
}
#endif
template<uint Site,uint Kind> DEV i2 cw_pack8(const float*t){
#if CW_PACK_CENSUS
 cw_census<Site,Kind>(t);
#endif
// Keep MODE and its four conversions indivisible: LLVM can hoist the FP8 intrinsics outside s_setreg.
 if constexpr((CW_PACK_MODE_MASK&(1u<<Site))!=0){int lo,hi;
  asm volatile("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1\n\t"
               "v_cvt_pk_fp8_f32 %0, %2, %3\n\t"
               "v_cvt_pk_fp8_f32 %1, %6, %7\n\t"
               "v_cvt_pk_fp8_f32 %0, %4, %5 op_sel:[0,0,1]\n\t"
               "v_cvt_pk_fp8_f32 %1, %8, %9 op_sel:[0,0,1]\n\t"
               "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 0"
               : "=&v"(lo),"=&v"(hi) : "v"(t[0]),"v"(t[1]),"v"(t[2]),"v"(t[3]),"v"(t[4]),"v"(t[5]),"v"(t[6]),"v"(t[7]) : "memory");
  return i2{lo,hi};}
 int lo=__builtin_amdgcn_cvt_pk_fp8_f32(cw_z(t[0]),cw_z(t[1]),0,false);lo=__builtin_amdgcn_cvt_pk_fp8_f32(cw_z(t[2]),cw_z(t[3]),lo,true);
 int hi=__builtin_amdgcn_cvt_pk_fp8_f32(cw_z(t[4]),cw_z(t[5]),0,false);hi=__builtin_amdgcn_cvt_pk_fp8_f32(cw_z(t[6]),cw_z(t[7]),hi,true);
 i2 a;a[0]=lo;a[1]=hi;return a;}

#define CW_Q8_DECL(A) i2 A;float A##_t[8]
#define CW_Q8_SET(A,E,X) A##_t[E]=(X)
#define CW_Q8_END_SITE(A,S) A=cw_pack8<S,cw_kind>(A##_t)
#define CW_Q8_END(A) CW_Q8_END_SITE(A,0)
#else
#define CW_Q8_DECL(A) i2 A{}
#define CW_Q8_SET(A,E,X) put_bits(A,E,fp8(X))
#define CW_Q8_END(A) (void)0
#define CW_Q8_END_SITE(A,S) (void)0
#endif

DEV i2 cw_input(const unsigned char*in,uint win,uint qt,uint ci,uint width,uint height,uint sx,uint sy,uint prevw,uint px,uint py,uint r,uint g){
 uint ww=width+2*sx;int x=int((win%(ww/8))*8+(qt*16+r)%8)-int(sx),y=int((win/(ww/8))*8+(qt*16+r)/8)-int(sy);i2 a{};
 if(x>=0&&y>=0&&x<int(width)&&y<int(height)){uint xx=uint(x)+px,yy=uint(y)+py,tile=(yy/8)*(prevw/8)+xx/8,tok=(yy%8)*8+xx%8;__builtin_memcpy(&a,in+(tile*64+tok)*32+ci*16+g*8,8);}return a;
}
#ifndef CW_PREPOST_BYTE
#define CW_PREPOST_BYTE 0 /* 1: extra exports c32_wave1_prefix_b8d / c32_wave1_mapped_b8 (block0 pooled output as E4M3 bytes, read by block1) and c32_wave1_finish_b8 / c32_wave1_post_b8 (block69 main as E4M3 bytes, read by the post): both tensors hold F(v), E4M3-exact (results/prefix-post-20260930) */
#endif
#if CW_PREPOST_BYTE && (!CW_VEC_INPUT || CW_TAIL_QUAD)
#error "CW_PREPOST_BYTE needs CW_VEC_INPUT and no CW_TAIL_QUAD"
#endif
#ifndef CW_PREFIX_TAIL_VEC
#define CW_PREFIX_TAIL_VEC 0 /* 1: prefix byte tail (main + down bytes) stored per lane as 8-channel groups: ds_load_b128 + cw_pack8 + 8-byte store instead of 64+16 per-channel byte loops; bytes unchanged (results/prefix-post-arith-20260930) */
#endif
#ifndef CW_FINISH_TAIL_VEC
#define CW_FINISH_TAIL_VEC 0 /* 1: finish_b8 / finish_dcrop_b8d byte tails (main bytes, DownCrop down bytes) per lane as 8-channel groups, same as CW_PREFIX_TAIL_VEC; bytes unchanged (results/tail-vec-20260930) */
#endif
#ifndef CW_UP_FUSED
#define CW_UP_FUSED 0
#endif
#ifndef CW_HOIST_UP
#define CW_HOIST_UP 0 /* hoist-loads 2026-09-30, bitmask, c32_wave1_up_b8: 1 low-res input per k step read branch-free (clamped index, select 0);
 2 per qt/ci the skip bytes (clamped), upw[2048+c] and fw[8704+c] for the lane's 8 channels are read up front instead of one b32 -> s_wait_loadcnt 0 per element */
#endif
#ifndef CW_DIAG_ONLY
#define CW_DIAG_ONLY 0 /* 1: chain residual skips the all-zero off-diagonal K16 half (6 of 12 WMMA); results/c32-diag-zero-20260925 proved the float32 bits equal for every finite E4M3 input */
#endif
#ifndef CW_SKIP_BYTE
#define CW_SKIP_BYTE 0 /* 1: extra exports c32_wave1_finish_dcrop_b8 / c32_wave1_up_b8: block4 main (the C32 skip) stored as E4M3 bytes fp8(F(v)) instead of f32 F(v); F(v) is E4M3-exact, so the up reads back the same value (results/c32-align-20260930) */
#endif
#if CW_SKIP_BYTE && CW_TAIL_QUAD
#error "CW_SKIP_BYTE: byte main store only on the non-quad finish tail"
#endif
DEV float cw_up_F(float x){uint a=bits(x)&0x7fffffffu;float c=__builtin_fminf(__builtin_fmaxf(x,-448.f),448.f);float r=__builtin_amdgcn_cvt_f32_fp8(__builtin_amdgcn_cvt_pk_fp8_f32(c,0.f,0,false),0);return a==0?0.f:r;}
DEV float cw_up_shuffle(float v,uint lane){return __builtin_bit_cast(float,__builtin_amdgcn_ds_bpermute(lane*4,__builtin_bit_cast(int,v)));}
template<bool Chain,bool Finish=false,bool DownCrop=false,bool Post=false,bool Prefix=false,bool Up=false,bool Byte=false,bool DownByte=false,bool InByte=false,bool UpLowByte=false,bool PostRGBA=false,bool PostTap=false>
DEV void cw_body(const unsigned char*in,const float*fw,const float*w,unsigned char*out,uint windows,uint mode,uint raw,uint width,uint height,uint sx,uint sy,uint prevw,uint px,uint py,float*main=nullptr,float*down=nullptr,const unsigned char*skip=nullptr,const float*scales=nullptr,const float*color=nullptr,const float*headw=nullptr,float*rgb=nullptr,float rgbscale=0.f,const float*history=nullptr,uint seed=0,uint temporal=0,const float*upw=nullptr,_Float16*post_features=nullptr,const float*logitw=nullptr,float*logit=nullptr){fp8_sat_mode();
 constexpr uint cw_kind=Prefix?5:Post?4:Finish?(DownCrop?3:2):Chain?0:1;
 uint win=bid();if(win>=windows)return;
 __attribute__((shared)) _Float16 residual[4*2*32*8];
 cw_i16 keys{},values{},queries{};
 f8 up_totals[2]{};
 if constexpr(Up){
  uint lane=__builtin_amdgcn_workitem_id_x(),r=lane&15u,g=lane>>4;
  uint ww=width+2*sx,iw=width/2,ih=height/2;
  int lx=(int((win%(ww/8))*8)-int(sx))/2+int(r%4),ly=(int((win/(ww/8))*8)-int(sy))/2+int(r/4);
  for(uint k=0;k<64;k+=16){h8 a{},b{};
#if CW_HOIST_UP&1
   {const bool ok=lx>=0&&ly>=0&&lx<int(iw)&&ly<int(ih);const uint base=ok?(uint(ly)*iw+uint(lx))*64:0u;for(uint e=0;e<8;e++){float v;if constexpr(UpLowByte)v=__builtin_amdgcn_cvt_f32_fp8(int(in[base+k+g*8+e]),0);else v=reinterpret_cast<const float*>(in)[base+k+g*8+e];a[e]=ok?(_Float16)v:(_Float16)0.f;}}
#else
   if(lx>=0&&ly>=0&&lx<int(iw)&&ly<int(ih))for(uint e=0;e<8;e++){const uint i=(uint(ly)*iw+uint(lx))*64+k+g*8+e;if constexpr(UpLowByte)a[e]=(_Float16)__builtin_amdgcn_cvt_f32_fp8(int(in[i]),0);else a[e]=(_Float16)reinterpret_cast<const float*>(in)[i];}
#endif
   for(uint ci=0;ci<2;ci++){
    __builtin_memcpy(&b,reinterpret_cast<const _Float16*>(upw)+(ci*16+r)*64+k+g*8,16);
    up_totals[ci]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b,a,up_totals[ci]);
   }
  }
 }
#if CW_ROLL_WINDOW
 _Pragma("clang loop unroll(disable)")
#else
 _Pragma("unroll 4")
#endif
 for(uint qt=0;qt<4;qt++){
  uint lane=__builtin_amdgcn_workitem_id_x();
#if CW_LAUNDER
  __asm__ volatile("" : "+v"(lane));lane&=31u;
#endif
  uint r=lane&15u,g=lane>>4;
  i2 input[2];f8 ffn[2]{};
  if constexpr(Chain){for(uint ci=0;ci<2;ci++)input[ci]=cw_input(in,win,qt,ci,width,height,sx,sy,prevw,px,py,r,g);}
  else if constexpr(Prefix){
#if CW_PREFIX_SPLIT
   // B1 prefix split: fused_prefix_values ran on the g==0 half only (16 lanes idle) and then shipped values 8..15 with eight
   // bpermutes. Here both halves run the same instruction stream on the same token r: the half-wave selects which Box-Muller
   // pair it evaluates (g==1: a,c -> g0; g==0: b,d -> g1,g2), and g0 is the only value exchanged (one bpermute). Each value's
   // operation sequence is identical to fused_prefix_values, so the features, and everything downstream, are bit-identical.
   h8 a{};
   {const float*rgba=reinterpret_cast<const float*>(in);uint p=win*64+qt*16+r,tile=p/64,x=(tile%(width/8))*8+p%8,y=(tile/(width/8))*8+(p%64)/8,rgbp=y*width+x;
    uint h=production_pcg((x*0x8da6b343u)^(y*0xd8163841u)^(seed*0x9e3779b9u)^0x243f6a88u);
    float u1=production_uniform(g?h*0xcaa5b80du+0x21dd796bu:h*0x2c9277b5u+0xac564b05u),u2=production_uniform(g?h*0x83232c31u+0x3463e0acu:h*0xfa6dc5f9u+0x4712a88eu);
    float rr=__builtin_amdgcn_sqrtf(-2.f*(__builtin_amdgcn_logf(u1)*.6931471805599453f)),tt=(6.283185482025146f*u2)*.15915494309189535f;
    cw_pv_t gc=cw_prefix_rtz(rr*__builtin_amdgcn_cosf(tt)),gs=cw_prefix_rtz(rr*__builtin_amdgcn_sinf(tt));
    cw_pv_t g0=cw_prefix_shuffle(gc,r+16);
    cw_pv_t red=cw_prefix_rtz(Hrtz(Hrtz(rgba[rgbp*4])-.5f)*.125f),green=cw_prefix_rtz(Hrtz(Hrtz(rgba[rgbp*4+1])-.5f)*.125f),blue=cw_prefix_rtz(Hrtz(Hrtz(rgba[rgbp*4+2])-.5f)*.125f);
    cw_pv_t hr=red,hg=green,hb=blue;
    if(temporal){uint q=(y*width+x)*4;hr=cw_prefix_rtz(Hrtz(Hrtz(history[q])-.5f)*.125f);hg=cw_prefix_rtz(Hrtz(Hrtz(history[q+1])-.5f)*.125f);hb=cw_prefix_rtz(Hrtz(Hrtz(history[q+2])-.5f)*.125f);}
    const cw_pv_t lo[8]={gc,gs,hg,hb,g0,1,(cw_pv_t)dlss5_style_feature,1},hi[8]={red,green,1,1,blue,hr,1,0};
    for(uint e=0;e<8;e++)a[e]=(_Float16)(g?hi[e]:lo[e]);}
#else
   float pv[16]{};if(g==0)fused_prefix_values(reinterpret_cast<const float*>(in),history,win*64+qt*16+r,width,seed,temporal,pv);
   h8 a{};for(uint e=0;e<8;e++){float hi=__builtin_bit_cast(float,__builtin_amdgcn_ds_bpermute(r*4,__builtin_bit_cast(int,pv[8+e])));a[e]=(_Float16)(g?hi:pv[e]);}
#endif
   for(uint ci=0;ci<2;ci++){
    h8 b{};for(uint e=0;e<8;e++)b[e]=(_Float16)fw[(ci*16+r)*16+g*8+e];
    f8 pre=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b,a,f8{});h8 zero{};
    pre=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(zero,zero,pre);
    CW_Q8_DECL(packed);for(uint e=0;e<8;e++){float v=Hrtz(pre[e]);CW_Q8_SET(packed,e,v);ffn[ci][e]=cw_clampf(v,-448.f,448.f)*fw[8704+ci*16+g*8+e];}CW_Q8_END(packed);input[ci]=packed;
   }
  }
  else if constexpr(Up){
   uint ww=width+2*sx,tok=qt*16+r,lo=(tok/16)*4+(tok%8)/2;
   int x=int((win%(ww/8))*8+tok%8)-int(sx),y=int((win/(ww/8))*8+tok/8)-int(sy);
   bool valid=x>=0&&y>=0&&x<int(width)&&y<int(height);
#if CW_HOIST_UP&2
   if constexpr(Byte){
   for(uint ci=0;ci<2;ci++){
    const uint c0=ci*16+g*8;unsigned long long sk;float uw[8],fv[8];
    __builtin_memcpy(&sk,__builtin_assume_aligned(skip+(valid?(uint(y)*width+uint(x))*32:0u)+c0,8),8);sk=valid?sk:0ull;
    __builtin_memcpy(uw,upw+2048+c0,32);__builtin_memcpy(fv,fw+8704+c0,32);
    CW_Q8_DECL(a);for(uint e=0;e<8;e++){
     float t=H(cw_up_shuffle(up_totals[ci][e],lo+g*16));
     float m=H(t+__builtin_amdgcn_cvt_f32_fp8(int((sk>>(8*e))&255u),0)*uw[e]);float v=valid?m:0.f;
     CW_Q8_SET(a,e,v);ffn[ci][e]=cw_clampf((CW_INPUT_HALF&4u)?v:float((_Float16)v),-448.f,448.f)*fv[e];
    }CW_Q8_END(a);input[ci]=a;
   }
   }else
#endif
   for(uint ci=0;ci<2;ci++){
    unsigned long long sk=0;if constexpr(Byte){if(valid)__builtin_memcpy(&sk,__builtin_assume_aligned(skip+(uint(y)*width+uint(x))*32+ci*16+g*8,8),8);}
    CW_Q8_DECL(a);for(uint e=0;e<8;e++){
     uint c=ci*16+g*8+e;float v=0.f,t=H(cw_up_shuffle(up_totals[ci][e],lo+g*16));
     if constexpr(Byte){if(valid)v=H(t+__builtin_amdgcn_cvt_f32_fp8(int((sk>>(8*e))&255u),0)*upw[2048+c]);}
     else if(valid)v=H(t+cw_up_F(reinterpret_cast<const float*>(skip)[(uint(y)*width+uint(x))*32+c])*upw[2048+c]);
     // Keep raw half-valued Up output for FFN residual; quantize only its matrix operand.
     CW_Q8_SET(a,e,v);ffn[ci][e]=cw_clampf((CW_INPUT_HALF&4u)?v:float((_Float16)v),-448.f,448.f)*fw[8704+c];
    }CW_Q8_END(a);input[ci]=a;
   }
  }
  else{
   uint ww=width+2*sx;int x=int((win%(ww/8))*8+(qt*16+r)%8)-int(sx),y=int((win/(ww/8))*8+(qt*16+r)/8)-int(sy);
   bool valid=x>=0&&y>=0&&x<int(width)&&y<int(height);
#if CW_VEC_INPUT
   // The lane's 8 channels c=ci*16+g*8..+7 are contiguous in every source (f32 features, FP8 skip bytes, scales, FFN residual
   // scale), 32-byte aligned (hipMalloc bases, channel offsets multiple of 8). Read them as whole rows (2x b128 / 1x b64)
   // instead of 8 per-element loads each; the per-element arithmetic and its order are unchanged.
   for(uint ci=0;ci<2;ci++){uint c0=ci*16+g*8;float src[8]{};
    if(valid){
     if constexpr(Post){uint p=uint(y)*width+uint(x),low=(uint(y)/2)*(width/2)+uint(x)/2;
      float lo[8],s0[8],s1[8];unsigned long long sk;
      if constexpr(InByte){unsigned long long lb;__builtin_memcpy(&lb,__builtin_assume_aligned(in+low*32+c0,8),8);for(uint e=0;e<8;e++)lo[e]=__builtin_amdgcn_cvt_f32_fp8(int((lb>>(8*e))&255u),0);}
      else __builtin_memcpy(lo,__builtin_assume_aligned(reinterpret_cast<const float*>(in)+low*32+c0,32),32);
      __builtin_memcpy(s0,__builtin_assume_aligned(scales+c0,32),32);__builtin_memcpy(s1,__builtin_assume_aligned(scales+32+c0,32),32);
      __builtin_memcpy(&sk,__builtin_assume_aligned(skip+p*32+c0,8),8);
      for(uint e=0;e<8;e++){float high=__builtin_amdgcn_cvt_f32_fp8(int((sk>>(8*e))&255u),0);src[e]=Hrtz(Hrtz(lo[e]*s0[e])+high*s1[e]);}
     }else if constexpr(InByte){unsigned long long ib;__builtin_memcpy(&ib,__builtin_assume_aligned(in+(uint(y)*width+uint(x))*32+c0,8),8);for(uint e=0;e<8;e++)src[e]=__builtin_amdgcn_cvt_f32_fp8(int((ib>>(8*e))&255u),0);}
     else __builtin_memcpy(src,__builtin_assume_aligned(reinterpret_cast<const float*>(in)+(uint(y)*width+uint(x))*32+c0,32),32);
    }
    float res[8];__builtin_memcpy(res,__builtin_assume_aligned(fw+8704+c0,32),32);
    CW_Q8_DECL(a);for(uint e=0;e<8;e++){float v=src[e];CW_Q8_SET(a,e,v);ffn[ci][e]=cw_clampf((CW_INPUT_HALF&(Post?1u:2u))?v:float((_Float16)v),-448.f,448.f)*res[e];}CW_Q8_END(a);input[ci]=a;}
#else
   for(uint ci=0;ci<2;ci++){CW_Q8_DECL(a);for(uint e=0;e<8;e++){uint c=ci*16+g*8+e;float v=0.f;if(valid){if constexpr(Post){uint p=uint(y)*width+uint(x),low=(uint(y)/2)*(width/2)+uint(x)/2;float high=__builtin_amdgcn_cvt_f32_fp8(int(skip[p*32+c]),0);v=Hrtz(Hrtz(reinterpret_cast<const float*>(in)[low*32+c]*scales[c])+high*scales[32+c]);}else v=reinterpret_cast<const float*>(in)[(uint(y)*width+uint(x))*32+c];}CW_Q8_SET(a,e,v);ffn[ci][e]=cw_clampf(float((_Float16)v),-448.f,448.f)*fw[8704+c];}CW_Q8_END(a);input[ci]=a;}
#endif
  }
  if constexpr(Chain)for(uint part=0;part<3;part++)for(uint ci=0;ci<2;ci++)for(uint kt=0;kt<2;kt++){
#if CW_DIAG_ONLY
   if(kt!=ci)continue; /* the three residual matrices are diagonal: the off-diagonal K16 half-block is all zero */
#endif
   i2 b=matrix8(fw,34944+((part*2+ci)*2+kt)*512+(g*16+r)*8);
   ffn[ci]=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(b,input[kt],ffn[ci]);
  }
#if CW_ROLL_HIDDEN
  _Pragma("clang loop unroll(disable)")
#endif
  for(uint ht=0;ht<8;ht++){
   f8 expanded{};for(uint kt=0;kt<2;kt++){i2 b=matrix8(fw+512,(ht*16+r)*32+kt*16+g*8);expanded=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(b,input[kt],expanded);}
#if CW_FAST_NUM&4
   /* FAST TIER (lossy): activation in packed half pairs (v_pk_max/min/fma/mul_f16), widened back to f32 for the fp8 pack */
   CW_Q8_DECL(a);for(uint e=0;e<8;e+=2){using hh=_Float16 __attribute__((ext_vector_type(2)));hh v={(_Float16)expanded[e],(_Float16)expanded[e+1]};
    hh g=__builtin_elementwise_min(__builtin_elementwise_max(v,hh{-4,-4}),hh{4,4});hh ag=__builtin_elementwise_abs(g);
    hh q=ag*hh{(_Float16)-.055908203125f,(_Float16)-.055908203125f}+hh{(_Float16).447265625f,(_Float16).447265625f};hh p=g*q+hh{(_Float16).89453125f,(_Float16).89453125f};hh o=v*p;
    CW_Q8_SET(a,e,float(o[0]));CW_Q8_SET(a,e+1,float(o[1]));}CW_Q8_END_SITE(a,1);
#else
   CW_Q8_DECL(a);for(uint e=0;e<8;e++){float v=expanded[e],g=cw_act_gate(v),q=__builtin_fmaf(absf(g),-.055908203125f,.447265625f),p=__builtin_fmaf(g,q,.89453125f);CW_Q8_SET(a,e,v*p);}CW_Q8_END_SITE(a,1);
#endif
   for(uint ci=0;ci<2;ci++){i2 b=matrix8(fw+4608,(ci*16+r)*128+ht*16+g*8);ffn[ci]=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(b,a,ffn[ci]);}
  }
  i2 feature[2];for(uint ci=0;ci<2;ci++){CW_Q8_DECL(a);
#if CW_RTZ_PAIR
   h8 rounded=cw_rtz_half8(ffn[ci]);__builtin_memcpy(residual+((qt*2+ci)*32+(g*16+r))*8,&rounded,16);
   for(uint e=0;e<8;e++)CW_Q8_SET(a,e,float(rounded[e]));
#else
   for(uint e=0;e<8;e++){float v=Hrtz(ffn[ci][e]);residual[((qt*2+ci)*32+(g*16+r))*8+e]=(_Float16)v;CW_Q8_SET(a,e,v);}
#endif
   CW_Q8_END_SITE(a,2);feature[ci]=a;}
  for(uint part=0;part<3;part++){
   uint qlane=__builtin_amdgcn_workitem_id_x();
#if CW_QKV_FENCE
   __asm__ volatile("" : "+v"(qlane));qlane&=31u;
#endif
   uint qr=qlane&15u,qg=qlane>>4;
   f8 q[2]{};for(uint ci=0;ci<2;ci++)for(uint kt=0;kt<2;kt++){i2 b=matrix8(w,part*1024+(ci*16+qr)*32+kt*16+qg*8);q[ci]=part<2?__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(b,feature[kt],q[ci]):__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(feature[kt],b,q[ci]);}
   f8 sum{};if(part<2)for(uint ci=0;ci<2;ci++){h8 squares{},ones{};for(uint e=0;e<8;e++){squares[e]=(_Float16)(q[ci][e]*q[ci][e]);ones[e]=(_Float16)1.f;}sum=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ones,squares,sum);}
#if CW_NORM_HOIST
   float lane_rsq=part<2?__builtin_amdgcn_rsqf(maxf(sum[0],6.198883056640625e-5f)):0.f;
   for(uint ci=0;ci<2;ci++){CW_Q8_DECL(a);for(uint e=0;e<8;e++){float v=q[ci][e];if(part<2)v*=lane_rsq*(part==0?w[8192]:1.f);
#else
   for(uint ci=0;ci<2;ci++){CW_Q8_DECL(a);for(uint e=0;e<8;e++){float v=q[ci][e];if(part<2)v*=__builtin_amdgcn_rsqf(maxf(sum[e],6.198883056640625e-5f))*(part==0?w[8192]:1.f);
#endif
CW_Q8_SET(a,e,v);}CW_Q8_END_SITE(a,3);
    if(part==0){queries[qt*4+ci*2]=a[0];queries[qt*4+ci*2+1]=a[1];}else if(part==1){keys[qt*4+ci*2]=a[0];keys[qt*4+ci*2+1]=a[1];}else{values[qt*4+ci*2]=a[0];values[qt*4+ci*2+1]=a[1];}
   }
#if CW_QKV_FENCE
   __builtin_amdgcn_sched_barrier(0);
#endif
  }
  __builtin_amdgcn_sched_barrier(0);
 }
 _Pragma("clang loop unroll(disable)") for(uint qt=0;qt<4;qt++){
  uint lane=__builtin_amdgcn_workitem_id_x();
#if CW_LAUNDER
  __asm__ volatile("" : "+v"(lane));lane&=31u;
#endif
  uint r=lane&15u,g=lane>>4;
  i2 q[2];for(uint ci=0;ci<2;ci++)q[ci]=i2{queries[qt*4+ci*2],queries[qt*4+ci*2+1]};
  h8 ex[4];
  _Pragma("unroll 4") for(uint kt=0;kt<4;kt++){
   f8 acc{};for(uint ci=0;ci<2;ci++)acc=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12((i2{keys[kt*4+ci*2],keys[kt*4+ci*2+1]}),q[ci],acc);
   h8 a{};for(uint e=0;e<8;e++){uint row=qt*16+r,k=kt*16+g*8+e;float score=acc[e]+w[4096+row*64+k],affine=cw_clampf(score*.044921875f+1.30078125f,1.03125f,1.5693359375f);uint ah=(bits(affine)>>13)-0x1c000u;ushort hb=static_cast<ushort>(((ah<<5)+0x8000u)&65535u);a[e]=__builtin_bit_cast(_Float16,hb);}ex[kt]=a;
  }
  f8 sums{};h8 ones{};for(uint e=0;e<8;e++)ones[e]=(_Float16)1.f;
  _Pragma("unroll 4") for(uint kt=0;kt<4;kt++)sums=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ones,ex[kt],sums);
  i2 prob[4];_Pragma("unroll 4") for(uint kt=0;kt<4;kt++){CW_Q8_DECL(a);for(uint e=0;e<8;e++)CW_Q8_SET(a,e,float(ex[kt][e])*norm_inverse(sums[e]));CW_Q8_END_SITE(a,4);prob[kt]=a;}
  i2 av[2];for(uint ci=0;ci<2;ci++){f8 acc{};_Pragma("unroll 4") for(uint kt=0;kt<4;kt++)acc=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12((i2{values[kt*4+ci*2],values[kt*4+ci*2+1]}),prob[kt],acc);CW_Q8_DECL(a);for(uint e=0;e<8;e++)CW_Q8_SET(a,e,acc[e]);CW_Q8_END_SITE(a,5);av[ci]=a;}
  for(uint ci=0;ci<2;ci++){
   f8 acc{};for(uint kt=0;kt<2;kt++){i2 b=matrix8(w,3072+(ci*16+r)*32+kt*16+g*8);acc=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(b,av[kt],acc);}
   CW_Q8_DECL(a);
#if CW_RTZ_PAIR
   f8 summed;for(uint e=0;e<8;e++){float res=float(residual[((qt*2+ci)*32+(g*16+r))*8+e]),prod=res*w[8193+ci*16+g*8+e];summed[e]=acc[e]+prod;}
   h8 rounded=cw_rtz_half8(summed);
   if constexpr(Finish||Post)__builtin_memcpy(residual+((qt*2+ci)*32+(g*16+r))*8,&rounded,16);
   else for(uint e=0;e<8;e++)CW_Q8_SET(a,e,cw_output_value(float(rounded[e])));
#else
   for(uint e=0;e<8;e++){float res=float(residual[((qt*2+ci)*32+(g*16+r))*8+e]),prod=res*w[8193+ci*16+g*8+e],v=Hrtz(acc[e]+prod);if constexpr(Finish||Post)residual[((qt*2+ci)*32+(g*16+r))*8+e]=(_Float16)v;else CW_Q8_SET(a,e,cw_output_value(v));}
#endif
   if constexpr(!Finish&&!Post)CW_Q8_END_SITE(a,6);
   if constexpr(!Finish&&!Post)__builtin_memcpy(out+(win*64+qt*16+r)*32+ci*16+g*8,&a,8);
  }
  __builtin_amdgcn_sched_barrier(0);
 }

#if CW_POST_HEAD_VEC || CW_POST_FULL_TILE
 if constexpr(Post){
  WG_FENCE(3);WG_FENCE(2);
  uint lane=__builtin_amdgcn_workitem_id_x(),ww=width+2*sx,tx=(win%(ww/8))*8,ty=(win/(ww/8))*8;
  auto emit=[&](auto full){
  for(uint t=lane;t<64;t+=32){int x=int(tx+t%8)-int(sx),y=int(ty+t/8)-int(sy);if(full||(x>=0&&y>=0&&x<int(width)&&y<int(height))){
   uint p=uint(y)*width+uint(x);float sums[3]{};
   if constexpr(PostTap)for(uint c=0;c<32;c++)post_features[p*32+c]=residual[((t/16*2+c/16)*32+(c%16/8*16+t%16))*8+c%8];
#if CW_POST_HEAD_VEC
   for(uint cg=0;cg<4;cg++){h8 fs;float hr[8],hg[8],hb[8];uint c=cg*8;
    __builtin_memcpy(&fs,residual+((t/16*2+c/16)*32+(c%16/8*16+t%16))*8,16);
    __builtin_memcpy(hr,headw+c,32);__builtin_memcpy(hg,headw+32+c,32);__builtin_memcpy(hb,headw+64+c,32);
    for(uint e=0;e<8;e++){float f=float(fs[e]);sums[0]+=f*hr[e];sums[1]+=f*hg[e];sums[2]+=f*hb[e];}}
#else
   for(uint c=0;c<32;c++){float f=float(residual[((t/16*2+c/16)*32+(c%16/8*16+t%16))*8+c%8]);sums[0]+=f*headw[c];sums[1]+=f*headw[32+c];sums[2]+=f*headw[64+c];}
#endif
   for(uint c=0;c<3;c++){float acc=Hrtz(sums[c]),centered=color[p*4+c]*.125f-.0625f,v=(acc*rgbscale+centered)*8.f+.5f;rgb[p*(PostRGBA?4:3)+c]=cw_clampf(v,0.f,1.f);}if constexpr(PostRGBA)rgb[p*4+3]=1.f;
   if(logit){float value=0.f;
    for(uint c=0;c<32;c++){float f=float(residual[((t/16*2+c/16)*32+(c%16/8*16+t%16))*8+c%8]);value+=f*logitw[c];}
    logit[p*2]=value;logit[p*2+1]=Hrtz(value);
   }
  }}};
#if CW_POST_FULL_TILE
  if(tx>=sx&&ty>=sy&&tx+8<=width+sx&&ty+8<=height+sy)emit(cw_true{});else emit(cw_false{});
#else
  emit(cw_false{});
#endif
 }
#else
 if constexpr(Post){
  WG_FENCE(3);WG_FENCE(2);
  uint lane=__builtin_amdgcn_workitem_id_x(),ww=width+2*sx,tx=(win%(ww/8))*8,ty=(win/(ww/8))*8;
  for(uint t=lane;t<64;t+=32){int x=int(tx+t%8)-int(sx),y=int(ty+t/8)-int(sy);if(x>=0&&y>=0&&x<int(width)&&y<int(height)){
   uint p=uint(y)*width+uint(x);float sums[3]{};
   if constexpr(PostTap)for(uint c=0;c<32;c++)post_features[p*32+c]=residual[((t/16*2+c/16)*32+(c%16/8*16+t%16))*8+c%8];
   for(uint c=0;c<32;c++){float f=float(residual[((t/16*2+c/16)*32+(c%16/8*16+t%16))*8+c%8]);sums[0]+=f*headw[c];sums[1]+=f*headw[32+c];sums[2]+=f*headw[64+c];}
   for(uint c=0;c<3;c++){float acc=Hrtz(sums[c]),centered=color[p*4+c]*.125f-.0625f,v=(acc*rgbscale+centered)*8.f+.5f;rgb[p*(PostRGBA?4:3)+c]=cw_clampf(v,0.f,1.f);}if constexpr(PostRGBA)rgb[p*4+3]=1.f;
   if(logit){float value=0.f;
    for(uint c=0;c<32;c++){float f=float(residual[((t/16*2+c/16)*32+(c%16/8*16+t%16))*8+c%8]);value+=f*logitw[c];}
    logit[p*2]=value;logit[p*2+1]=Hrtz(value);
   }
  }}
 }
#endif
 if constexpr(CW_FINISH_FULL_TILE&&!Prefix){
 if constexpr(Finish){
  WG_FENCE(3);WG_FENCE(2);
  uint c=__builtin_amdgcn_workitem_id_x(),ww=width+2*sx,tx=(win%(ww/8))*8,ty=(win/(ww/8))*8;
  auto load=[&](uint t)->float{return float(residual[((t/16*2+c/16)*32+(c%16/8*16+t%16))*8+c%8]);};
#if CW_TAIL_QUAD
  auto put_main=[&](uint t,float raw_v){int x=int(tx+t%8)-int(sx),y=int(ty+t/8)-int(sy);if(x>=0&&y>=0&&x<int(width)&&y<int(height)){uint dest=(uint(y)*width+uint(x))*32+c;float v=F(raw_v);
   if constexpr(Prefix)*reinterpret_cast<unsigned char*>(reinterpret_cast<char*>(main)+dest)=static_cast<unsigned char>(fp8(v));else *reinterpret_cast<float*>(reinterpret_cast<char*>(main)+dest*4u)=v;}};
  for(uint k=0;k<16;k++){uint t=(k/4)*16+(k%4)*2;float a0=load(t),a1=load(t+1),b0=load(t+8),b1=load(t+9);
   if(main){put_main(t,a0);put_main(t+1,a1);put_main(t+8,b0);put_main(t+9,b1);}
   if(down){float top=Hrtz(a0+a1),bottom=Hrtz(b0+b1);uint x=tx+t%8,y=ty+t/8;
    if constexpr(DownCrop){int cx=int(x/2)-int(sx/2),cy=int(y/2)-int(sy/2);if(cx>=0&&cy>=0&&cx<int(width/2)&&cy<int(height/2))*reinterpret_cast<float*>(reinterpret_cast<char*>(down)+((uint(cy)*(width/2)+uint(cx))*32+c)*4u)=F(Hrtz(Hrtz(top+bottom)*.25f));}
    else *reinterpret_cast<float*>(reinterpret_cast<char*>(down)+(((y/2)*(ww/2)+x/2)*32+c)*4u)=F(Hrtz(Hrtz(top+bottom)*.25f));}
  }
  // The quad route above emits both outputs.
#endif
#if CW_FINISH_TAIL_VEC && !CW_TAIL_QUAD
  // Byte main (and byte DownCrop down) tail, vectorised like CW_PREFIX_TAIL_VEC: lane = (pixel slot c>>2, 8-channel group c&3),
  // one ds_load_b128 of 8 contiguous residual halves, one MODE-saturated cw_pack8, one 8-byte store; bounds tested per pixel
  // exactly as the per-channel loop. Bytes unchanged: fp8(F(x)) == fp8(x+0) for the finite half values x held here.
  if constexpr(Byte&&(DownByte||!DownCrop)){
   uint cg=c&3u,sl=c>>2;
   auto row=[&](uint t)->h8{h8 v;__builtin_memcpy(&v,__builtin_assume_aligned(residual+((t/16*2+cg/2)*32+((cg&1u)*16+t%16))*8,16),16);return v;};
   auto emitv=[&](auto full){
    if(main)for(uint i=0;i<8;i++){uint t=i*8+sl;int x=int(tx+t%8)-int(sx),y=int(ty+t/8)-int(sy);
     if(full||(x>=0&&y>=0&&x<int(width)&&y<int(height))){h8 hv=row(t);float q[8];for(uint e=0;e<8;e++)q[e]=float(hv[e])+0.f;
      i2 b=cw_pack8<0,cw_kind>(q);__builtin_memcpy(__builtin_assume_aligned(reinterpret_cast<unsigned char*>(main)+((uint(y)*width+uint(x))*32+cg*8),8),&b,8);}}
    if constexpr(DownByte&&DownCrop)if(down)for(uint j=0;j<2;j++){uint k=j*8+sl,t=(k/4)*16+(k%4)*2;uint x=tx+t%8,y=ty+t/8;int cx=int(x/2)-int(sx/2),cy=int(y/2)-int(sy/2);
     if(full||(cx>=0&&cy>=0&&cx<int(width/2)&&cy<int(height/2))){h8 a0=row(t),a1=row(t+1),b0=row(t+8),b1=row(t+9);float q[8];
      for(uint e=0;e<8;e++){float top=Hrtz(float(a0[e])+float(a1[e])),bottom=Hrtz(float(b0[e])+float(b1[e]));q[e]=Hrtz(Hrtz(top+bottom)*.25f)+0.f;}
      i2 b=cw_pack8<0,cw_kind>(q);__builtin_memcpy(__builtin_assume_aligned(reinterpret_cast<unsigned char*>(down)+((uint(cy)*(width/2)+uint(cx))*32+cg*8),8),&b,8);}}
   };
   if(tx>=sx&&ty>=sy&&tx+8<=width+sx&&ty+8<=height+sy&&(!DownCrop||((sx|sy)&1u)==0))emitv(cw_true{});else emitv(cw_false{});
   if constexpr(!DownCrop)if(down)for(uint k=0;k<16;k++){uint t=(k/4)*16+(k%4)*2;float top=Hrtz(load(t)+load(t+1)),bottom=Hrtz(load(t+8)+load(t+9));uint x=tx+t%8,y=ty+t/8;
    if constexpr(DownByte)reinterpret_cast<unsigned char*>(down)[((y/2)*(ww/2)+x/2)*32+c]=static_cast<unsigned char>(fp8(F(Hrtz(Hrtz(top+bottom)*.25f))));
    else down[((y/2)*(ww/2)+x/2)*32+c]=F(Hrtz(Hrtz(top+bottom)*.25f));}
   return;
  }
#endif
  auto emit=[&](auto full){if(main)for(uint t=0;t<64;t++){int x=int(tx+t%8)-int(sx),y=int(ty+t/8)-int(sy);if(full||(Prefix&&CW_PREFIX_FULL_TILE)||(x>=0&&y>=0&&x<int(width)&&y<int(height))){uint dest=(uint(y)*width+uint(x))*32+c;float v=load(t);if constexpr(Prefix)reinterpret_cast<unsigned char*>(main)[dest]=static_cast<unsigned char>(cw_prefix_byte(v));else if constexpr(Byte)reinterpret_cast<unsigned char*>(main)[dest]=static_cast<unsigned char>(fp8(F(v)));else main[dest]=F(v);}}
  if(!CW_TAIL_QUAD&&down)for(uint k=0;k<16;k++){uint t=(k/4)*16+(k%4)*2;float top=Hrtz(load(t)+load(t+1)),bottom=Hrtz(load(t+8)+load(t+9));uint x=tx+t%8,y=ty+t/8;
   if constexpr(DownCrop){int cx=int(x/2)-int(sx/2),cy=int(y/2)-int(sy/2);if(full||(cx>=0&&cy>=0&&cx<int(width/2)&&cy<int(height/2))){float dv=F(Hrtz(Hrtz(top+bottom)*.25f));uint di=(uint(cy)*(width/2)+uint(cx))*32+c;if constexpr(DownByte)reinterpret_cast<unsigned char*>(down)[di]=static_cast<unsigned char>(fp8(dv));else down[di]=dv;}}
   else if constexpr(DownByte)reinterpret_cast<unsigned char*>(down)[((y/2)*(ww/2)+x/2)*32+c]=static_cast<unsigned char>(fp8(F(Hrtz(Hrtz(top+bottom)*.25f))));
   else down[((y/2)*(ww/2)+x/2)*32+c]=F(Hrtz(Hrtz(top+bottom)*.25f));
  }
  };
#if !CW_TAIL_QUAD
  // DownCrop uses sx/2,sy/2; the production shifts are even. Keep the generic edge path for odd shifts.
  if(tx>=sx&&ty>=sy&&tx+8<=width+sx&&ty+8<=height+sy&&(!DownCrop||((sx|sy)&1u)==0))emit(cw_true{});else emit(cw_false{});
#endif
 }
 }else{
 if constexpr(Finish){
  WG_FENCE(3);WG_FENCE(2);
  uint c=__builtin_amdgcn_workitem_id_x(),ww=width+2*sx,tx=(win%(ww/8))*8,ty=(win/(ww/8))*8;
  auto load=[&](uint t)->float{return float(residual[((t/16*2+c/16)*32+(c%16/8*16+t%16))*8+c%8]);};
#if CW_TAIL_QUAD
  auto put_main=[&](uint t,float raw_v){int x=int(tx+t%8)-int(sx),y=int(ty+t/8)-int(sy);if(x>=0&&y>=0&&x<int(width)&&y<int(height)){uint dest=(uint(y)*width+uint(x))*32+c;float v=F(raw_v);
   if constexpr(Prefix)*reinterpret_cast<unsigned char*>(reinterpret_cast<char*>(main)+dest)=static_cast<unsigned char>(fp8(v));else *reinterpret_cast<float*>(reinterpret_cast<char*>(main)+dest*4u)=v;}};
  for(uint k=0;k<16;k++){uint t=(k/4)*16+(k%4)*2;float a0=load(t),a1=load(t+1),b0=load(t+8),b1=load(t+9);
   if(main){put_main(t,a0);put_main(t+1,a1);put_main(t+8,b0);put_main(t+9,b1);}
   if(down){float top=Hrtz(a0+a1),bottom=Hrtz(b0+b1);uint x=tx+t%8,y=ty+t/8;
    if constexpr(DownCrop){int cx=int(x/2)-int(sx/2),cy=int(y/2)-int(sy/2);if(cx>=0&&cy>=0&&cx<int(width/2)&&cy<int(height/2))*reinterpret_cast<float*>(reinterpret_cast<char*>(down)+((uint(cy)*(width/2)+uint(cx))*32+c)*4u)=F(Hrtz(Hrtz(top+bottom)*.25f));}
    else *reinterpret_cast<float*>(reinterpret_cast<char*>(down)+(((y/2)*(ww/2)+x/2)*32+c)*4u)=F(Hrtz(Hrtz(top+bottom)*.25f));}
  }
  if(false)
#endif
#if CW_PREFIX_TAIL_VEC
  // Prefix byte tail, vectorised: lane = (pixel slot c>>2, 8-channel group c&3); one ds_load_b128 of 8 contiguous residual
  // halves, one MODE-saturated cw_pack8 (same bytes as fp8() for the finite half values held here, the site-2 argument),
  // one 8-byte store; 8 lanes x 4 groups cover a 256-byte pixel row. Values per byte unchanged: main fp8(v+0),
  // down fp8(F(x)) == fp8(x+0) for finite x (F only differs at +-0, where both give 0x00).
  if constexpr(Prefix&&DownByte&&CW_PREFIX_FULL_TILE){
   uint cg=c&3u,sl=c>>2;
   auto row=[&](uint t)->h8{h8 v;__builtin_memcpy(&v,__builtin_assume_aligned(residual+((t/16*2+cg/2)*32+((cg&1u)*16+t%16))*8,16),16);return v;};
   if(main)for(uint i=0;i<8;i++){uint t=i*8+sl;h8 hv=row(t);float q[8];for(uint e=0;e<8;e++)q[e]=float(hv[e])+0.f;
    i2 b=cw_pack8<0,cw_kind>(q);__builtin_memcpy(__builtin_assume_aligned(reinterpret_cast<unsigned char*>(main)+((ty+t/8)*width+tx+t%8)*32+cg*8,8),&b,8);}
   if(down)for(uint j=0;j<2;j++){uint k=j*8+sl,t=(k/4)*16+(k%4)*2;h8 a0=row(t),a1=row(t+1),b0=row(t+8),b1=row(t+9);float q[8];
    for(uint e=0;e<8;e++){float top=Hrtz(float(a0[e])+float(a1[e])),bottom=Hrtz(float(b0[e])+float(b1[e]));q[e]=Hrtz(Hrtz(top+bottom)*.25f)+0.f;}
    uint x=tx+t%8,y=ty+t/8;i2 b=cw_pack8<0,cw_kind>(q);__builtin_memcpy(__builtin_assume_aligned(reinterpret_cast<unsigned char*>(down)+((y/2)*(ww/2)+x/2)*32+cg*8,8),&b,8);}
   return;
  }
#endif
  if(main)for(uint t=0;t<64;t++){int x=int(tx+t%8)-int(sx),y=int(ty+t/8)-int(sy);if((Prefix&&CW_PREFIX_FULL_TILE)||(x>=0&&y>=0&&x<int(width)&&y<int(height))){uint dest=(uint(y)*width+uint(x))*32+c;float v=load(t);if constexpr(Prefix)reinterpret_cast<unsigned char*>(main)[dest]=static_cast<unsigned char>(cw_prefix_byte(v));else if constexpr(Byte)reinterpret_cast<unsigned char*>(main)[dest]=static_cast<unsigned char>(fp8(F(v)));else main[dest]=F(v);}}
  if(!CW_TAIL_QUAD&&down)for(uint k=0;k<16;k++){uint t=(k/4)*16+(k%4)*2;float top=Hrtz(load(t)+load(t+1)),bottom=Hrtz(load(t+8)+load(t+9));uint x=tx+t%8,y=ty+t/8;
   if constexpr(DownCrop){int cx=int(x/2)-int(sx/2),cy=int(y/2)-int(sy/2);if(cx>=0&&cy>=0&&cx<int(width/2)&&cy<int(height/2)){float dv=F(Hrtz(Hrtz(top+bottom)*.25f));uint di=(uint(cy)*(width/2)+uint(cx))*32+c;if constexpr(DownByte)reinterpret_cast<unsigned char*>(down)[di]=static_cast<unsigned char>(fp8(dv));else down[di]=dv;}}
   else if constexpr(DownByte)reinterpret_cast<unsigned char*>(down)[((y/2)*(ww/2)+x/2)*32+c]=static_cast<unsigned char>(fp8(F(Hrtz(Hrtz(top+bottom)*.25f))));
   else down[((y/2)*(ww/2)+x/2)*32+c]=F(Hrtz(Hrtz(top+bottom)*.25f));
  }
 }
 }
}
#define CW_ENTRY KERNEL __attribute__((amdgpu_flat_work_group_size(32,32)))
CW_ENTRY void c32_wave1_chain(const unsigned char*in,const float*fw,const float*w,unsigned char*out,uint windows,uint mode,uint raw,uint width,uint height,uint sx,uint sy,uint prevw,uint px,uint py){cw_body<true>(in,fw,w,out,windows,mode,raw,width,height,sx,sy,prevw,px,py);}
CW_ENTRY void c32_wave1_mapped(const unsigned char*in,const float*fw,const float*w,unsigned char*out,uint windows,uint mode,uint raw,uint width,uint height,uint sx,uint sy){cw_body<false>(in,fw,w,out,windows,mode,raw,width,height,sx,sy,0,0,0);}
#define CW_FINISH(NAME,DC) CW_ENTRY void NAME(const unsigned char*in,const float*fw,const float*w,float*main,float*down,uint windows,uint mode,uint raw,uint width,uint height,uint sx,uint sy,uint prevw,uint px,uint py){cw_body<true,true,DC>(in,fw,w,nullptr,windows,mode,raw,width,height,sx,sy,prevw,px,py,main,down);}
CW_FINISH(c32_wave1_finish,false)
CW_FINISH(c32_wave1_finish_dcrop,true)
CW_ENTRY void c32_wave1_post(const unsigned char*low,const unsigned char*skip,const float*scales,const float*fw,const float*w,const float*color,const float*headw,float*rgb,uint windows,uint width,uint height,uint sx,uint sy,float scale){cw_body<false,false,false,true>(low,fw,w,nullptr,windows,0,1,width,height,sx,sy,0,0,0,nullptr,nullptr,skip,scales,color,headw,rgb,scale);}
CW_ENTRY void c32_wave1_post_logit(const unsigned char*low,const unsigned char*skip,const float*scales,const float*fw,const float*w,const float*color,const float*headw,float*rgb,uint windows,uint width,uint height,uint sx,uint sy,float scale,const float*logitw,float*logit){cw_body<false,false,false,true>(low,fw,w,nullptr,windows,0,1,width,height,sx,sy,0,0,0,nullptr,nullptr,skip,scales,color,headw,rgb,scale,nullptr,0,0,nullptr,nullptr,logitw,logit);}
CW_ENTRY void c32_wave1_prefix(const unsigned char*rgba,const float*history,const float*fw,const float*w,float*main,float*down,uint windows,uint mode,uint raw,uint width,uint height,uint seed,uint temporal){cw_body<false,true,false,false,true>(rgba,fw,w,nullptr,windows,mode,raw,width,height,0,0,0,0,0,main,down,nullptr,nullptr,nullptr,nullptr,nullptr,0.f,history,seed,temporal);}

#if CW_SKIP_BYTE
CW_ENTRY void c32_wave1_finish_dcrop_b8(const unsigned char*in,const float*fw,const float*w,unsigned char*main,float*down,uint windows,uint mode,uint raw,uint width,uint height,uint sx,uint sy,uint prevw,uint px,uint py){cw_body<true,true,true,false,false,false,true>(in,fw,w,nullptr,windows,mode,raw,width,height,sx,sy,prevw,px,py,reinterpret_cast<float*>(main),down);}
CW_ENTRY void c32_wave1_finish_dcrop_b8d(const unsigned char*in,const float*fw,const float*w,unsigned char*main,unsigned char*down,uint windows,uint mode,uint raw,uint width,uint height,uint sx,uint sy,uint prevw,uint px,uint py){cw_body<true,true,true,false,false,false,true,true>(in,fw,w,nullptr,windows,mode,raw,width,height,sx,sy,prevw,px,py,reinterpret_cast<float*>(main),reinterpret_cast<float*>(down));}
#endif
#if CW_PREPOST_BYTE
CW_ENTRY void c32_wave1_prefix_b8d(const unsigned char*rgba,const float*history,const float*fw,const float*w,unsigned char*main,unsigned char*down,uint windows,uint mode,uint raw,uint width,uint height,uint seed,uint temporal){cw_body<false,true,false,false,true,false,false,true>(rgba,fw,w,nullptr,windows,mode,raw,width,height,0,0,0,0,0,reinterpret_cast<float*>(main),reinterpret_cast<float*>(down),nullptr,nullptr,nullptr,nullptr,nullptr,0.f,history,seed,temporal);}
CW_ENTRY void c32_wave1_mapped_b8(const unsigned char*in,const float*fw,const float*w,unsigned char*out,uint windows,uint mode,uint raw,uint width,uint height,uint sx,uint sy){cw_body<false,false,false,false,false,false,false,false,true>(in,fw,w,out,windows,mode,raw,width,height,sx,sy,0,0,0);}
CW_ENTRY void c32_wave1_finish_b8(const unsigned char*in,const float*fw,const float*w,unsigned char*main,float*down,uint windows,uint mode,uint raw,uint width,uint height,uint sx,uint sy,uint prevw,uint px,uint py){cw_body<true,true,false,false,false,false,true>(in,fw,w,nullptr,windows,mode,raw,width,height,sx,sy,prevw,px,py,reinterpret_cast<float*>(main),down);}
CW_ENTRY void c32_wave1_post_b8(const unsigned char*low,const unsigned char*skip,const float*scales,const float*fw,const float*w,const float*color,const float*headw,float*rgb,uint windows,uint width,uint height,uint sx,uint sy,float scale){cw_body<false,false,false,true,false,false,false,false,true>(low,fw,w,nullptr,windows,0,1,width,height,sx,sy,0,0,0,nullptr,nullptr,skip,scales,color,headw,rgb,scale);}
CW_ENTRY void c32_wave1_post_b8_logit(const unsigned char*low,const unsigned char*skip,const float*scales,const float*fw,const float*w,const float*color,const float*headw,float*rgb,uint windows,uint width,uint height,uint sx,uint sy,float scale,const float*logitw,float*logit){cw_body<false,false,false,true,false,false,false,false,true>(low,fw,w,nullptr,windows,0,1,width,height,sx,sy,0,0,0,nullptr,nullptr,skip,scales,color,headw,rgb,scale,nullptr,0,0,nullptr,nullptr,logitw,logit);}
CW_ENTRY void c32_wave1_post_b8_rgba(const unsigned char*low,const unsigned char*skip,const float*scales,const float*fw,const float*w,const float*color,const float*headw,float*rgb,uint windows,uint width,uint height,uint sx,uint sy,float scale){cw_body<false,false,false,true,false,false,false,false,true,false,true>(low,fw,w,nullptr,windows,0,1,width,height,sx,sy,0,0,0,nullptr,nullptr,skip,scales,color,headw,rgb,scale);}
// Diagnostic/paired tap: final post features before the head. Old RGB ABI and arithmetic stay intact.
CW_ENTRY void c32_wave1_post_b8_features(const unsigned char*low,const unsigned char*skip,const float*scales,const float*fw,const float*w,const float*color,const float*headw,float*rgb,uint windows,uint width,uint height,uint sx,uint sy,float scale,_Float16*features){cw_body<false,false,false,true,false,false,false,false,true,false,false,true>(low,fw,w,nullptr,windows,0,1,width,height,sx,sy,0,0,0,nullptr,nullptr,skip,scales,color,headw,rgb,scale,nullptr,0,0,nullptr,features);}

#endif
#if CW_UP_FUSED
#if CW_SKIP_BYTE
CW_ENTRY void c32_wave1_up_b8(const unsigned char*low,const float*upw,const unsigned char*skip,const float*fw,const float*aw,unsigned char*out,uint windows,uint width,uint height,uint sx,uint sy){cw_body<false,false,false,false,false,true,true>(low,fw,aw,out,windows,0,1,width,height,sx,sy,0,0,0,nullptr,nullptr,skip,nullptr,nullptr,nullptr,nullptr,0.f,nullptr,0,0,upw);}
#endif
CW_ENTRY void c32_wave1_up(const unsigned char*low,const float*upw,const float*skip,const float*fw,const float*aw,unsigned char*out,uint windows,uint width,uint height,uint sx,uint sy){cw_body<false,false,false,false,false,true>(low,fw,aw,out,windows,0,1,width,height,sx,sy,0,0,0,nullptr,nullptr,reinterpret_cast<const unsigned char*>(skip),nullptr,nullptr,nullptr,nullptr,0.f,nullptr,0,0,upw);}
/* CW_UP_LOW_BYTES (2026-10-01, swin-body-gap): the fused Up's low-resolution input (block 65, post 0: F(Hrtz), on the E4M3 lattice)
   arrives as E4M3 bytes; cvt back gives the same (_Float16) operand. New exports only (*_lb); the host uses them only when present. */
#ifndef CW_UP_LOW_BYTES
#define CW_UP_LOW_BYTES 0
#endif
#if CW_UP_LOW_BYTES
CW_ENTRY void c32_wave1_up_lb(const unsigned char*low,const float*upw,const float*skip,const float*fw,const float*aw,unsigned char*out,uint windows,uint width,uint height,uint sx,uint sy){cw_body<false,false,false,false,false,true,false,false,false,true>(low,fw,aw,out,windows,0,1,width,height,sx,sy,0,0,0,nullptr,nullptr,reinterpret_cast<const unsigned char*>(skip),nullptr,nullptr,nullptr,nullptr,0.f,nullptr,0,0,upw);}
#if CW_SKIP_BYTE
CW_ENTRY void c32_wave1_up_b8_lb(const unsigned char*low,const float*upw,const unsigned char*skip,const float*fw,const float*aw,unsigned char*out,uint windows,uint width,uint height,uint sx,uint sy){cw_body<false,false,false,false,false,true,true,false,false,true>(low,fw,aw,out,windows,0,1,width,height,sx,sy,0,0,0,nullptr,nullptr,skip,nullptr,nullptr,nullptr,nullptr,0.f,nullptr,0,0,upw);}
#endif
#endif
#endif

