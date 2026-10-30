#define HIP_ISA_HALF 1
#define CW_PACK_MODE_MASK 0
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

// Native gfx1201 FP8 WMMA C32 FFN. Host ABI remains f32; kernels are separate from the scalar reference.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
typedef unsigned uint;
#define DEV __attribute__((device)) __attribute__((always_inline))
#define KERNEL extern "C" __attribute__((global))
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
DEV float F(float x){uint b=bits(x),a=b&0x7fffffffu,s=b&0x80000000u;if(a==0)return 0.f;
 if(a>=0x43e00000u)return value(s|0x43e00000u); // finite E4M3 saturation, including overflow
 if(a<0x3c800000u){float scaled=absf(x)*512.f;uint q=uint(scaled);float r=scaled-float(q);q+=uint(r>.5f||(r==.5f&&(q&1u)));float f=float(q)/512.f;return s?-f:f;}
 uint r=(a+0x7ffffu+((a>>20)&1u))&0xfff00000u;return value(s|(r>0x43e00000u?0x43e00000u:r));}

using int2=int __attribute__((ext_vector_type(2)));
using float8=float __attribute__((ext_vector_type(8)));
DEV uint packed_fp8(float x){x=__builtin_amdgcn_fmed3f(x,-448.f,448.f);return uint(__builtin_amdgcn_cvt_pk_fp8_f32(x,x,0,false))&255u;}
// One wave computes 16 tokens x 16 output channels. A row and B column use lane%16;
// each half-wave supplies eight K entries, and D rows are half-wave*8+element.
KERNEL __attribute__((amdgpu_flat_work_group_size(32,32)))
void c32_ffn_expand_wmma(const float* input,const float* weights,float* hidden,uint tokens){
 uint block=__builtin_amdgcn_workgroup_id_x(),first=(block/8)*16u,col=(block%8)*16u;
 if(first>=tokens||col>=128)return;
 uint lane=__builtin_amdgcn_workitem_id_x(),rc=lane&15,group=lane>>4;
 float8 sum{};
 #pragma unroll
 for(uint tile=0;tile<2;tile++){
  int2 a{},b{};
  #pragma unroll
  for(uint e=0;e<8;e++){uint k=tile*16+group*8+e;uint av=first+rc<tokens?packed_fp8(F(input[(first+rc)*32+k])):0u;uint bv=packed_fp8(weights[512+(col+rc)*32+k]);a[e/4]=int(uint(a[e/4])|(av<<(8*(e%4))));b[e/4]=int(uint(b[e/4])|(bv<<(8*(e%4))));}
  sum=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,sum);
 }
 #pragma unroll
 for(uint e=0;e<8;e++){uint p=first+group*8+e;if(p>=tokens)continue;float a=H(sum[e]),g=clampf(a,-4.f,4.f);float poly=H(g*H(absf(g)*(-.055908203125f)+.447265625f)+.89453125f);hidden[p*128+col+rc]=F(H(a*poly));}
}
KERNEL __attribute__((amdgpu_flat_work_group_size(32,32)))
void c32_ffn_contract_wmma(const float* input,const float* hidden,const float* weights,float* output,uint tokens,uint raw_output){
 uint block=__builtin_amdgcn_workgroup_id_x(),first=(block/2)*16u,col=(block%2)*16u;
 if(first>=tokens||col>=32)return;
 uint lane=__builtin_amdgcn_workitem_id_x(),rc=lane&15,group=lane>>4;
 float8 residual{};
 #pragma unroll
 for(uint e=0;e<8;e++){uint p=first+group*8+e;residual[e]=p<tokens?H(input[p*32+col+rc]*weights[8704+col+rc]):0.f;}
 #pragma unroll
 for(uint chunk=0;chunk<4;chunk++){
  // Keep the residual outside the K16 MMA seed: sum(K32) starts from zero.
  float8 sum{};
  #pragma unroll
  for(uint tile=0;tile<2;tile++){
   int2 a{},b{};
   #pragma unroll
   for(uint e=0;e<8;e++){uint k=chunk*32+tile*16+group*8+e;uint av=first+rc<tokens?packed_fp8(hidden[(first+rc)*128+k]):0u;uint bv=packed_fp8(weights[4608+(col+rc)*128+k]);a[e/4]=int(uint(a[e/4])|(av<<(8*(e%4))));b[e/4]=int(uint(b[e/4])|(bv<<(8*(e%4))));}
   sum=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,sum);
  }
  #pragma unroll
  for(uint e=0;e<8;e++)residual[e]=H(residual[e]+sum[e]);
 }
 #pragma unroll
 for(uint e=0;e<8;e++){uint p=first+group*8+e;if(p<tokens)output[p*32+col+rc]=raw_output?residual[e]:F(residual[e]);}
}

DEV float half_add_midpoint(float a,float b){float sum=a+b,vb=sum-a,error=(a-(sum-vb))+(b-vb);uint u=bits(sum),m=u&0x7fffffffu;
 if(m>=0x38800000u&&m<0x47800000u&&(m&0x1fffu)==0x1000u&&error!=0)sum=value(((error>0)==(sum>0))?u+1:u-1);return H(sum);}
DEV float exp_approx(float x){float a=clampf(H(x*.044921875f+1.30078125f),1.03125f,1.5693359375f);return from_half(((half_bits(a)<<5)+0x8000u)&65535u);}

// Bases already select the wave lane's A row and B column. K stride is in floats.
DEV float8 matrix32(const float*a,const float*b,uint abase,uint bbase,uint astep,uint bstep){
 uint group=__builtin_amdgcn_workitem_id_x()>>4;float8 sum{};
 #pragma unroll
 for(uint tile=0;tile<2;tile++){int2 pa{},pb{};
  #pragma unroll
  for(uint e=0;e<8;e++){uint k=tile*16+group*8+e;uint av=packed_fp8(a[abase+k*astep]),bv=packed_fp8(b[bbase+k*bstep]);pa[e/4]=int(uint(pa[e/4])|(av<<(8*(e%4))));pb[e/4]=int(uint(pb[e/4])|(bv<<(8*(e%4))));}
  sum=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(pa,pb,sum);
 }return sum;
}
KERNEL __attribute__((amdgpu_flat_work_group_size(32,32)))
void c32_attn_qkv_wmma(const float*input,const float*weights,float*qkv,uint tokens){
 uint block=__builtin_amdgcn_workgroup_id_x(),perpart=tokens/8,part=block/perpart,local=block%perpart,first=(local/2)*16,col=(local%2)*16;
 if(part>=3)return;uint lane=__builtin_amdgcn_workitem_id_x(),rc=lane&15,group=lane>>4;
 float8 sum=matrix32(input,weights,(first+rc)*32,part*1024+(col+rc)*32,1,1);
 #pragma unroll
 for(uint e=0;e<8;e++){uint p=first+group*8+e;float v=H(sum[e]);qkv[(part*tokens+p)*32+col+rc]=part==2?F(v):v;}
}
KERNEL __attribute__((amdgpu_flat_work_group_size(32,32)))
void c32_attn_scores_wmma(const float*normalized,const float*weights,float*ex,uint windows){
 uint block=__builtin_amdgcn_workgroup_id_x(),w=block/16,qr=((block%16)/4)*16,kr=(block%4)*16;
 if(w>=windows)return;uint lane=__builtin_amdgcn_workitem_id_x(),rc=lane&15,group=lane>>4,tokens=windows*64;
 float8 sum=matrix32(normalized,normalized,(w*64+qr+rc)*32,(tokens+w*64+kr+rc)*32,1,1);
 #pragma unroll
 for(uint e=0;e<8;e++){uint q=qr+group*8+e,k=kr+rc;ex[(w*64+q)*64+k]=exp_approx(H(sum[e]+weights[4096+q*64+k]));}
}
KERNEL __attribute__((amdgpu_flat_work_group_size(32,32)))
void c32_attn_av_wmma(const float*prob,const float*qkv,float*av,uint windows){
 uint block=__builtin_amdgcn_workgroup_id_x(),w=block/8,qr=((block%8)/2)*16,col=(block%2)*16;
 if(w>=windows)return;uint lane=__builtin_amdgcn_workitem_id_x(),rc=lane&15,group=lane>>4,tokens=windows*64;float8 acc{};
 #pragma unroll
 for(uint chunk=0;chunk<2;chunk++){float8 sum=matrix32(prob,qkv,(w*64+qr+rc)*64+chunk*32,(2*tokens+w*64+chunk*32)*32+col+rc,1,32);
  #pragma unroll
  for(uint e=0;e<8;e++)acc[e]=H(acc[e]+sum[e]);
 }
 #pragma unroll
 for(uint e=0;e<8;e++)av[(w*64+qr+group*8+e)*32+col+rc]=F(acc[e]);
}
KERNEL __attribute__((amdgpu_flat_work_group_size(32,32)))
void c32_attn_project_wmma(const float*input,const float*av,const float*weights,float*output,uint tokens,uint raw_output){
 uint block=__builtin_amdgcn_workgroup_id_x(),first=(block/2)*16,col=(block%2)*16;
 if(first>=tokens)return;uint lane=__builtin_amdgcn_workitem_id_x(),rc=lane&15,group=lane>>4;
 float8 sum=matrix32(av,weights,(first+rc)*32,3072+(col+rc)*32,1,1);
 #pragma unroll
 for(uint e=0;e<8;e++){uint p=first+group*8+e;float result=half_add_midpoint(sum[e],H(input[p*32+col+rc]*weights[8193+col+rc]));output[p*32+col+rc]=raw_output?result:F(result);}
}

