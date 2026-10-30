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

#define HIP_ISA_HALF 1
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
typedef unsigned uint;
#define DEV __attribute__((device)) __attribute__((always_inline))
#define KERNEL extern "C" __attribute__((global))
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
DEV float F(float x){uint b=bits(x),a=b&0x7fffffffu,s=b&0x80000000u;if(a==0)return 0.f;
 if(a>=0x43e00000u)return value(s|0x43e00000u); // finite E4M3 saturation, including overflow
 if(a<0x3c800000u){float scaled=absf(x)*512.f;uint q=uint(scaled);float r=scaled-float(q);q+=uint(r>.5f||(r==.5f&&(q&1u)));float f=float(q)/512.f;return s?-f:f;}
 uint r=(a+0x7ffffu+((a>>20)&1u))&0xfff00000u;return value(s|(r>0x43e00000u?0x43e00000u:r));}

using int2=int __attribute__((ext_vector_type(2)));
using float8=float __attribute__((ext_vector_type(8)));
DEV uint packed_fp8(float x){x=__builtin_amdgcn_fmed3f(x,-448.f,448.f);return uint(__builtin_amdgcn_cvt_pk_fp8_f32(x,x,0,false))&255u;}
DEV float half_add_midpoint(float a,float b){float sum=a+b,vb=sum-a,error=(a-(sum-vb))+(b-vb);uint u=bits(sum),m=u&0x7fffffffu;
 if(m>=0x38800000u&&m<0x47800000u&&(m&0x1fffu)==0x1000u&&error!=0)sum=value(((error>0)==(sum>0))?u+1:u-1);return H(sum);}

// Production D3D f32tof16 on this driver truncates toward zero; do not reuse exact/RNE H().
DEV float Hrtz(float x){uint b=bits(x),a=b&0x7fffffffu,sign=(b>>16)&0x8000u;int e=int(a>>23)-127;
 if(a>=0x7f800000u)return x;
 if(e<-24)return from_half(sign);
 if(e<-14)return from_half(sign|(((a&0x7fffffu)|0x800000u)>>uint(-e-1)));
 if(e>15)return from_half(sign|0x7bffu);
 return from_half(sign|((a>>13)-0x1c000u));
}

using half8=_Float16 __attribute__((ext_vector_type(8)));
DEV uint production_pcg(uint s){uint w=((s>>((s>>28)+4))^s)*0x108ef2d9u;return (w>>22)^w;}
DEV float production_uniform(uint s){uint w=((s>>((s>>28)+4))^s)*0x108ef2d9u;return float(((w>>30)^(w>>8))+1)*5.9604644775390625e-8f;}
KERNEL void dlss5_prefix_fast_features(const float*rgba,const float*history,float*features,uint width,uint height,uint seed,uint temporal){
 uint p=__builtin_amdgcn_workgroup_id_x()*256+__builtin_amdgcn_workitem_id_x();if(p>=width*height)return;
 uint tile=p/64,x=(tile%(width/8))*8+p%8,y=(tile/(width/8))*8+(p%64)/8;
 uint h=production_pcg((x*0x8da6b343u)^(y*0xd8163841u)^(seed*0x9e3779b9u)^0x243f6a88u);
 float a=production_uniform(h*0xcaa5b80du+0x21dd796bu),b=production_uniform(h*0x2c9277b5u+0xac564b05u),c=production_uniform(h*0x83232c31u+0x3463e0acu),d=production_uniform(h*0xfa6dc5f9u+0x4712a88eu);
 float r0=__builtin_amdgcn_sqrtf(-2.f*(__builtin_amdgcn_logf(a)*.6931471805599453f)),r1=__builtin_amdgcn_sqrtf(-2.f*(__builtin_amdgcn_logf(b)*.6931471805599453f));
 float ct=(6.283185482025146f*c)*.15915494309189535f,dt=(6.283185482025146f*d)*.15915494309189535f;
 float g0=Hrtz(r0*__builtin_amdgcn_cosf(ct)),g1=Hrtz(r1*__builtin_amdgcn_cosf(dt)),g2=Hrtz(r1*__builtin_amdgcn_sinf(dt));
 float red=Hrtz(Hrtz(Hrtz(rgba[p*4])-.5f)*.125f),green=Hrtz(Hrtz(Hrtz(rgba[p*4+1])-.5f)*.125f),blue=Hrtz(Hrtz(Hrtz(rgba[p*4+2])-.5f)*.125f);
 float f[16]={g1,g2,green,blue,g0,1,dlss5_style_feature,1,red,green,1,1,blue,red,1,0};
 if(temporal){uint q=(y*width+x)*4;f[13]=Hrtz(Hrtz(Hrtz(history[q])-.5f)*.125f);f[2]=Hrtz(Hrtz(Hrtz(history[q+1])-.5f)*.125f);f[3]=Hrtz(Hrtz(Hrtz(history[q+2])-.5f)*.125f);}
 for(uint i=0;i<32;i++)features[p*32+i]=i<16?f[i]:0.f;
}
KERNEL __attribute__((amdgpu_flat_work_group_size(32,32)))
void dlss5_prefix_fast_project(const float*features,const float*weights,float*prefix,uint tokens){
 uint block=__builtin_amdgcn_workgroup_id_x(),first=(block/2)*16,col=(block%2)*16;if(first>=tokens)return;
 uint lane=__builtin_amdgcn_workitem_id_x(),rc=lane&15,group=lane>>4;float8 sum{};
 #pragma unroll
 for(uint tile=0;tile<2;tile++){half8 a{},b{};
  #pragma unroll
  for(uint e=0;e<8;e++){uint k=tile*16+group*8+e;a[e]=(_Float16)(first+rc<tokens?features[(first+rc)*32+k]:0.f);b[e]=(_Float16)(k<16?weights[(col+rc)*16+k]:0.f);}
  sum=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,b,sum);
 }
 #pragma unroll
 for(uint e=0;e<8;e++){uint p=first+group*8+e;if(p<tokens)prefix[p*32+col+rc]=Hrtz(sum[e]);}
}

template<bool Raster=false>
DEV void fused_prefix_values(const float*rgba,const float*history,uint p,uint width,uint seed,uint temporal,float*f){
 uint tile=p/64,x=(tile%(width/8))*8+p%8,y=(tile/(width/8))*8+(p%64)/8;
 uint rgbp=Raster?(y*width+x):p;
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
KERNEL __attribute__((amdgpu_flat_work_group_size(32,32)))
void dlss5_prefix_fast_fused(const float*rgba,const float*history,const float*weights,float*prefix,uint width,uint height,uint seed,uint temporal){
 uint first=__builtin_amdgcn_workgroup_id_x()*16,lane=__builtin_amdgcn_workitem_id_x(),rc=lane&15,group=lane>>4,tokens=width*height;
 __attribute__((shared)) float values[16*16];
 if(lane<16){float f[16]{};if(first+lane<tokens)fused_prefix_values(rgba,history,first+lane,width,seed,temporal,f);for(uint k=0;k<16;k++)values[lane*16+k]=f[k];}
 __builtin_amdgcn_s_barrier();
 float8 sum[2]{};
 for(uint tile=0;tile<2;tile++){half8 a{};for(uint e=0;e<8;e++){uint k=tile*16+group*8+e;a[e]=(_Float16)(k<16?values[rc*16+k]:0.f);}
  for(uint col=0;col<2;col++){half8 b{};for(uint e=0;e<8;e++){uint k=tile*16+group*8+e;b[e]=(_Float16)(k<16?weights[(col*16+rc)*16+k]:0.f);}sum[col]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,b,sum[col]);}
 }
 for(uint col=0;col<2;col++)for(uint e=0;e<8;e++){uint p=first+group*8+e;if(p<tokens)prefix[p*32+col*16+rc]=Hrtz(sum[col][e]);}
}

KERNEL __attribute__((amdgpu_flat_work_group_size(32,32)))
void dlss5_prefix_fast_fused_raster(const float*rgba,const float*history,const float*weights,float*prefix,uint width,uint height,uint seed,uint temporal){
 uint first=__builtin_amdgcn_workgroup_id_x()*16,lane=__builtin_amdgcn_workitem_id_x(),rc=lane&15,group=lane>>4,tokens=width*height;
 __attribute__((shared)) float values[16*16];
 if(lane<16){float f[16]{};if(first+lane<tokens)fused_prefix_values<true>(rgba,history,first+lane,width,seed,temporal,f);for(uint k=0;k<16;k++)values[lane*16+k]=f[k];}
 __builtin_amdgcn_s_barrier();
 float8 sum[2]{};
 for(uint tile=0;tile<2;tile++){half8 a{};for(uint e=0;e<8;e++){uint k=tile*16+group*8+e;a[e]=(_Float16)(k<16?values[rc*16+k]:0.f);}
  for(uint col=0;col<2;col++){half8 b{};for(uint e=0;e<8;e++){uint k=tile*16+group*8+e;b[e]=(_Float16)(k<16?weights[(col*16+rc)*16+k]:0.f);}sum[col]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,b,sum[col]);}
 }
 for(uint col=0;col<2;col++)for(uint e=0;e<8;e++){uint p=first+group*8+e;if(p<tokens)prefix[p*32+col*16+rc]=Hrtz(sum[col][e]);}
}

