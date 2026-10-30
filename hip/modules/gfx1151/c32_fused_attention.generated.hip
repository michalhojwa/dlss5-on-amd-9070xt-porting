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

// Production C32 FAST4 attention precision, separate from exact reference.
// All kernels flat grid ceil(work_items/256), block32; token count divisible64.
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

using i2=int __attribute__((ext_vector_type(2)));
using f8=float __attribute__((ext_vector_type(8)));
using h8=_Float16 __attribute__((ext_vector_type(8)));
#define WAVE KERNEL __attribute__((amdgpu_flat_work_group_size(32,32)))
DEV uint rc(){return __builtin_amdgcn_workitem_id_x()&15;}
DEV uint gr(){return (__builtin_amdgcn_workitem_id_x()&31)>>4;}
DEV uint bid(){return __builtin_amdgcn_workgroup_id_x();}
// D3D explicit f32tof16 maps to RTZ here; matrix Cast<F16> remains RNE.
DEV float Hrtz(float x){uint b=bits(x),a=b&0x7fffffffu,sg=(b>>16)&0x8000u;int e=int(a>>23)-127;
 if(a>=0x7f800000u)return from_half(sg|0x7c00u|((a&0x7fffffu)?0x200u:0u));
 if(e>15)return from_half(sg|0x7bffu);
 if(e<-24)return from_half(sg);
 if(e<-14)return from_half(sg|(((a&0x7fffffu)|0x800000u)>>uint(-e-1)));
 return from_half(sg|((a>>13)-0x1c000u));
}
DEV uint fp8(float f){return uint(__builtin_amdgcn_cvt_pk_fp8_f32(clampf(f,-448.f,448.f),0.f,0,false))&255u;}
DEV void put(i2&v,uint e,float x){v[e/4]=int(uint(v[e/4])|(fp8(x)<<(8*(e%4))));}

DEV void sync_window(){__builtin_amdgcn_fence(3,"workgroup");__builtin_amdgcn_s_barrier();__builtin_amdgcn_fence(2,"workgroup");}

// Packed attention: shared raw/exp scratch 8448 bytes + normalized/prob/AV bytes 6912.
using ushort=unsigned short;
DEV void put_bits(i2&v,uint e,uint b){v[e/4]=int(uint(v[e/4])|((b&255u)<<(8*(e%4))));}
KERNEL __attribute__((amdgpu_flat_work_group_size(128,128)))
void c32_fast_attention_fused(const float*raw_input,const float*w,float*out,uint windows,uint raw_output){
 uint window=bid();if(window>=windows)return;uint tid=__builtin_amdgcn_workitem_id_x(),wave=tid/32,first=wave*16,base=window*64;
 union Scratch {float raw[64*33];_Float16 ex[64*66];};
 __attribute__((shared)) Scratch scratch;
 __attribute__((shared)) unsigned char packed[3*64*36];
 // A single raw QKV plane is reused after its norm has been packed separately.
 for(uint part=0;part<3;part++){
  for(uint col=0;col<32;col+=16){f8 acc{};
   for(uint kt=0;kt<2;kt++){i2 a{},b{};for(uint e=0;e<8;e++){uint k=kt*16+gr()*8+e;put(a,e,raw_input[(base+first+rc())*32+k]);put(b,e,w[part*1024+(col+rc())*32+k]);}acc=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,acc);}
   for(uint e=0;e<8;e++)scratch.raw[(first+gr()*8+e)*33+col+rc()]=acc[e];
  }
  sync_window();f8 sum{};
  if(part<2)for(uint kt=0;kt<2;kt++){h8 a{},ones{};for(uint e=0;e<8;e++){uint k=kt*16+gr()*8+e;float v=scratch.raw[(first+rc())*33+k];a[e]=(_Float16)(v*v);ones[e]=(_Float16)1.f;}sum=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,ones,sum);}
  for(uint col=0;col<32;col+=16)for(uint e=0;e<8;e++){uint row=first+gr()*8+e;float v=scratch.raw[row*33+col+rc()];if(part<2)v*=__builtin_amdgcn_rsqf(maxf(sum[e],6.198883056640625e-5f))*(part==0?w[8192]:1.f);packed[(part*64+row)*36+col+rc()]=static_cast<unsigned char>(fp8(v));}
  sync_window();
 }
 // Q/K packed rows: 36-byte stride. Exponent values stored exactly as half bits.
 for(uint key=0;key<64;key+=16){f8 acc{};
  for(uint kt=0;kt<2;kt++){i2 a{},b{};for(uint e=0;e<8;e++){uint k=kt*16+gr()*8+e;put_bits(a,e,packed[(first+rc())*36+k]);put_bits(b,e,packed[(64+key+rc())*36+k]);}acc=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,acc);}
  for(uint e=0;e<8;e++){uint q=first+gr()*8+e,k=key+rc();float score=acc[e]+w[4096+q*64+k],affine=clampf(score*.044921875f+1.30078125f,1.03125f,1.5693359375f);uint ah=(bits(affine)>>13)-0x1c000u;ushort hb=static_cast<ushort>(((ah<<5)+0x8000u)&65535u);scratch.ex[q*66+k]=__builtin_bit_cast(_Float16,hb);}
 }
 sync_window();
 // Scores no longer consume Q/K; their 4608 bytes become 64 probability rows of 68 bytes.
 {f8 sum{};for(uint kt=0;kt<4;kt++){h8 a{},ones{};for(uint e=0;e<8;e++){uint k=kt*16+gr()*8+e;a[e]=scratch.ex[(first+rc())*66+k];ones[e]=(_Float16)1.f;}sum=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a,ones,sum);}
  for(uint key=0;key<64;key+=16)for(uint e=0;e<8;e++){uint row=first+gr()*8+e,k=key+rc();float v=float(scratch.ex[row*66+k])*(1.f/sum[e]);packed[row*68+k]=static_cast<unsigned char>(fp8(v));}
 }
 sync_window();
 // Both AV column fragments stay live until every wave has finished reading probability rows.
 f8 av[2];
 for(uint ci=0;ci<2;ci++){f8 acc{};uint col=ci*16;
  for(uint kt=0;kt<4;kt++){i2 a{},b{};for(uint e=0;e<8;e++){uint k=kt*16+gr()*8+e;put_bits(a,e,packed[(first+rc())*68+k]);put_bits(b,e,packed[(128+k)*36+col+rc()]);}acc=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,acc);}av[ci]=acc;
 }
 sync_window();
 for(uint ci=0;ci<2;ci++)for(uint e=0;e<8;e++)packed[(first+gr()*8+e)*36+ci*16+rc()]=static_cast<unsigned char>(fp8(av[ci][e]));
 sync_window();
 for(uint col=0;col<32;col+=16){f8 acc{};
  for(uint kt=0;kt<2;kt++){i2 a{},b{};for(uint e=0;e<8;e++){uint k=kt*16+gr()*8+e;put_bits(a,e,packed[(first+rc())*36+k]);put(b,e,w[3072+(col+rc())*32+k]);}acc=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,acc);}
  for(uint e=0;e<8;e++){uint i=(base+first+gr()*8+e)*32+col+rc();float prod=raw_input[i]*w[8193+col+rc()],v=Hrtz(acc[e]+prod);out[i]=raw_output?v:F(v);}
 }
}

