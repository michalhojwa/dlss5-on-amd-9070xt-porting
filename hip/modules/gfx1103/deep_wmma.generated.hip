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

// Native gfx1201 F16 WMMA deep matrices. f32 ABI; all operands must be exact finite half values.
// Flat grid ceil(output_work_items/256), block32. One block = 16 tokens x 16 channels.
// Tokens and output channels must be divisible by16; no scalar GEMM fallback.
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

using f8=float __attribute__((ext_vector_type(8)));
using h8=_Float16 __attribute__((ext_vector_type(8)));
#define WAVE KERNEL __attribute__((amdgpu_flat_work_group_size(32,32)))
DEV uint rc(){return __builtin_amdgcn_workitem_id_x()&15u;}
DEV uint group(){return __builtin_amdgcn_workitem_id_x()>>4;}
DEV uint bid(){return __builtin_amdgcn_workgroup_id_x();}
DEV float activation(float a){float g=clampf(a,-4.f,4.f);return F(H(a*H(g*H(absf(g)*(-.055908203125f)+.447265625f)+.89453125f)));}
// K32 group sum begins at zero; round H(previous + group) only after BOTH K16 MMAs.
DEV f8 dot(const float*a,const float*b,uint stride_a,uint stride_b,uint first,uint row,uint off_a,uint off_b,uint begin,uint end,f8 acc,uint valid_tokens=~0u){
 for(uint k=begin;k<end;k+=32){f8 sum{};
  #pragma unroll
  for(uint tile=0;tile<2;tile++){h8 av{},bv{};
   #pragma unroll
   for(uint e=0;e<8;e++){uint j=k+tile*16+group()*8+e;av[e]=first+rc()<valid_tokens?(_Float16)a[(first+rc())*stride_a+off_a+j]:(_Float16)0;bv[e]=(_Float16)b[off_b+(row+rc())*stride_b+j];}
   sum=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(av,bv,sum);
  }
  #pragma unroll
  for(uint e=0;e<8;e++)acc[e]=H(acc[e]+sum[e]);
 }
 return acc;
}
DEV f8 residual(const float*input,const float*weights,uint first,uint row,uint channels,uint offset){f8 a{};
 #pragma unroll
 for(uint e=0;e<8;e++)a[e]=H(input[(first+group()*8+e)*channels+row+rc()]*weights[offset+row+rc()]);return a;
}
DEV void store(float*out,f8 a,uint first,uint row,uint channels,uint activated){
 #pragma unroll
 for(uint e=0;e<8;e++)out[(first+group()*8+e)*channels+row+rc()]=activated?activation(a[e]):F(a[e]);
}
WAVE void split_mix(const float*input,const float*weights,float*output,uint tokens){uint first=bid()/32*16,row=bid()%32*16;if(first>=tokens)return;f8 a{};a=dot(input,weights,512,512,first,row,0,0,0,512,a);store(output,a,first,row,512,0);}
WAVE void split_expand(const float*input,const float*weights,float*output,uint tokens){uint first=bid()/128*16,col=bid()%128*16,g=col/256,row=col%256;if(first>=tokens)return;f8 a{};a=dot(input,weights,512,64,first,row,g*64,262144+g*16384,0,64,a);store(output,a,first,col,2048,1);}
WAVE void split_contract(const float*input,const float*weights,float*output,uint tokens){uint first=bid()/32*16,col=bid()%32*16,g=col/64,row=col%64;if(first>=tokens)return;f8 a{};a=dot(input,weights,2048,256,first,row,g*256,393216+g*16384,0,256,a);store(output,a,first,col,512,0);}
WAVE void split_projection(const float*input,const float*weights,const float*skip,float*output,uint tokens){uint first=bid()/32*16,row=bid()%32*16;if(first>=tokens)return;f8 a=residual(skip,weights,first,row,512,262144);a=dot(input,weights,512,512,first,row,0,0,0,512,a);store(output,a,first,row,512,0);}
WAVE void vit_expand(const float*input,const float*weights,float*output,uint tokens,uint inputs,uint outputs){uint tiles=outputs/16,first=bid()/tiles*16,row=bid()%tiles*16;if(first>=tokens)return;f8 a{};a=dot(input,weights,inputs,inputs,first,row,0,0,0,inputs,a);store(output,a,first,row,outputs,1);}
WAVE void vit_project(const float*input,const float*weights,const float*skip,float*output,uint tokens,uint inputs,uint outputs){uint tiles=outputs/16,first=bid()/tiles*16,row=bid()%tiles*16;if(first>=tokens)return;f8 total{};
 for(uint p=0;p<4;p++){f8 a{};if(p==0)a=residual(skip,weights,first,row,outputs,inputs*outputs);a=dot(input,weights,inputs,inputs,first,row,0,0,p*(inputs/4),(p+1)*(inputs/4),a);
  #pragma unroll
  for(uint e=0;e<8;e++)total[e]=p?H(total[e]+a[e]):a[e];
 }store(output,total,first,row,outputs,0);
}
WAVE void vit_qkv_project(const float*input,const float*weights,float*output,uint tokens){uint perpart=(tokens/16)*64,part=bid()/perpart,local=bid()%perpart,first=local/64*16,row=local%64*16;if(part>=3)return;f8 total{};
 for(uint p=0;p<2;p++){f8 a{};a=dot(input,weights,1024,1024,first,row,0,part*1048576,p*512,(p+1)*512,a);
  #pragma unroll
  for(uint e=0;e<8;e++)total[e]=p?H(total[e]+a[e]):a[e];
 }
 #pragma unroll
 for(uint e=0;e<8;e++)output[(part*tokens+first+group()*8+e)*1024+row+rc()]=total[e];
}
WAVE void decoder_project2x(const float*input,const float*weights,const float*skip,float*output,uint iw,uint ih,uint ow,uint oh,uint inputs,uint outputs){uint tiles=outputs/16,first=bid()/tiles*16,row=bid()%tiles*16;if(first>=iw*ih)return;uint partitions=inputs==1024?4:1;f8 total{};
 for(uint p=0;p<partitions;p++){f8 a{};a=dot(input,weights,inputs,inputs,first,row,0,0,p*(inputs/partitions),(p+1)*(inputs/partitions),a,iw*ih);
  #pragma unroll
  for(uint e=0;e<8;e++)total[e]=p?H(total[e]+a[e]):a[e];
 }
 #pragma unroll
 for(uint e=0;e<8;e++){uint token=first+group()*8+e;if(token>=iw*ih)continue;
  #pragma unroll
  for(uint dy=0;dy<2;dy++){
   #pragma unroll
   for(uint dx=0;dx<2;dx++){uint x=(token%iw)*2+dx,y=(token/iw)*2+dy;if(x>=ow||y>=oh)continue;uint i=(y*ow+x)*outputs+row+rc();float merged=H(total[e]+F(skip[i])*weights[inputs*outputs+row+rc()]);output[i]=outputs==32?merged:F(merged);}
  }
 }
}

