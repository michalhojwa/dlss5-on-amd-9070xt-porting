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

// Production FAST3/ffn_fused ordinary-residual schedule, modes 0/1 only. Separate from exact C32.
#define HIP_ISA_HALF 1
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
// Match NativePreblockRuntime::e4m3r packing, including its subnormal q<=7 rule.
DEV float scale_piece(float v){float m=absf(v);if(m<.015625f){float x=m*512.f;uint q=uint(x);float r=x-float(q);q+=uint(r>.5f||(r==.5f&&(q&1u)));if(q>7)q=7;return value((bits(v)&0x80000000u)|bits(float(q)/512.f));}return F(v);}
DEV uint pack4(const float*p){
 float x0=__builtin_amdgcn_fmed3f(p[0],-448.f,448.f),x1=__builtin_amdgcn_fmed3f(p[1],-448.f,448.f);
 float x2=__builtin_amdgcn_fmed3f(p[2],-448.f,448.f),x3=__builtin_amdgcn_fmed3f(p[3],-448.f,448.f);
 int lo=__builtin_amdgcn_cvt_pk_fp8_f32(x0,x1,0,false);
 return uint(__builtin_amdgcn_cvt_pk_fp8_f32(x2,x3,lo,true));
}
DEV void sync_block(){__builtin_amdgcn_fence(3,"workgroup");__builtin_amdgcn_s_barrier();__builtin_amdgcn_fence(2,"workgroup");}
// Mode 0 expand, 1 contract, 2 QKV, 3 projection. LDS rows hold eight packed K32 words.
template<uint Mode,bool Map3=false> DEV void tiled(const float*input,const float*operand,const float*weights,float*output,uint tokens,uint raw_output){
 constexpr uint N=Mode==0?64:32,Threads=Mode==0?512:256,K=Mode==1?128:32;
 __attribute__((shared)) uint a8[64*8];
 __attribute__((shared)) uint b8[N*8];
 uint id=__builtin_amdgcn_workgroup_id_x(),blocks=(tokens+63)/64,part=0,first=0,col=0;
 if(Mode==0){first=(id/2)*64;col=(id%2)*64;}else if(Mode==2){part=id/blocks;first=(id%blocks)*64;if(part>=3)return;}else first=id*64;
 if(first>=tokens)return;
 uint tid=__builtin_amdgcn_workitem_id_x(),wave=tid/32,lane=tid%32,rc=lane&15,half=lane>>4;
 uint wave_row=(wave/(N/16))*16,wave_col=(wave%(N/16))*16;
 float8 acc{};
 if(Mode==1&&!Map3){
  #pragma unroll
  for(uint e=0;e<8;e++){uint p=first+wave_row+half*8+e;acc[e]=p<tokens?clampf(input[p*32+wave_col+rc],-448.f,448.f)*weights[8704+wave_col+rc]:0.f;}
 }
 if(Mode==1&&Map3){
  // Production CHAIN_RAW residual: quantized input times three E4M3 diagonal matrices.
  for(uint i=tid;i<64*8;i+=Threads){uint row=i/8,k=(i%8)*4;a8[i]=first+row<tokens?pack4(input+(first+row)*32+k):0u;}
  #pragma unroll
  for(uint part=0;part<3;part++){
   for(uint i=tid;i<N*8;i+=Threads){uint c=i/8,k0=(i%8)*4;float remaining=weights[8704+c],piece=0;
    #pragma unroll
    for(uint j=0;j<=part;j++){piece=scale_piece(remaining);remaining=remaining-piece;}
    uint word=0;if(c>=k0&&c<k0+4)word=packed_fp8(piece)<<(8*(c-k0));b8[i]=word;
   }
   sync_block();
   #pragma unroll
   for(uint tile=0;tile<2;tile++){uint ai=(wave_row+rc)*8+tile*4+half*2,bi=(wave_col+rc)*8+tile*4+half*2;int2 a={int(a8[ai]),int(a8[ai+1])},b={int(b8[bi]),int(b8[bi+1])};acc=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,acc);}
   sync_block();
  }
 }
 #pragma unroll
 for(uint chunk=0;chunk<K/32;chunk++){
  // Coalesced f32 loads and exactly one conversion per input/weight value per block.
  for(uint i=tid;i<64*8;i+=Threads){uint row=i/8,k=(i%8)*4+chunk*32;const float*src=Mode==1||Mode==3?operand:input;uint stride=Mode==1?128:32;a8[i]=first+row<tokens?pack4(src+(first+row)*stride+k):0u;}
  for(uint i=tid;i<N*8;i+=Threads){uint row=col+i/8,k=(i%8)*4+chunk*32;uint offset=Mode==0?512:Mode==1?4608:Mode==2?part*1024:3072;b8[i]=pack4(weights+offset+row*K+k);}
  sync_block();float8 sum=Mode==1?acc:float8{};
  #pragma unroll
  for(uint tile=0;tile<2;tile++){uint ai=(wave_row+rc)*8+tile*4+half*2,bi=(wave_col+rc)*8+tile*4+half*2;
   int2 a={int(a8[ai]),int(a8[ai+1])},b={int(b8[bi]),int(b8[bi+1])};
   sum=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,sum);
  }
  if(Mode==1){
   #pragma unroll
   for(uint e=0;e<8;e++)acc[e]=sum[e];
  }else acc=sum;
  sync_block();
 }
 #pragma unroll
 for(uint e=0;e<8;e++){uint p=first+wave_row+half*8+e,c=col+wave_col+rc;if(p>=tokens)continue;float v=acc[e];
  if(Mode==0){float a=v,g=clampf(a,-4.f,4.f),q=__builtin_fmaf(absf(g),-.055908203125f,.447265625f),poly=__builtin_fmaf(g,q,.89453125f);output[p*128+c]=F(a*poly);}
  else if(Mode==1){if(raw_output==2)output[p*32+c]=v;else{v=Hrtz(v);output[p*32+c]=raw_output?v:F(v);}}
  else if(Mode==2){v=H(v);output[(part*tokens+p)*32+c]=part==2?F(v):v;}
  else{v=half_add_midpoint(v,H(input[p*32+c]*weights[8193+c]));output[p*32+c]=raw_output?v:F(v);}
 }
}
KERNEL __attribute__((amdgpu_flat_work_group_size(512,512)))
void c32_ffn_expand_fast(const float*input,const float*weights,float*hidden,uint tokens){tiled<0>(input,input,weights,hidden,tokens,1);}
KERNEL __attribute__((amdgpu_flat_work_group_size(256,256)))
void c32_ffn_contract_fast(const float*input,const float*hidden,const float*weights,float*output,uint tokens,uint raw_output){tiled<1>(input,hidden,weights,output,tokens,raw_output);}

KERNEL __attribute__((amdgpu_flat_work_group_size(256,256)))
void c32_ffn_contract_fast_map3(const float*input,const float*hidden,const float*weights,float*output,uint tokens,uint raw_output){tiled<1,true>(input,hidden,weights,output,tokens,raw_output);}

