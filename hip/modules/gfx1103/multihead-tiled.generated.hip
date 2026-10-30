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

// Native gfx1201 FP8 WMMA multihead kernels. Original externalfloat32 ABI.
// Mirrors native_c64.hlsl non-wave math, native_c64_shift.hlsl, native_c64_ds.hlsl.
// Host validates geometry/channel contracts; all tensors/weights externally float32.
// Every exported kernel uses one-dimensional blocks of exactly512 threads.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
typedef unsigned uint;
#define DEV __attribute__((device)) __attribute__((always_inline))
#define KERNEL extern "C" __attribute__((global))
DEV uint bits(float x){union{float f;uint u;}v;v.f=x;return v.u;}
DEV float value(uint x){union{float f;uint u;}v;v.u=x;return v.f;}
DEV uint lane(){return __builtin_amdgcn_workgroup_id_x()*256u+__builtin_amdgcn_workitem_id_x();}
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
DEV float H(float x){if((bits(x)&0x7fffffffu)>=0x7f800000u)return x;return from_half(half_bits(x));}
#endif
DEV float F(float x){uint b=bits(x),a=b&0x7fffffffu,s=b&0x80000000u;if(a==0)return 0.f;
 if(a>=0x43e00000u)return value(s|0x43e00000u); // finite E4M3 saturation, including overflow
 if(a<0x3c800000u){float scaled=absf(x)*512.f;uint q=uint(scaled);float r=scaled-float(q);q+=uint(r>.5f||(r==.5f&&(q&1u)));float f=float(q)/512.f;return s?-f:f;}
 uint r=(a+0x7ffffu+((a>>20)&1u))&0xfff00000u;return value(s|(r>0x43e00000u?0x43e00000u:r));}
DEV float half_square_pair(float lower,float upper){float a=lower*lower,b=H(upper*upper),sum=a+b;
 if((bits(sum)&0x7fffffffu)>=0x7f800000u)return H(sum);
 float large=maxf(a,b),small=minf(a,b),recovered=sum-large,error=small-recovered;
 if(sum==65520.f)return error<0?65504.f:H(sum);
 float rounded=H(sum);if(error==0||(bits(rounded)&0x7fffffffu)>=0x7f800000u||rounded==sum)return rounded;
 uint hb=half_bits(rounded),lo_bits=sum<rounded?hb-1:hb;float lo=from_half(lo_bits),hi=from_half(lo_bits+1),mid=(lo+hi)*.5f;
 return sum==mid?(error>0?hi:lo):rounded;
}
DEV float exp_approx(float x){float a=clampf(H(x*.044921875f+1.30078125f),1.03125f,1.5693359375f);return from_half(((half_bits(a)<<5)+0x8000u)&65535u);}
DEV bool valid_c(uint c){return c==64u||c==128u||c==256u;}
DEV bool valid_attn(uint c){return valid_c(c)||c==512u;}
DEV bool valid_spatial(uint c){return c==32u||valid_attn(c);}
DEV uint pixel_in_window(uint window,uint token,uint width){return ((window/(width/8))*8+token/8)*width+(window%(width/8))*8+token%8;}
DEV uint window_of_pixel(uint p,uint width){return (p/width/8)*(width/8)+(p%width)/8;}
DEV uint token_of_pixel(uint p,uint width){return ((p/width)%8)*8+(p%width)%8;}

#if !defined(HIP_WMMA_SHIM) && !__has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12)
#error "gfx12 FP8 WMMA compiler required; no scalar fallback"
#endif
using i2=int __attribute__((ext_vector_type(2)));
using f8=float __attribute__((ext_vector_type(8)));
#define TILE KERNEL __attribute__((amdgpu_flat_work_group_size(512,512)))
DEV uint q8(float x){return uint(__builtin_amdgcn_cvt_pk_fp8_f32(x,x,0,false))&255u;}
DEV void pack(i2&v,uint e,float x){v[e/4]=int(uint(v[e/4])|(q8(x)<<(8*(e%4))));}
//64x64 output tile,16 waves. Each thread stages one packed4-byte A row word
// and one B column word; each packed operand is reused by four waves.
DEV uint pack4(float a,float b,float c,float d){uint lo=uint(__builtin_amdgcn_cvt_pk_fp8_f32(a,b,0,false));uint hi=uint(__builtin_amdgcn_cvt_pk_fp8_f32(c,d,0,false));return (lo&65535u)|((hi&65535u)<<16);}
DEV void tiled_dense(const float* input,const float* feature,const float* weights,float* output,
 uint M,uint N,uint K,uint matrix_offset,uint skip_offset,uint epilogue,uint width=0,uint valid_width=0,uint valid_height=0){
 __attribute__((shared)) uint sa[512];
 __attribute__((shared)) uint sb[512];
 uint block=__builtin_amdgcn_workgroup_id_x(),nt=(N+63)/64;
 if(!nt||block>=((M+63)/64)*nt)return;
 uint mr=block/nt*64,nc=block%nt*64,t=__builtin_amdgcn_workitem_id_x(),wave=t/32,l=t%32,rc=l&15,group=l>>4;
 uint wm=wave/4,wn=wave%4,first=mr+wm*16,col=nc+wn*16;
 f8 accum{};
 #pragma unroll
 for(uint e=0;e<8;e++){uint p=first+group*8+e,n=col+rc;if(p<M&&n<N&&skip_offset!=~0u)accum[e]=H(feature[p*N+n]*weights[skip_offset+n]);}
 for(uint chunk=0;chunk<K/32;chunk++){
  uint row=t/8,word=t%8,k=chunk*32+word*4,p=mr+row,n=nc+row;
  float a0=0,a1=0,a2=0,a3=0,b0=0,b1=0,b2=0,b3=0;
  if(p<M&&(!valid_width||(p%width<valid_width&&p/width<valid_height))){uint off=p*K+k;a0=input[off];a1=input[off+1];a2=input[off+2];a3=input[off+3];}
  if(n<N){uint off=matrix_offset+n*K+k;b0=weights[off];b1=weights[off+1];b2=weights[off+2];b3=weights[off+3];}
  sa[t]=pack4(a0,a1,a2,a3);sb[t]=pack4(b0,b1,b2,b3);
  __builtin_amdgcn_s_barrier();
  f8 sum{};
  #pragma unroll
  for(uint half=0;half<2;half++){
   uint word0=half*4+group*2;
   i2 a={int(sa[(wm*16+rc)*8+word0]),int(sa[(wm*16+rc)*8+word0+1])};
   i2 b={int(sb[(wn*16+rc)*8+word0]),int(sb[(wn*16+rc)*8+word0+1])};
   sum=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a,b,sum);
  }
  // All sixteen waves finish reading before any writer reuses this K32 tile.
  __builtin_amdgcn_s_barrier();
  #pragma unroll
  for(uint e=0;e<8;e++)accum[e]=H(accum[e]+sum[e]);
 }
 #pragma unroll
 for(uint e=0;e<8;e++){uint p=first+group*8+e,n=col+rc;if(p>=M||n>=N)continue;float a=accum[e],v;
  if(epilogue==1){float gate=clampf(a,-4.f,4.f),poly=H(gate*H(absf(gate)*(-.055908203125f)+.447265625f)+.89453125f);v=F(H(a*poly));}
  else v=epilogue==2?a:F(a);
  output[p*N+n]=(valid_width&&(p%width>=valid_width||p/width>=valid_height))?0.f:v;
 }
}
TILE void mh_ffn_expand_tiled(const float* input,const float* weights,float* hidden,uint tokens,uint channels){if(valid_c(channels))tiled_dense(input,input,weights,hidden,tokens,4*channels,channels,0,~0u,1);}
TILE void mh_ffn_contract_tiled(const float* hidden,const float* weights,float* middle,uint tokens,uint channels){if(valid_c(channels))tiled_dense(hidden,hidden,weights,middle,tokens,channels,4*channels,4*channels*channels,~0u,0);}
TILE void mh_ffn_project_tiled(const float* middle,const float* feature,const float* weights,float* output,uint tokens,uint channels){if(valid_c(channels))tiled_dense(middle,feature,weights,output,tokens,channels,channels,8*channels*channels,9*channels*channels,0);}
TILE void mh_qkv_tiled(const float* input,const float* weights,float* qkv,uint tokens,uint channels){if(valid_attn(channels))tiled_dense(input,input,weights,qkv,tokens,3*channels,channels,0,~0u,2);}
TILE void mh_attention_project_tiled(const float* av,const float* feature,const float* weights,float* output,uint tokens,uint channels,uint raw_output){if(valid_attn(channels))tiled_dense(av,feature,weights,output,tokens,channels,channels,3*channels*channels,4*channels*channels+(channels/32)*4096+channels/32,raw_output?2:0);}
TILE void mh_pool_project_tiled(const float* pooled,const float* weights,float* output,uint width,uint height,uint valid_width,uint valid_height,uint channels){if(valid_spatial(channels))tiled_dense(pooled,pooled,weights,output,width*height,2*channels,channels,0,~0u,0,width,valid_width,valid_height);}
// Arbitrary dense sizes for tail-N32/M400 and residual-placement validation.
// This extra entry is test-only; original family exports above retain their ABI.
TILE void mh_dense_tiled_probe(const float* input,const float* feature,const float* weights,float* output,uint M,uint N,uint K,uint skip_offset,uint raw){if(K&&!(K%32))tiled_dense(input,feature,weights,output,M,N,K,0,skip_offset,raw?2:0);}

