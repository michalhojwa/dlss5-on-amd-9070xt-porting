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

// SDKless geometry and exact head. Float tensors; all kernels block256.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
typedef unsigned uint;
#define DEV __attribute__((device)) __attribute__((always_inline))
#define KERNEL extern "C" __attribute__((global))
DEV uint bits(float x){return __builtin_bit_cast(uint,x);}
DEV float value(uint x){return __builtin_bit_cast(float,x);}
DEV uint index1(){return __builtin_amdgcn_workgroup_id_x()*256u+__builtin_amdgcn_workitem_id_x();}
DEV float absf(float x){return value(bits(x)&0x7fffffffu);}
DEV uint rshift(uint x,uint n){if(n>31)return 0;uint q=x>>n,r=x&((1u<<n)-1u),m=1u<<(n-1);return q+uint(r>m||(r==m&&(q&1)));}
DEV float half_from(uint h){uint s=(h&0x8000u)<<16,e=(h>>10)&31u,m=h&1023;if(e==31)return value(s|0x7f800000u|(m<<13));if(!e){if(!m)return value(s);float f=float(m)*5.9604644775390625e-8f;return s?-f:f;}return value(s|((e+112)<<23)|(m<<13));}
#ifdef HIP_ISA_HALF
// Keep f32 expression evaluation and both rounding boundaries opaque to fusion.
DEV float H(float x){uint h;float result;asm("v_cvt_f16_f32 %0, %1" : "=v"(h) : "v"(x));asm("v_cvt_f32_f16 %0, %1" : "=v"(result) : "v"(h));return result;}
#else
DEV float H(float x){uint b=bits(x),a=b&0x7fffffffu,s=(b>>16)&0x8000u;int e=int(a>>23)-127;if(a>=0x7f800000u)return x;if(e<-25)return value(b&0x80000000u);if(e<-14)return half_from(s|rshift((a&0x7fffffu)|0x800000u,uint(-e-1)));if(e>15)return half_from(s|0x7c00u);uint h=((a+0xfffu+((a>>13)&1u))>>13)-0x1c000u;return half_from(s|(h>=0x7c00u?0x7c00u:h));}
#endif
DEV uint reflected(uint p,uint size){if(size<=1)return 0;uint n=p%(2*size-2);return n<size?n:2*size-2-n;}
DEV uint tile_pixel(uint x,uint y,uint width){return ((y/8)*(width/8)+x/8)*64+(y%8)*8+x%8;}
KERNEL void hip_input_reflect(const float*input,float*base,float*tiles,uint vw,uint vh,uint pw,uint ph){uint p=index1();if(p>=pw*ph)return;uint x=p%pw,y=p/pw,s=(reflected(y,vh)*vw+reflected(x,vw))*4,t=tile_pixel(x,y,pw)*4;for(uint c=0;c<4;c++){float v=input[s+c];base[p*4+c]=v;tiles[t+c]=v;}}
KERNEL void hip_c32_pack(const float*input,float*output,uint w,uint h,uint ww,uint wh,uint sx,uint sy){uint i=index1();if(i>=ww*wh*32)return;uint p=i/32,c=i%32,t=p/64,x=(t%(ww/8))*8+p%8,y=(t/(ww/8))*8+(p%64)/8;int a=int(x)-int(sx),b=int(y)-int(sy);output[i]=a>=0&&b>=0&&a<int(w)&&b<int(h)?input[(uint(b)*w+uint(a))*32+c]:0;}
KERNEL void hip_crop(const float*input,float*output,uint w,uint h,uint workw,uint sx,uint sy,uint channels,uint tiled){uint i=index1();if(i>=w*h*channels)return;uint p=i/channels,c=i%channels,x=p%w+sx,y=p/w+sy;uint src=tiled?tile_pixel(x,y,workw):y*workw+x;output[i]=input[src*channels+c];}
KERNEL void hip_post_merge(const float*low,const float*skip,const float*weights,float*output,uint w,uint h){uint i=index1();if(i>=w*h*32)return;uint p=i/32,c=i%32,q=(p/w/2)*(w/2)+(p%w)/2;output[i]=H(H(low[q*32+c]*weights[c])+skip[i]*weights[32+c]);}
DEV float aligned_half(int sum,float acc,float scale,int e){
 float scaled=acc*scale;if(scaled!=float(int(scaled))||absf(scaled)>=67108864.f)return value(0x7fc00000u);
 bool neg=sum<0,aneg=acc<0;uint m=neg?0u-uint(sum):uint(sum),other=uint(absf(scaled));
 if(neg==aneg)m+=other;else if(m>=other)m-=other;else{m=other-m;neg=aneg;}if(!m)return 0;
 int quantum=e-27,top=31-__builtin_clz(m),step=(top+quantum>-14?top+quantum:-14)-10,drop=step-quantum;uint r=m;
 if(drop>32)r=0;else if(drop==32)r=m>0x80000000u?1:0;else if(drop>0)r=rshift(m,uint(drop));else step=quantum;
 float result=float(r)*value(uint(step+127)<<23);if(result>=65520.f)result=value(0x7f800000u);return neg?-result:result;
}
KERNEL void hip_post_head_exact(const float*raw,const float*base,const float*weights,float*output,uint w,uint h,uint workw,uint sx,uint sy,float input_scale){
 uint i=index1();if(i>=w*h*3)return;uint p=i/3,row=i%3,fb=tile_pixel(p%w+sx,p/w+sy,workw)*32;float acc=0;
 for(uint part=0;part<2;part++){
  float products[16];int e=acc==0?-1000:int((bits(acc)>>23)&255u)-125;
  for(uint j=0;j<16;j++){float a=raw[fb+part*16+j],b=weights[row*32+part*16+j];products[j]=a*b;if(products[j]!=0){int k=int((bits(a)>>23)&255u)+int((bits(b)>>23)&255u)-252;e=e>k?e:k;}}
  if(e!=-1000){float scale=value(uint(27-e+127)<<23);int sum=0;for(uint j=0;j<16;j++)sum+=int(products[j]*scale);acc=aligned_half(sum,acc,scale,e);}
 }
 float centered=base[p*4+row]*.125f-.0625f,encoded=acc*input_scale+centered,rgb=encoded*8.f+.5f;output[i]=rgb<0?0:rgb>1?1:rgb;
}

