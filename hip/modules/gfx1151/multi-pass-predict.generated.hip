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

// Same RGB patch gain for all channels. Two real passes, predict only pass three.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
typedef unsigned uint;
#define KERNEL extern "C" __attribute__((global))
#define DEV __attribute__((device)) __attribute__((always_inline))
DEV float clip(float v,float lo,float hi){return __builtin_fminf(__builtin_fmaxf(v,lo),hi);}
// bpermute returns uint bits; retain float reinterpretation.
DEV float sum32(float v){for(uint s=16;s;s>>=1){union{float f;uint u;}a,b;a.f=v;b.u=__builtin_amdgcn_ds_bpermute(((__builtin_amdgcn_workitem_id_x()+s)&31)*4,a.u);v+=b.f;}return v;}
KERNEL __attribute__((amdgpu_flat_work_group_size(32,32)))
void mp_predict_gain(const float*x,const float*y1,const float*y2,float*gain,uint w,uint h){
 uint lane=__builtin_amdgcn_workitem_id_x(),tile=__builtin_amdgcn_workgroup_id_x(),cols=(w+15)/16;
 float aa=0,bb=0,ab=0,count=0;
 for(uint k=lane;k<256;k+=32){uint px=(tile%cols)*16+k%16,py=(tile/cols)*16+k/16;if(px>=w||py>=h)continue;uint p=py*w+px;count+=1;
 for(uint c=0;c<3;c++){float a=y1[p*3+c]-x[p*4+c],b=y2[p*3+c]-y1[p*3+c];aa+=a*a;bb+=b*b;ab+=a*b;}}
 aa=sum32(aa);bb=sum32(bb);ab=sum32(ab);count=sum32(count);
 if(!lane){float eps=count*1.e-7f;float r=0;if(aa>=eps&&ab>0&&ab*ab>=.25f*aa*bb)r=clip(ab/(aa+eps),0,1);gain[tile]=r;}
}
KERNEL void mp_predict_apply(const float*y1,const float*y2,const float*gain,float*out,uint w,uint h){
 uint p=__builtin_amdgcn_workgroup_id_x()*256+__builtin_amdgcn_workitem_id_x();if(p>=w*h)return;
 uint cols=(w+15)/16,rows=(h+15)/16;float tx=clip((float(p%w)+.5f)/16.f-.5f,0,float(cols-1)),ty=clip((float(p/w)+.5f)/16.f-.5f,0,float(rows-1));uint ix=uint(tx),iy=uint(ty),jx=ix+1<cols?ix+1:ix,jy=iy+1<rows?iy+1:iy;float fx=tx-ix,fy=ty-iy;
 float top=gain[iy*cols+ix]*(1-fx)+gain[iy*cols+jx]*fx,bot=gain[jy*cols+ix]*(1-fx)+gain[jy*cols+jx]*fx,r=top*(1-fy)+bot*fy;
 for(uint c=0;c<3;c++){uint i=p*3+c;out[i]=clip(y2[i]+r*(y2[i]-y1[i]),0,1);}
}

// Direct RGBA first pass, remaining inputs/output keep RGB.
KERNEL __attribute__((amdgpu_flat_work_group_size(32,32)))
void mp_predict_gain_stride(const float*x,const float*y1,const float*y2,float*gain,uint w,uint h,uint first_stride){
 uint lane=__builtin_amdgcn_workitem_id_x(),tile=__builtin_amdgcn_workgroup_id_x(),cols=(w+15)/16;
 float aa=0,bb=0,ab=0,count=0;
 for(uint k=lane;k<256;k+=32){uint px=(tile%cols)*16+k%16,py=(tile/cols)*16+k/16;if(px>=w||py>=h)continue;uint p=py*w+px;count+=1;
 for(uint c=0;c<3;c++){float a=y1[p*first_stride+c]-x[p*4+c],b=y2[p*3+c]-y1[p*first_stride+c];aa+=a*a;bb+=b*b;ab+=a*b;}}
 aa=sum32(aa);bb=sum32(bb);ab=sum32(ab);count=sum32(count);
 if(!lane){float eps=count*1.e-7f;float r=0;if(aa>=eps&&ab>0&&ab*ab>=.25f*aa*bb)r=clip(ab/(aa+eps),0,1);gain[tile]=r;}
}
KERNEL void mp_predict_apply_stride(const float*y1,const float*y2,const float*gain,float*out,uint w,uint h,uint first_stride){
 uint p=__builtin_amdgcn_workgroup_id_x()*256+__builtin_amdgcn_workitem_id_x();if(p>=w*h)return;
 uint cols=(w+15)/16,rows=(h+15)/16;float tx=clip((float(p%w)+.5f)/16.f-.5f,0,float(cols-1)),ty=clip((float(p/w)+.5f)/16.f-.5f,0,float(rows-1));uint ix=uint(tx),iy=uint(ty),jx=ix+1<cols?ix+1:ix,jy=iy+1<rows?iy+1:iy;float fx=tx-ix,fy=ty-iy;
 float top=gain[iy*cols+ix]*(1-fx)+gain[iy*cols+jx]*fx,bot=gain[jy*cols+ix]*(1-fx)+gain[jy*cols+jx]*fx,r=top*(1-fy)+bot*fy;
 for(uint c=0;c<3;c++){uint i=p*3+c;out[i]=clip(y2[i]+r*(y2[i]-y1[p*first_stride+c]),0,1);}
}

