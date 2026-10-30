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

// SDKless scalar GPU reference, not a WMMA performance backend.
// Every kernel uses one-dimensional blocks of exactly 256 threads.
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
DEV float H(float x){return from_half(half_bits(x));}
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
DEV float half_add_midpoint(float a,float b){float sum=a+b,vb=sum-a,error=(a-(sum-vb))+(b-vb);uint u=bits(sum),m=u&0x7fffffffu;
 if(m>=0x38800000u&&m<0x47800000u&&(m&0x1fffu)==0x1000u&&error!=0)sum=value(((error>0)==(sum>0))?u+1:u-1);return H(sum);}
DEV float exp_approx(float x){float a=clampf(H(x*.044921875f+1.30078125f),1.03125f,1.5693359375f);return from_half(((half_bits(a)<<5)+0x8000u)&65535u);}
// All input/output feature tensors up through projection are tile-major [window][64][32].
KERNEL void c32_ffn_expand(const float* input,const float* weights,float* hidden,uint tokens){uint i=lane();if(i>=tokens*128u)return;uint p=i/128u,row=i%128u;float sum=0;
 for(uint j=0;j<32;j++)sum+=F(input[p*32+j])*weights[512+row*32+j];
 float a=H(sum),g=clampf(a,-4.f,4.f);float poly=H(g*H(absf(g)*(-.055908203125f)+.447265625f)+.89453125f);hidden[i]=F(H(a*poly));}
KERNEL void c32_ffn_contract(const float* input,const float* hidden,const float* weights,float* output,uint tokens,uint raw_output){uint i=lane();if(i>=tokens*32u)return;uint p=i/32u,c=i%32u;float a=H(input[i]*weights[8704+c]);
 for(uint group=0;group<4;group++){float s=0;for(uint j=0;j<32;j++)s+=hidden[p*128+group*32+j]*weights[4608+c*128+group*32+j];a=H(a+s);}output[i]=raw_output?a:F(a);}
// qkv layout [part=Q,K,V][tokens][32]; Q/K stay H values, V is F(H).
KERNEL void c32_attn_qkv(const float* input,const float* weights,float* qkv,uint tokens){uint i=lane();if(i>=tokens*96u)return;uint part=i/(tokens*32u),pc=i%(tokens*32u),p=pc/32u,c=pc%32u;float sum=0;
 for(uint j=0;j<32;j++)sum+=F(input[p*32+j])*weights[part*1024+c*32+j];qkv[i]=part==2?F(H(sum)):H(sum);}
// One GPU thread per token, 16-value reductions reproduce the original half reduction tree.
KERNEL void c32_attn_normalize(const float* qkv,const float* weights,float* normalized,uint tokens){uint p=lane();if(p>=tokens)return;
 for(uint part=0;part<2;part++){float sums[16];uint base=(part*tokens+p)*32;
  for(uint i=0;i<16;i++)sums[i]=half_square_pair(qkv[base+i],qkv[base+i+16]);
  for(uint i=0;i<8;i++)sums[i]=H(sums[i*2]+sums[i*2+1]);
  for(uint step=4;step>0;step/=2)for(uint i=0;i<step;i++)sums[i]=H(sums[i]+sums[i+step]);
  float inv=H(__builtin_amdgcn_rsqf(maxf(sums[0],6.198883056640625e-5f)));
  for(uint c=0;c<32;c++){float x=H(qkv[base+c]*inv);normalized[base+c]=F(part==0?H(x*H(weights[8192])):x);}
 }
}
// scores/ex/prob layout [window][query64][key64], no standard softmax substitution.
KERNEL void c32_attn_scores(const float* normalized,const float* weights,float* ex,uint windows){uint i=lane();if(i>=windows*4096u)return;uint w=i/4096u,q=(i/64u)%64u,k=i%64u,tokens=windows*64u;float dot=0;
 for(uint c=0;c<32;c++)dot+=normalized[(w*64+q)*32+c]*normalized[(tokens+w*64+k)*32+c];ex[i]=exp_approx(H(dot+weights[4096+q*64+k]));}
KERNEL void c32_attn_probabilities(const float* ex,float* prob,uint windows){uint p=lane();if(p>=windows*64u)return;uint row=p*64u;float parity[2];
 for(uint odd=0;odd<2;odd++){float total=0;for(uint l=0;l<4;l++){uint base=odd+(l%2)*2+(l/2)*8;float partial=H(ex[row+base]+ex[row+base+16]);partial=H(partial+H(ex[row+base+4]+ex[row+base+20]));partial=H(partial+H(ex[row+base+32]+ex[row+base+48]));partial=H(partial+H(ex[row+base+36]+ex[row+base+52]));total=l==0?partial:H(total+partial);}parity[odd]=total;}
 float inv=H(1.f/H(parity[0]+parity[1]));for(uint k=0;k<64;k++)prob[row+k]=F(H(ex[row+k]*inv));}
KERNEL void c32_attn_av(const float* prob,const float* qkv,float* av,uint windows){uint i=lane(),tokens=windows*64u;if(i>=tokens*32u)return;uint p=i/32u,c=i%32u,w=p/64u,q=p%64u;float a=0;
 for(uint group=0;group<2;group++){float s=0;for(uint key=0;key<32;key++){uint k=group*32+key;s+=prob[(w*64+q)*64+k]*qkv[(2*tokens+w*64+k)*32+c];}a=H(a+s);}av[i]=F(a);}
KERNEL void c32_attn_project(const float* input,const float* av,const float* weights,float* output,uint tokens,uint raw_output){uint i=lane();if(i>=tokens*32u)return;uint p=i/32u,c=i%32u;float a=0;
 for(uint j=0;j<32;j++)a+=av[p*32+j]*weights[3072+c*32+j];float result=half_add_midpoint(a,H(input[i]*weights[8193+c]));output[i]=raw_output?result:F(result);}
DEV uint tile_index(uint x,uint y,uint c,uint width){uint tile=(y/8)*(width/8)+x/8;return (tile*64+(y%8)*8+x%8)*32+c;}
// main and down are raster HWC. Pool reads raw values before F(), exactly as preblock_finish.
KERNEL void c32_finish(const float* raw,float* main,float* down,uint width,uint height){uint i=lane();if(i>=width*height*32u)return;uint p=i/32u,c=i%32u,x=p%width,y=p/width;main[i]=F(raw[tile_index(x,y,c,width)]);
 if(!(x&1u)&&!(y&1u)){float top=H(raw[tile_index(x,y,c,width)]+raw[tile_index(x+1,y,c,width)]),bottom=H(raw[tile_index(x,y+1,c,width)]+raw[tile_index(x+1,y+1,c,width)]);down[((y/2)*(width/2)+x/2)*32+c]=F(H(H(top+bottom)*.25f));}}

// Append after c32_reference.hip: uses its SDKless DEV/KERNEL, H(), bits/value helpers.
// Exact noise-table prefix from preblock_input_mix.hlsl, LIVE_PROFILE=1, NATIVE_FAST_PREFIX=0.
// block256; pixels are 8x8 tile-major RGBA. History is processing-grid raster RGBA.
// DLSS5_STYLE (2026-10-01): NGX Style control as preprocess feature 6 = Style/128. Default 1/128 (Style=1, the value captured
// from the original runtime on 09-06); the host overwrites it per module with hipModuleGetGlobal when DLSS5_STYLE is set.
#ifndef DLSS5_STYLE_FEATURE_DEFINED
#define DLSS5_STYLE_FEATURE_DEFINED 1
__attribute__((constant)) float dlss5_style_feature=.0078125f;
#endif
DEV uint prefix_pcg(uint s){uint w=((s>>((s>>28)+4))^s)*0x108ef2d9u;return (w>>22)^w;}
DEV uint prefix_uniform_index(uint s){uint w=((s>>((s>>28)+4))^s)*0x108ef2d9u;return (w>>30)^(w>>8);}
DEV uint prefix_round_shift(uint m,int shift){
 if(shift<=0)return m<<uint(-shift);
 if(shift>=32)return shift==32&&m>0x80000000u?1u:0u;
 uint q=m>>shift,rem=m&((1u<<shift)-1u),half=1u<<(shift-1);
 return q+uint(rem>half||(rem==half&&(q&1u)));
}
DEV float prefix_scaled_half(int n,int exponent){
 uint sign=n<0?0x8000u:0u,m=n<0?0u-uint(n):uint(n);if(!m)return 0;
 int lead=31-__builtin_clz(m),e=lead+exponent;uint b;
 if(e < -14)b=prefix_round_shift(m,-exponent-24);
 else{uint q=prefix_round_shift(m,lead-10);if(q==2048u){q=1024;e++;}b=e>15?0x7c00u:(uint(e+15)<<10)|(q-1024u);}
 return from_half(sign|b);
}
KERNEL void dlss5_prefix_reference(const float*rgba_tiles,const float*history_raster,const float*noise,const float*weights,float*prefix,float*debug_features,uint width,uint height,uint seed,uint temporal){
 uint p=lane();if(p>=width*height)return;
 uint tile=p/64,x=(tile%(width/8))*8+p%8,y=(tile/(width/8))*8+(p%64)/8;
 uint h=prefix_pcg((x*0x8da6b343u)^(y*0xd8163841u)^(seed*0x9e3779b9u)^0x243f6a88u);
 uint ia=prefix_uniform_index(h*0xcaa5b80du+0x21dd796bu),ib=prefix_uniform_index(h*0x2c9277b5u+0xac564b05u);
 uint ic=prefix_uniform_index(h*0x83232c31u+0x3463e0acu),id=prefix_uniform_index(h*0xfa6dc5f9u+0x4712a88eu);
 float g0=H(noise[ia*3]*noise[ic*3+1]),g1=H(noise[ib*3]*noise[id*3+1]),g2=H(noise[ib*3]*noise[id*3+2]);
 float r=H(H(H(rgba_tiles[p*4])-.5f)*.125f),g=H(H(H(rgba_tiles[p*4+1])-.5f)*.125f),b=H(H(H(rgba_tiles[p*4+2])-.5f)*.125f);
 float f[16]={g1,g2,g,b,g0,1,dlss5_style_feature,1,r,g,1,1,b,r,1,0};
 if(temporal){uint q=(y*width+x)*4;f[13]=H(H(H(history_raster[q])-.5f)*.125f);f[2]=H(H(H(history_raster[q+1])-.5f)*.125f);f[3]=H(H(H(history_raster[q+2])-.5f)*.125f);}
 if(debug_features)for(uint i=0;i<32;i++)debug_features[p*32+i]=i<16?f[i]:0;
 for(uint c=0;c<32;c++){
  float products[16];int largest=-1000;
  for(uint i=0;i<16;i++){float a=f[i],b=weights[c*16+i];products[i]=a*b;if(products[i]!=0){int e=int((bits(a)>>23)&255u)-126+int((bits(b)>>23)&255u)-126;largest=largest>e?largest:e;}}
  if(largest==-1000){prefix[p*32+c]=0;continue;}
  float scale=value(uint(27-largest+127)<<23);int sum=0;for(uint i=0;i<16;i++)sum+=int(products[i]*scale);
  prefix[p*32+c]=prefix_scaled_half(sum,largest-27);
 }
}

