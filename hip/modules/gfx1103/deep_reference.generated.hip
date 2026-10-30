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
// Split FFN: per-output kernels with explicit mixed/hidden scratch.
DEV float activation(float a){float g=clampf(a,-4.f,4.f);return F(H(a*H(g*H(absf(g)*(-.055908203125f)+.447265625f)+.89453125f)));}
KERNEL void split_mix(const float* input,const float* weights,float* mixed,uint tokens){uint i=lane();if(i>=tokens*512u)return;uint p=i/512,row=i%512;float a=0;for(uint k=0;k<512;k+=32){float s=0;for(uint j=0;j<32;j++)s+=input[p*512+k+j]*weights[row*512+k+j];a=H(a+s);}mixed[i]=F(a);}
KERNEL void split_expand(const float* mixed,const float* weights,float* hidden,uint tokens){uint i=lane();if(i>=tokens*2048u)return;uint p=i/2048,g=(i%2048)/256,row=i%256;float a=0;for(uint k=0;k<64;k+=32){float s=0;for(uint j=0;j<32;j++)s+=mixed[p*512+g*64+k+j]*weights[262144+g*16384+row*64+k+j];a=H(a+s);}hidden[i]=activation(a);}
KERNEL void split_contract(const float* hidden,const float* weights,float* output,uint tokens){uint i=lane();if(i>=tokens*512u)return;uint p=i/512,g=(i%512)/64,row=i%64;float a=0;for(uint k=0;k<256;k+=32){float s=0;for(uint j=0;j<32;j++)s+=hidden[p*2048+g*256+k+j]*weights[393216+g*16384+row*256+k+j];a=H(a+s);}output[i]=F(a);}
KERNEL void split_projection(const float* input,const float* weights,const float* residual,float* output,uint tokens){uint i=lane();if(i>=tokens*512u)return;uint p=i/512,row=i%512;float a=H(residual[i]*weights[262144+row]);for(uint k=0;k<512;k+=32){float s=0;for(uint j=0;j<32;j++)s+=input[p*512+k+j]*weights[row*512+k+j];a=H(a+s);}output[i]=F(a);}
KERNEL void vit_qkv_project(const float* input,const float* weights,float* output,uint tokens){uint i=lane();if(i>=tokens*3072u)return;uint part=i/(tokens*1024),token=(i/1024)%tokens,row=i%1024;float total=0;for(uint group=0;group<2;group++){float a=0;for(uint k=group*512;k<(group+1)*512;k+=32){float s=0;for(uint j=0;j<32;j++)s+=input[token*1024+k+j]*weights[part*1048576+row*1024+k+j];a=H(a+s);}total=group?H(total+a):a;}output[i]=total;}
DEV uint tensor_channel(uint c){return ((c&1)<<1)|((c&2)>>1)|((c&4)<<2)|((c&8)>>1)|((c&16)>>1);}
KERNEL void vit_qkv_normalize(const float* input,const float* weights,float* output,uint tokens){uint i=lane();if(i>=tokens*96u)return;uint part=i/(tokens*32),token=(i/32)%tokens,head=i%32,base=(part*tokens+token)*1024+head*32;if(part==2){for(uint c=0;c<32;c++)output[base+c]=F(input[base+c]);return;}float v[32],s[16];for(uint c=0;c<32;c++)v[tensor_channel(c)]=input[base+c];for(uint c=0;c<16;c++)s[c]=half_square_pair(v[c],v[c+16]);for(uint c=0;c<8;c++)s[c]=H(s[c*2]+s[c*2+1]);for(uint w=4;w;w/=2)for(uint c=0;c<w;c++)s[c]=H(s[c]+s[c+w]);float inverse=H(__builtin_amdgcn_rsqf(maxf(s[0],6.198883056640625e-5f)));for(uint c=0;c<32;c++){float a=H(input[base+c]*inverse);if(part==0)a=H(H(a*5.65625f)*H(weights[3145728+head]));output[base+c]=F(a);}}
// Scores [query][head32][key]. Attention tokens must be a multiple of 64.
KERNEL void vit_attention_scores(const float* qkv,float* ex,uint tokens){uint i=lane();if(i>=tokens*32u*tokens)return;uint key=i%tokens,head=(i/tokens)%32,query=i/(tokens*32);float score=0;for(uint c=0;c<32;c++)score+=qkv[query*1024+head*32+c]*qkv[(tokens+key)*1024+head*32+c];float affine=clampf(H(H(score)*from_half(0x2dbb)+1.708984375f),1.439453125f,1.9775390625f);ex[i]=from_half(((half_bits(affine)<<4)+0x4000u)&65535u);}
DEV uint key_index(uint c){return ((c&16)>>4)|((c&1)<<1)|((c&2)<<1)|(c&8)|((c&4)<<2)|(c&32);}
KERNEL void vit_attention_inverse(const float* ex,float* inverse,uint tokens){uint i=lane();if(i>=tokens*32u)return;uint row=i*tokens;float denominator=0;for(uint chunk=0;chunk<tokens;chunk+=64){float sums[2];for(uint parity=0;parity<2;parity++){float accumulated=0;for(uint group=0;group<4;group++){uint base=parity+(group&1)*2+(group>>1)*8;float a=H(ex[row+chunk+key_index(base)]+ex[row+chunk+key_index(base+16)]);a=H(a+H(ex[row+chunk+key_index(base+4)]+ex[row+chunk+key_index(base+20)]));a=H(a+H(ex[row+chunk+key_index(base+32)]+ex[row+chunk+key_index(base+48)]));a=H(a+H(ex[row+chunk+key_index(base+36)]+ex[row+chunk+key_index(base+52)]));accumulated=group?H(accumulated+a):a;}sums[parity]=accumulated;}denominator=H(denominator+H(sums[0]+sums[1]));}inverse[i]=H(1.f/denominator);}
KERNEL void vit_attention_av(const float* qkv,const float* ex,const float* inverse,float* output,uint tokens){uint i=lane();if(i>=tokens*1024u)return;uint query=i/1024,head=(i%1024)/32,c=i%32,row=(query*32+head)*tokens;float accumulated=0;for(uint part=0;part<tokens/32;part++){float a=0;for(uint j=0;j<32;j++){uint key=part*32+j;a+=F(ex[row+key])*qkv[(2*tokens+key)*1024+head*32+c];}accumulated=H(accumulated+a);}output[i]=F(H(accumulated*inverse[query*32+head]));}
KERNEL void vit_expand(const float* input,const float* weights,float* output,uint tokens,uint input_channels,uint output_channels){uint i=lane();if(i>=tokens*output_channels)return;uint token=i/output_channels,row=i%output_channels;float a=0;for(uint k=0;k<input_channels;k+=32){float s=0;for(uint j=0;j<32;j++)s+=input[token*input_channels+k+j]*weights[row*input_channels+k+j];a=H(a+s);}output[i]=activation(a);}
KERNEL void vit_project(const float* input,const float* weights,const float* residual,float* output,uint tokens,uint input_channels,uint output_channels){uint i=lane();if(i>=tokens*output_channels)return;uint token=i/output_channels,row=i%output_channels;float total=0;for(uint part=0;part<4;part++){float a=part==0?H(residual[i]*weights[input_channels*output_channels+row]):0;for(uint k=part*(input_channels/4);k<(part+1)*(input_channels/4);k+=32){float s=0;for(uint j=0;j<32;j++)s+=input[token*input_channels+k+j]*weights[row*input_channels+k+j];a=H(a+s);}total=part==0?a:H(total+a);}output[i]=F(total);}
// Project once per source pixel/channel then emit its clipped nearest 2x footprint.
KERNEL void decoder_project2x(const float* input,const float* weights,const float* residual,float* output,uint input_width,uint input_height,uint output_width,uint output_height,uint input_channels,uint output_channels){uint i=lane();if(i>=input_width*input_height*output_channels)return;uint token=i/output_channels,row=i%output_channels,partitions=input_channels==1024?4:1;float total=0;for(uint part=0;part<partitions;part++){float a=0;for(uint k=part*(input_channels/partitions);k<(part+1)*(input_channels/partitions);k+=32){float s=0;for(uint j=0;j<32;j++)s+=input[token*input_channels+k+j]*weights[row*input_channels+k+j];a=H(a+s);}total=part==0?a:H(total+a);}for(uint dy=0;dy<2;dy++)for(uint dx=0;dx<2;dx++){uint x=token%input_width*2+dx,y=token/input_width*2+dy;if(x>=output_width||y>=output_height)continue;uint idx=(y*output_width+x)*output_channels+row;float merged=H(total+F(residual[idx])*weights[input_channels*output_channels+row]);output[idx]=output_channels==32?merged:F(merged);}}
KERNEL void vit_gather(const float* input,const uint* indices,float* output,uint count){uint i=lane();if(i<count)output[i]=input[indices[i]];}
// Production FP8 fast path has a different accumulation contract from legacy above.
// Scalar sum order may differ from hardware WMMA; no inserted half rounding.
KERNEL void vit_attention_scores_fast(const float* qkv,float* ex,uint tokens){uint i=lane();if(i>=tokens*32u*tokens)return;uint key=i%tokens,head=(i/tokens)%32,query=i/(tokens*32);float score=0;for(uint c=0;c<32;c++)score+=qkv[query*1024+head*32+c]*qkv[(tokens+key)*1024+head*32+c];float affine=clampf(score*from_half(0x2dbb)+1.708984375f,1.439453125f,1.9775390625f);ex[i]=from_half(((half_bits(affine)<<4)+0x4000u)&65535u);}
KERNEL void vit_attention_inverse_fast(const float* ex,float* inverse,uint tokens){uint i=lane();if(i>=tokens*32u)return;float denominator=0;for(uint key=0;key<tokens;key++)denominator+=ex[i*tokens+key];inverse[i]=1.f/denominator;}
KERNEL void vit_attention_av_fast(const float* qkv,const float* ex,const float* inverse,float* output,uint tokens){uint i=lane();if(i>=tokens*1024u)return;uint query=i/1024,head=(i%1024)/32,c=i%32,row=(query*32+head)*tokens;float accumulated=0;for(uint key=0;key<tokens;key++)accumulated+=F(ex[row+key])*qkv[(2*tokens+key)*1024+head*32+c];output[i]=F(H(accumulated*inverse[query*32+head]));}

