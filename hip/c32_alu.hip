// gfx1103 ALU-first twin of c32_wmma.hip: identical kernel names, grid and work-group contract,
// but no matrix instruction at all (measured gfx1103 has no mai-insts, no fp8 conversion insts
// and no gfx12 WMMA). Every 16x16 tile at K=32 is carried by v_fma_f32 over LDS-staged values.
// Host ABI remains f32; numerics keep the E4M3 rounding the packed_fp8 loads performed (F()), which
// is exact in F16/F32, so only the accumulator order differs from the gfx12 WMMA path.
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
 uint h=((a+0xfffu+((a>>13)&1u))>>13)-0x1c000u;return s|(h>=0x7c00u?0x7c00u:h);}
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
using float8=float __attribute__((ext_vector_type(8)));
DEV float half_add_midpoint(float a,float b){float sum=a+b,vb=sum-a,error=(a-(sum-vb))+(b-vb);uint u=bits(sum),m=u&0x7fffffffu;
 if(m>=0x38800000u&&m<0x47800000u&&(m&0x1fffu)==0x1000u&&error!=0)sum=value(((error>0)==(sum>0))?u+1:u-1);return H(sum);}
DEV float exp_approx(float x){float a=clampf(H(x*.044921875f+1.30078125f),1.03125f,1.5693359375f);return from_half(((half_bits(a)<<5)+0x8000u)&65535u);}
#ifndef HIP_LDS_FENCE
#define HIP_LDS_FENCE 1 /* same LDS-scoped fence scope as c32_fused_ffn_attention.hip:90; here it also keeps the workgroup fence off the global address space, see Development/results/fence-scope-all-20260922 */
#endif
#if HIP_LDS_FENCE
#define WG_FENCE(k) __builtin_amdgcn_fence(k,"workgroup","local")
#else
#define WG_FENCE(k) __builtin_amdgcn_fence(k,"workgroup")
#endif
// A(row,k) = a[a0+row*astr+k*akstr], zero for row>=rowcap; B(col,k) = b[b0+col*bstr+k*bkstr].
// Bases already select the wave's 16 rows / 16 columns; K stride is in floats. Output slot is
// (row=group*8+e, col=lane&15) - the fragment layout c32_wmma.hip writes.
// Tiles are K-major, (k,slot) at k*16+slot: the compute loop then reads one broadcast address
// per half-wave for A and 16 contiguous addresses for B, so the 256 FMAs per lane run
// LDS-conflict-free; the cooperative stage (32 lanes x 16 elements each) pays the bank overlap instead.
DEV float8 alu_matrix32(const float*a,const float*b,uint a0,uint astr,uint akstr,uint rowcap,uint b0,uint bstr,uint bkstr){
 uint lane=__builtin_amdgcn_workitem_id_x(),rc=lane&15,group=lane>>4;
 __attribute__((shared)) float sa[16*32];
 __attribute__((shared)) float sb[32*16];
 #pragma unroll
 for(uint i=0;i<16;i++){uint row=lane>>1,k=(lane&1u)*16u+i;
  sa[k*16u+row]=row<rowcap?F(a[a0+row*astr+k*akstr]):0.f;
  sb[k*16u+row]=F(b[b0+row*bstr+k*bkstr]);}
 WG_FENCE(3);__builtin_amdgcn_s_barrier();WG_FENCE(2);
 float8 acc{};
 #pragma unroll
 for(uint k=0;k<32;k++){float bv=sb[k*16u+rc];
  #pragma unroll
  for(uint e=0;e<8;e++)acc[e]=__builtin_fmaf(sa[k*16u+group*8u+e],bv,acc[e]);}
 return acc;}
// One wave computes 16 tokens x 16 output channels. A row and B column use lane%16;
// each half-wave supplies eight K entries, and D rows are half-wave*8+element.
KERNEL __attribute__((amdgpu_flat_work_group_size(32,32)))
void c32_ffn_expand_wmma(const float* input,const float* weights,float* hidden,uint tokens){
 uint block=__builtin_amdgcn_workgroup_id_x(),first=(block/8)*16u,col=(block%8)*16u;
 if(first>=tokens||col>=128)return;
 uint lane=__builtin_amdgcn_workitem_id_x(),rc=lane&15,group=lane>>4;
 float8 sum=alu_matrix32(input,weights,first*32u,32u,1u,tokens-first,512u+col*32u,32u,1u);
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
  // Keep the residual outside the K32 sum: each chunk starts from zero.
  float8 sum=alu_matrix32(hidden,weights,first*128u+chunk*32u,128u,1u,tokens-first,4608u+col*128u+chunk*32u,128u,1u);
  #pragma unroll
  for(uint e=0;e<8;e++)residual[e]=H(residual[e]+sum[e]);
 }
 #pragma unroll
 for(uint e=0;e<8;e++){uint p=first+group*8+e;if(p<tokens)output[p*32+col+rc]=raw_output?residual[e]:F(residual[e]);}
}
KERNEL __attribute__((amdgpu_flat_work_group_size(32,32)))
void c32_attn_qkv_wmma(const float*input,const float*weights,float*qkv,uint tokens){
 uint block=__builtin_amdgcn_workgroup_id_x(),perpart=tokens/8,part=block/perpart,local=block%perpart,first=(local/2)*16,col=(local%2)*16;
 if(part>=3)return;uint lane=__builtin_amdgcn_workitem_id_x(),rc=lane&15,group=lane>>4;
 float8 sum=alu_matrix32(input,weights,first*32u,32u,1u,-1,part*1024u+col*32u,32u,1u);
 #pragma unroll
 for(uint e=0;e<8;e++){uint p=first+group*8+e;float v=H(sum[e]);qkv[(part*tokens+p)*32+col+rc]=part==2?F(v):v;}
}
KERNEL __attribute__((amdgpu_flat_work_group_size(32,32)))
void c32_attn_scores_wmma(const float*normalized,const float*weights,float*ex,uint windows){
 uint block=__builtin_amdgcn_workgroup_id_x(),w=block/16,qr=((block%16)/4)*16,kr=(block%4)*16;
 if(w>=windows)return;uint lane=__builtin_amdgcn_workitem_id_x(),rc=lane&15,group=lane>>4,tokens=windows*64;
 float8 sum=alu_matrix32(normalized,normalized,(w*64u+qr)*32u,32u,1u,-1,(tokens+w*64u+kr)*32u,32u,1u);
 #pragma unroll
 for(uint e=0;e<8;e++){uint q=qr+group*8+e,k=kr+rc;ex[(w*64+q)*64+k]=exp_approx(H(sum[e]+weights[4096+q*64+k]));}
}
KERNEL __attribute__((amdgpu_flat_work_group_size(32,32)))
void c32_attn_av_wmma(const float*prob,const float*qkv,float*av,uint windows){
 uint block=__builtin_amdgcn_workgroup_id_x(),w=block/8,qr=((block%8)/2)*16,col=(block%2)*16;
 if(w>=windows)return;uint lane=__builtin_amdgcn_workitem_id_x(),rc=lane&15,group=lane>>4,tokens=windows*64;float8 acc{};
 #pragma unroll
 for(uint chunk=0;chunk<2;chunk++){float8 sum=alu_matrix32(prob,qkv,(w*64u+qr)*64u+chunk*32u,64u,1u,-1,(2u*tokens+w*64u+chunk*32u)*32u+col,32u,32u);
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
 float8 sum=alu_matrix32(av,weights,first*32u,32u,1u,-1,3072u+col*32u,32u,1u);
 #pragma unroll
 for(uint e=0;e<8;e++){uint p=first+group*8+e;float result=half_add_midpoint(sum[e],H(input[p*32+col+rc]*weights[8193+col+rc]));output[p*32+col+rc]=raw_output?result:F(result);}
}
