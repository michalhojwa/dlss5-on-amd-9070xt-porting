// Can a function-like macro shadow an amdgcn builtin name at every call site?
// If this PASSES for gfx1103, ONE prepended shim in build-modules.ps1 can cover
// every fp8-convert call site without touching the .hip sources.
typedef unsigned uint;
#define DEV __attribute__((device)) __attribute__((always_inline))
DEV float qwen_F(float x){uint b=0;uint a=b&0x7fffffffu;return (float)a;}
DEV int qwen_pk(float a,float b,int c,bool d){return (int)(a+b+c+d);}
DEV float qwen_up(int p,int i){return (float)p*(1.0f+(float)i);}
#define __builtin_amdgcn_cvt_pk_fp8_f32(a,b,c,d) qwen_pk(a,b,c,d)
#define __builtin_amdgcn_cvt_f32_fp8(p,i) qwen_up(p,i)
extern "C" __attribute__((global)) void shim(int* out){
  int packed=__builtin_amdgcn_cvt_pk_fp8_f32(1.5f,0.f,0,false);
  out[__builtin_amdgcn_workitem_id_x()]=packed+(int)__builtin_amdgcn_cvt_f32_fp8(packed,0);
}
