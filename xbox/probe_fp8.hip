extern "C" __attribute__((global)) void probe(float* out) { int p=__builtin_amdgcn_cvt_pk_fp8_f32(1.5f,0.f,0,false); out[__builtin_amdgcn_workitem_id_x()]=(float)__builtin_amdgcn_cvt_f32_fp8(p,0); }
