extern "C" __attribute__((global)) void wave_probe(float* out) { out[__builtin_amdgcn_workitem_id_x()] = (float)__builtin_amdgcn_wave_width(); }
