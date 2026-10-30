extern "C" __attribute__((global)) void probe(float* out) { out[__builtin_amdgcn_workitem_id_x()] = 1.0f; }
