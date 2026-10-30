typedef unsigned uint;
using h4 = __fp16 __attribute__((ext_vector_type(4)));
using f4 = float __attribute__((ext_vector_type(4)));
extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(32,32)))
void alu_fma(float* out, const float* a, const float* b, const float* c) {
  uint lane = __builtin_amdgcn_workitem_id_x();
  h4 x{}, y{}, z{}; f4 u{}, v{}, w{};
  for (uint i = 0; i < 4; i++) {
    x[i] = (__fp16)a[lane*4+i]; y[i] = (__fp16)b[lane*4+i]; z[i] = (__fp16)c[lane*4+i];
    u[i] = a[lane*4+i]; v[i] = b[lane*4+i]; w[i] = c[lane*4+i];
  }
  h4 r = x*y + z; f4 s = u*v + w;
  for (uint i = 0; i < 4; i++) out[lane*8+i] = (float)r[i] + s[i];
}
