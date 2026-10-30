typedef unsigned uint;
using f8 = float __attribute__((ext_vector_type(8)));
using p16 = __fp16 __attribute__((ext_vector_type(16)));
extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(32,32)))
void wmma_16x16x16_w32(float* out, const float* a, const float* b) {
  uint lane = __builtin_amdgcn_workitem_id_x();
  f8 acc{}; p16 x{}, y{};
  for (uint i = 0; i < 16; i++) { x[i] = (__fp16)a[lane*16+i]; y[i] = (__fp16)b[lane*16+i]; }
  acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(x, y, acc);
  out[lane*8] = acc[0];
}
