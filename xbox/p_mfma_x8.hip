typedef unsigned uint;
using f8 = float __attribute__((ext_vector_type(8)));
using p8 = __fp16 __attribute__((ext_vector_type(8)));
extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(32,32)))
void mfma_x8(float* out, const float* a, const float* b) {
  uint lane = __builtin_amdgcn_workitem_id_x();
  f8 acc{}; p8 x{}, y{};
  for (uint i = 0; i < 8; i++) { x[i] = (__fp16)a[lane*8+i]; y[i] = (__fp16)b[lane*8+i]; }
  acc = __builtin_amdgcn_mfma_f32_16x16x16f16(x, y, acc, 0, 0, 0);
  out[lane*8] = acc[0];
}
