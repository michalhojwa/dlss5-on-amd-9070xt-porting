typedef unsigned uint;
using f32v = float __attribute__((ext_vector_type(32)));
using p4 = __fp16 __attribute__((ext_vector_type(4)));
extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(32,32)))
void mfma_32x32x8(float* out, const float* a, const float* b) {
  uint lane = __builtin_amdgcn_workitem_id_x();
  f32v acc{}; p4 x{}, y{};
  for (uint i = 0; i < 4; i++) { x[i] = (__fp16)a[lane*4+i]; y[i] = (__fp16)b[lane*4+i]; }
  acc = __builtin_amdgcn_mfma_f32_32x32x8f16(x, y, acc, 0, 0, 0);
  out[lane*32] = acc[0];
}
