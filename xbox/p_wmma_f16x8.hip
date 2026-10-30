// unsuffixed _w32 builtin, but with the 8-element _Float16 fragment the gfx12 sources use
// (deep_wmma.hip h8). If this fails, dropping "_gfx12" is NOT a drop-in rename.
typedef unsigned uint;
using f8 = float __attribute__((ext_vector_type(8)));
using h8 = _Float16 __attribute__((ext_vector_type(8)));
extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(32,32)))
void wmma_f16x8(float* out, const float* a, const float* b) {
  uint lane = __builtin_amdgcn_workitem_id_x();
  f8 acc{}; h8 x{}, y{};
  for (uint i = 0; i < 8; i++) { x[i] = (_Float16)a[lane*8+i]; y[i] = (_Float16)b[lane*8+i]; }
  acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(x, y, acc);
  out[lane*8] = acc[0];
}
