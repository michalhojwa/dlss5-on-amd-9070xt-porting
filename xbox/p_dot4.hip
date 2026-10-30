typedef unsigned uint;
using i4 = int __attribute__((ext_vector_type(4)));
using u4 = unsigned char __attribute__((ext_vector_type(4)));
extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(32,32)))
void dot4_probe(int* out, const int* a, const int* b) {
  uint lane = __builtin_amdgcn_workitem_id_x();
  i4 x{}, y{}; u4 sx{}, sy{};
  for (uint i = 0; i < 4; i++) { x[i] = a[lane*4+i]; y[i] = b[lane*4+i]; sx[i]=(u4)a[lane*4+i]; sy[i]=(u4)b[lane*4+i]; }
  out[lane] = __builtin_amdgcn_dot4a(x, y) + __builtin_amdgcn_usdot4a(sx, sy);
}
