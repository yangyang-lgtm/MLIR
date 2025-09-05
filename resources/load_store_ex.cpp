#include "utils.h"

void add_ex_kernel(float* a, float* b, float* c, int32_t len);

int main() {
  Tensor<float> v1({1024}), v2({1024}), v3({1024}), v4({1024});

  init(v1);
  init(v2);
  init(v3);
  init(v4);

  for (auto i = 0; i < v3.impl->numel;++i){
    v3[i] = v1[i] - v2[i];
  }

  add_ex_kernel(v1.impl->get(), v2.impl->get(), v4.impl->get(), 1024);

  std::cout << get_diff_num(v3, v4) << " / 1024" << std::endl;
}
