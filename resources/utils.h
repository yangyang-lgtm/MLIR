
#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <chrono>

#define min(a, b) ((a) > (b)) ? (b) : (a)

template<typename T>
struct TensorImpl{
  static int64_t compute_numel(const std::vector<int64_t>& shape){
    if (shape.empty()){
      return 0;
    }
    int64_t numel = 1;
    for (const auto i : shape){ numel *= i; }
    return numel;
  }

  static T* memory_alloc(int64_t numel){
    if (numel == 0){ return nullptr; }
    return (T*)malloc(numel * sizeof(T));
  }

  TensorImpl(const std::vector<int64_t>& shape) : shape(shape), numel(compute_numel(shape)){}

  T *get(){
    if (!data){
      data = memory_alloc(numel);
    }
    return data;
  }

  T& get(int64_t index){
    return get()[index];
  }

  const T &get(int64_t index) const {
    return get()[index];
  }

  ~TensorImpl(){
    if (data){
      free(data);
    }
  }

  std::vector<int64_t> shape;
  int64_t numel;
  T* data{nullptr};
};

template<typename T>
struct Tensor{
  explicit Tensor(const std::vector<int64_t>& shape) : impl(std::make_shared<TensorImpl<T>>(shape)){}

  T &operator[](int64_t index){
    return impl->get(index);
  }

  const T &operator[](int64_t index) const {
    return impl->get(index);
  }

  std::shared_ptr<TensorImpl<T>> impl;
};

template<typename T, typename Len>
void copy(const T* from, T* to, const Len len){
  for (int i = 0; i < len; ++i){
    to[i] = from[i];
  }
}

template <typename T>
void load(Tensor<T>& dst, const T* src, int32_t len){
  copy(src, dst.impl->get(), len);
}

template <typename T>
void store(T* dst, const Tensor<T>& src, int32_t len){
  copy(src.impl->get(), dst, len);
}

template<typename T>
Tensor<T> operator+(const Tensor<T>& a, const Tensor<T>& b){
  Tensor<T> out(a.impl->shape);
  for (auto i = 0; i < out.impl->numel; ++i){
    out[i] = a[i] + b[i];
  }
  return out;
}

template<typename T>
Tensor<T> operator-(const Tensor<T>& a, const Tensor<T>& b){
  Tensor<T> out(a.impl->shape);
  for (auto i = 0; i < out.impl->numel; ++i){
    out[i] = a[i] - b[i];
  }
  return out;
}

template <typename T>
void init(Tensor<T>& in){
  static std::mt19937 generator(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_int_distribution<int> distribution(-10086, 10086);
  for (auto i = 0; i < in.impl->numel; ++i){
    in[i] = (T)distribution(generator) / 10000;
  }
}

template <typename T>
int64_t get_diff_num(const Tensor<T>& a, const Tensor<T>& b){
  if (a.impl->numel != b.impl->numel){
    return false;
  }
  double diff_max = 1e-5;
  if constexpr (std::is_integral_v<T>){
    diff_max = 0;
  }

  int64_t diff_num = 0;
  for (auto i = 0; i < a.impl->numel; ++i){
    diff_num += std::abs(a[i] - b[i]) > diff_max;
  }
  return diff_num;
}
