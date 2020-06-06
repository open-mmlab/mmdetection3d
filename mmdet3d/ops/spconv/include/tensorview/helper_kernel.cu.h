#pragma once
// from tensorflow
namespace tv {
namespace detail {

template <typename T>
class KernelLoop {
  struct Iterator {
    __forceinline__ __device__ Iterator(T index, T delta)
        : index_(index), delta_(delta) {}
    __forceinline__ __device__ T operator*() const { return index_; }
    __forceinline__ __device__ Iterator &operator++() {
      index_ += delta_;
      return *this;
    }
    __forceinline__ __device__ bool operator!=(const Iterator &other) const {
      bool greater = index_ > other.index_;
      bool less = index_ < other.index_;
      // Anything past an end iterator (delta_ == 0) is equal.
      // In range-based for loops, this optimizes to 'return less'.
      if (!other.delta_) {
        return less;
      }
      if (!delta_) {
        return greater;
      }
      return less || greater;
    }

   private:
    T index_;
    const T delta_;
  };

 public:
  __forceinline__ __device__ KernelLoop(T begin, T delta, T end)
      : begin_(begin), delta_(delta), end_(end) {}

  __forceinline__ __device__ Iterator begin() const {
    return Iterator{begin_, delta_};
  }
  __forceinline__ __device__ Iterator end() const { return Iterator{end_, 0}; }

 private:
  T begin_;
  T delta_;
  T end_;
};

}  // namespace detail
template <typename T, int NumILP = 1>
__forceinline__ __device__ detail::KernelLoop<T> KernelLoopX(T count) {
  return detail::KernelLoop<T>(blockIdx.x * blockDim.x + threadIdx.x,
                               gridDim.x * blockDim.x * NumILP, count);
}

// Helper to visit indices in the range 0 <= i < count using the y-coordinate.
// Usage: for(int i : KernelLoopY(count)) { visit(i); }
template <typename T, int NumILP = 1>
__forceinline__ __device__ detail::KernelLoop<T> KernelLoopY(T count) {
  return detail::KernelLoop<T>(blockIdx.y * blockDim.y + threadIdx.y,
                               gridDim.y * blockDim.y * NumILP, count);
}

// Helper to visit indices in the range 0 <= i < count using the z-coordinate.
// Usage: for(int i : KernelLoopZ(count)) { visit(i); }
template <typename T, int NumILP = 1>
__forceinline__ __device__ detail::KernelLoop<T> KernelLoopZ(T count) {
  return detail::KernelLoop<T>(blockIdx.z * blockDim.z + threadIdx.z,
                               gridDim.z * blockDim.z * NumILP, count);
}

}  // namespace tv
