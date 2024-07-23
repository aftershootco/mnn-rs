#pragma once

#include <MNN/Tensor.hpp>
#include <mnn-sys/src/shared.rs>
#include <stdint.h>

namespace MNN {
// typedef struct TensorShape {
//   rust::Vec<int32_t> dims;
// } TensorShape;

// struct TensorHalideType {
// public:
//   static halide_type_t create_float() {
//     static halide_type_t t = halide_type_of<float>();
//     return t;
//   }
//   static halide_type_t create_double() {
//     static halide_type_t t = halide_type_of<double>();
//     return t;
//   };
// };

Tensor *glueTensorCreateDevice(TensorShape const &shape, HalideTypes &type,
                               Tensor::DimensionType dimType);
} // namespace MNN
