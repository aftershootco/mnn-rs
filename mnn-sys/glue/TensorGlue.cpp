#include "TensorGlue.hpp"
namespace MNN {
Tensor *glueTensorCreateDevice(TensorShape const &shape, HalideTypes &type,
                               Tensor::DimensionType dimType) {

  halide_type_t h_type;
  switch (type) {
  case HalideTypes::halide_float:
    h_type = halide_type_of<float>();
    break;
  case HalideTypes::halide_double:
    h_type = halide_type_of<double>();
    break;
  case HalideTypes::halide_bool:
    h_type = halide_type_of<bool>();
    break;
  case HalideTypes::halide_uint8_t:
    h_type = halide_type_of<uint8_t>();
    break;
  case HalideTypes::halide_uint16_t:
    h_type = halide_type_of<uint16_t>();
    break;
  case HalideTypes::halide_uint32_t:
    h_type = halide_type_of<uint32_t>();
    break;
  case HalideTypes::halide_uint64_t:
    h_type = halide_type_of<uint64_t>();
    break;
  case HalideTypes::halide_int8_t:
    h_type = halide_type_of<int8_t>();
    break;
  case HalideTypes::halide_int16_t:
    h_type = halide_type_of<int16_t>();
    break;
  case HalideTypes::halide_int32_t:
    h_type = halide_type_of<int32_t>();
    break;
  case HalideTypes::halide_int64_t:
    h_type = halide_type_of<int64_t>();
    break;
  };
  return Tensor::createDevice(
      std::vector<int>(shape.dims.begin(), shape.dims.end()), h_type, dimType);
}
} // namespace MNN
