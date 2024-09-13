#include "tensor_c.h"
#include "MNN/Tensor.hpp"
#include "utils.h"
#include <cstdio>
#ifdef __DEBUG
#include <iostream>
void code_bits_lanes(const char *name, halide_type_t *type) {
  printf("====================================\n");
  printf("sizes: \n");
  std::cout << "code: " << sizeof(type->code) << std::endl;
  std::cout << "bits: " << sizeof(type->bits) << std::endl;
  std::cout << "lanes: " << sizeof(type->lanes) << std::endl;
  printf("%s: cbt %d %d %d\n", name, type->code, type->bits, type->lanes);
  printf("sizeof(%s): %lu\n", name, sizeof(*type));
  printf("====================================\n");
}
#endif
extern "C" {
Tensor *Tensor_create(int dimSize, DimensionType type) {
  return reinterpret_cast<Tensor *>(
      new MNN::Tensor(dimSize, static_cast<MNN::Tensor::DimensionType>(type)));
}
Tensor *Tensor_createFromTensor(const Tensor *tensor, DimensionType type,
                                int allocMemory) {
  return reinterpret_cast<Tensor *>(new MNN::Tensor(
      reinterpret_cast<const MNN::Tensor *>(tensor),
      static_cast<MNN::Tensor::DimensionType>(type), allocMemory));
}
void Tensor_destroy(Tensor *tensor) {
  delete reinterpret_cast<MNN::Tensor *>(tensor);
}
Tensor *Tensor_createDevice(const int *shape, size_t shapeSize,
                            halide_type_t typeCode, DimensionType dimType) {
  std::vector<int> shapeVec(shape, shape + shapeSize);
  return reinterpret_cast<Tensor *>(MNN::Tensor::createDevice(
      shapeVec, typeCode, static_cast<MNN::Tensor::DimensionType>(dimType)));
}
Tensor *Tensor_createWith(const int *shape, size_t shapeSize,
                          halide_type_t typeCode, void *data,
                          DimensionType dimType) {
  std::vector<int> shapeVec(shape, shape + shapeSize);
  auto mnn_tensor =
      MNN::Tensor::create(shapeVec, typeCode, data,
                          static_cast<MNN::Tensor::DimensionType>(dimType));
  return reinterpret_cast<Tensor *>(mnn_tensor);
}

int Tensor_copyFromHostTensor(Tensor *deviceTensor, const Tensor *hostTensor) {
  return reinterpret_cast<MNN::Tensor *>(deviceTensor)
      ->copyFromHostTensor(reinterpret_cast<const MNN::Tensor *>(hostTensor));
}
int Tensor_copyToHostTensor(const Tensor *deviceTensor, Tensor *hostTensor) {
  return reinterpret_cast<const MNN::Tensor *>(deviceTensor)
      ->copyToHostTensor(reinterpret_cast<MNN::Tensor *>(hostTensor));
}
Tensor *Tensor_createHostTensorFromDevice(const Tensor *deviceTensor,
                                          int copyData) {
  return reinterpret_cast<Tensor *>(MNN::Tensor::createHostTensorFromDevice(
      reinterpret_cast<const MNN::Tensor *>(deviceTensor), copyData));
}
const void *Tensor_host(const Tensor *tensor) {
  return reinterpret_cast<const MNN::Tensor *>(tensor)->host<void>();
}

void *Tensor_host_mut(Tensor *tensor) {
  return reinterpret_cast<MNN::Tensor *>(tensor)->host<void>();
}

uint64_t Tensor_deviceId(const Tensor *tensor) {
  return reinterpret_cast<const MNN::Tensor *>(tensor)->deviceId();
}

int Tensor_dimensions(const Tensor *tensor) {
  return reinterpret_cast<const MNN::Tensor *>(tensor)->dimensions();
}
/**
 * @brief get all dimensions' extent.
 * @return dimensions' extent.
 */
TensorShape Tensor_shape(const Tensor *tensor) {
  auto shapeVec = reinterpret_cast<const MNN::Tensor *>(tensor)->shape();
  TensorShape shape;
  shape.size = shapeVec.size();
  for (size_t i = 0; i < shapeVec.size(); i++) {
    shape.shape[i] = shapeVec[i];
  }
  return shape;
}

int Tensor_size(const Tensor *tensor) {
  return reinterpret_cast<const MNN::Tensor *>(tensor)->size();
}
size_t Tensor_usize(const Tensor *tensor) {
  return reinterpret_cast<const MNN::Tensor *>(tensor)->usize();
}
int Tensor_elementSize(const Tensor *tensor) {
  return reinterpret_cast<const MNN::Tensor *>(tensor)->elementSize();
}
int Tensor_width(const Tensor *tensor) {
  return reinterpret_cast<const MNN::Tensor *>(tensor)->width();
}
int Tensor_height(const Tensor *tensor) {
  return reinterpret_cast<const MNN::Tensor *>(tensor)->height();
}
int Tensor_channel(const Tensor *tensor) {
  return reinterpret_cast<const MNN::Tensor *>(tensor)->channel();
}
int Tensor_batch(const Tensor *tensor) {
  return reinterpret_cast<const MNN::Tensor *>(tensor)->batch();
}
int Tensor_stride(const Tensor *tensor, int index) {
  return reinterpret_cast<const MNN::Tensor *>(tensor)->stride(index);
}
int Tensor_length(const Tensor *tensor, int index) {
  return reinterpret_cast<const MNN::Tensor *>(tensor)->length(index);
}
void Tensor_setStride(Tensor *tensor, int index, int stride) {
  reinterpret_cast<MNN::Tensor *>(tensor)->setStride(index, stride);
}
void Tensor_setLength(Tensor *tensor, int index, int length) {
  reinterpret_cast<MNN::Tensor *>(tensor)->setLength(index, length);
}

int Tensor_getDeviceInfo(const Tensor *tensor, void *dst, int forwardType) {
  return reinterpret_cast<const MNN::Tensor *>(tensor)->getDeviceInfo(
      dst, forwardType);
}
void Tensor_print(const Tensor *tensor) {
  reinterpret_cast<const MNN::Tensor *>(tensor)->print();
}
void Tensor_printShape(const Tensor *tensor) {
  reinterpret_cast<const MNN::Tensor *>(tensor)->printShape();
}
void *Tensor_map(Tensor *tensor, MapType mtype, DimensionType dtype) {
  return reinterpret_cast<MNN::Tensor *>(tensor)->map(
      static_cast<MNN::Tensor::MapType>(mtype),
      static_cast<MNN::Tensor::DimensionType>(dtype));
}
void Tensor_unmap(Tensor *tensor, MapType mtype, DimensionType dtype,
                  void *mapPtr) {
  reinterpret_cast<MNN::Tensor *>(tensor)->unmap(
      static_cast<MNN::Tensor::MapType>(mtype),
      static_cast<MNN::Tensor::DimensionType>(dtype), mapPtr);
}
int Tensor_wait(Tensor *tensor, MapType mtype, int finish) {
  return reinterpret_cast<MNN::Tensor *>(tensor)->wait(
      static_cast<MNN::Tensor::MapType>(mtype), finish);
}
int Tensor_setDevicePtr(Tensor *tensor, const void *devicePtr, int memoryType) {
  return reinterpret_cast<MNN::Tensor *>(tensor)->setDevicePtr(devicePtr,
                                                               memoryType);
}

const halide_buffer_t *Tensor_buffer(const Tensor *tensor) {
  return &reinterpret_cast<const MNN::Tensor *>(tensor)->buffer();
}

halide_buffer_t *Tensor_buffer_mut(Tensor *tensor) {
  return &reinterpret_cast<MNN::Tensor *>(tensor)->buffer();
}
DimensionType Tensor_getDimensionType(const Tensor *tensor) {
  return static_cast<DimensionType>(
      reinterpret_cast<const MNN::Tensor *>(tensor)->getDimensionType());
}
halide_type_t Tensor_getType(const Tensor *tensor) {
  auto mnn_tensor = reinterpret_cast<const MNN::Tensor *>(tensor);
  return mnn_tensor->getType();
}

bool Tensor_isTypeOf(const Tensor *tensor, struct halide_type_t other) {
  auto my = Tensor_getType(tensor);
  auto ret = (my.code == other.code && my.bits == other.bits &&
              my.lanes == other.lanes);
  return ret;
}

Tensor *Tensor_clone(const Tensor *tensor) {
  auto mnn_tensor = reinterpret_cast<const MNN::Tensor *>(tensor);
  auto ret = MNN::Tensor::clone(mnn_tensor, true);
  return reinterpret_cast<Tensor *>(ret);
}

} // extern "C"
