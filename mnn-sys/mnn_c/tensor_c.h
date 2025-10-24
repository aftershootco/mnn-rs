#ifndef TENSOR_C_H
#define TENSOR_C_H
#include "utils.h"
#include <MNN/HalideRuntime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
typedef struct Tensor Tensor;
typedef struct {
  int shape[4];
  size_t size;
} TensorShape;
typedef enum { TENSORFLOW, CAFFE, CAFFE_C4 } DimensionType;
typedef enum { HANDLE_NONE = 0, HANDLE_STRING = 1 } HandleDataType;
typedef enum { MAP_TENSOR_WRITE = 0, MAP_TENSOR_READ = 1 } MapType;
Tensor *Tensor_create(int dimSize, DimensionType type);
Tensor *Tensor_createFromTensor(const Tensor *tensor, DimensionType type,
                                int allocMemory);
void Tensor_destroy(Tensor *tensor);
Tensor *Tensor_createDevice(const int *shape, size_t shapeSize,
                            struct halide_type_t typeCode,
                            DimensionType dimType);
Tensor *Tensor_createWith(const int *shape, size_t shapeSize,
                          struct halide_type_t typeCode, void *data,
                          DimensionType dimType);
int Tensor_copyFromHostTensor(Tensor *deviceTensor, const Tensor *hostTensor);
int Tensor_copyToHostTensor(const Tensor *deviceTensor, Tensor *hostTensor);
Tensor *Tensor_createHostTensorFromDevice(const Tensor *deviceTensor,
                                          int copyData);
DimensionType Tensor_getDimensionType(const Tensor *tensor);
const halide_buffer_t *Tensor_buffer(const Tensor *tensor);
halide_buffer_t *Tensor_buffer_mut(Tensor *tensor);
const void *Tensor_host(const Tensor *tensor);
void *Tensor_host_mut(Tensor *tensor);
uint64_t Tensor_deviceId(const Tensor *tensor);
int Tensor_dimensions(const Tensor *tensor);
TensorShape Tensor_shape(const Tensor *tensor);
int Tensor_size(const Tensor *tensor);
size_t Tensor_usize(const Tensor *tensor);
int Tensor_elementSize(const Tensor *tensor);
int Tensor_width(const Tensor *tensor);
int Tensor_height(const Tensor *tensor);
int Tensor_channel(const Tensor *tensor);
int Tensor_batch(const Tensor *tensor);
int Tensor_stride(const Tensor *tensor, int index);
int Tensor_length(const Tensor *tensor, int index);
void Tensor_setStride(Tensor *tensor, int index, int stride);
void Tensor_setLength(Tensor *tensor, int index, int length);
int Tensor_getDeviceInfo(const Tensor *tensor, void *dst, int forwardType);
void Tensor_print(const Tensor *tensor);
void Tensor_printShape(const Tensor *tensor);
void *Tensor_map(Tensor *tensor, MapType mtype, DimensionType dtype);
void Tensor_unmap(Tensor *tensor, MapType mtype, DimensionType dtype,
                  void *mapPtr);
Tensor *Tensor_clone(const Tensor *tensor);
int Tensor_wait(Tensor *tensor, MapType mtype, int finish);
int Tensor_setDevicePtr(Tensor *tensor, const void *devicePtr, int memoryType);
struct halide_type_t Tensor_getType(const Tensor *tensor);
bool Tensor_isTypeOf(const Tensor *tensor, struct halide_type_t type);
#ifdef __cplusplus
}
#endif
#endif // TENSOR_C_H
