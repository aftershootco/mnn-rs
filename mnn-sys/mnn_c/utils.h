#ifndef UTILS_H
#define UTILS_H
#include <MNN/HalideRuntime.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  char *data;
  size_t size;
} CString;
CString createCString(const char *data, size_t size);
void destroyCString(CString *string);
// This must always be
typedef struct {
  // Name of the tensor
  CString name;
  // Points to a raw tensor object
  void *tensor;
} TensorInfo;

typedef struct {
  TensorInfo *tensors;
  size_t size;
} TensorInfoArray;

TensorInfoArray createTensorInfoArray(size_t count);
void destroyTensorInfoArray(TensorInfoArray *array);
TensorInfo *getTensorInfoArray(TensorInfoArray const *array, size_t index);
typedef struct {
  halide_type_code_t code;
  uint8_t bits;
  uint16_t lanes;
} halide_type_c;

#ifdef __cplusplus
}
#endif

#endif // UTILS_H
