#ifndef UTILS_H
#define UTILS_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
// This must always be
typedef struct {
  // Name of the tensor
  const char *name;
  // Points to a raw tensor object
  void *tensor;
} TensorInfo;

typedef struct {
  TensorInfo *tensors;
  size_t size;
} TensorInfoArray;

TensorInfoArray createTensorInfoArray(size_t count);
void destroyTensorInfoArray(TensorInfoArray *array);

#ifdef __cplusplus
}
#endif

#endif // UTILS_H
