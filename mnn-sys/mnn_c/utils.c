#include "utils.h"
#include <memory.h>
#include <stdlib.h>
TensorInfoArray createTensorInfoArray(size_t count) {
  TensorInfoArray array;
  array.size = count;
  array.tensors = (TensorInfo *)malloc(count * sizeof(TensorInfo));
  return array;
}

void destroyTensorInfoArray(TensorInfoArray *array) {
  free(array->tensors);
  array->tensors = NULL;
  array->size = 0;
}

TensorInfo *getTensorInfoArray(TensorInfoArray const *array, size_t index) {
  if (index >= array->size) {
    return NULL;
  }
  return array->tensors + index;
}

CString createCString(const char *str, size_t max_size) {
  CString cstr;
  // Find out the size of the input
  size_t size = 0;
  while (str[size] != '\0' || size <= max_size) {
    size++;
  }
  cstr.size = size;
  cstr.data = (char *)malloc(size + 1);
  if (cstr.data) {
    memcpy((void *)cstr.data, str, size);
    cstr.data[size] = '\0';
  }
  return cstr;
}

void destroyCString(CString *cstr) {
  free(cstr->data);
  cstr->data = NULL;
  cstr->size = 0;
}
