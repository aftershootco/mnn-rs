#include "utils.h"
#include <memory.h>
#include <stdlib.h>
#ifdef __DEBUG
#include <cstdio>
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
TensorInfoArray *createTensorInfoArray(size_t count) {
  TensorInfoArray *array;
  array = (TensorInfoArray *)malloc(sizeof(TensorInfoArray));
  array->size = count;
  array->tensors = (TensorInfo *)malloc(count * sizeof(TensorInfo));
  return array;
}

void destroyTensorInfoArray(TensorInfoArray *array) {
  for (size_t i = 0; i < array->size; i++) {
    destroyCString(&array->tensors[i].name);
  }
  free(array->tensors);
  array->tensors = NULL;
  array->size = 0;
  free(array);
  array = NULL;
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

#ifdef __DISABLED
struct halide_type_t halide_type_to_halide_type_t(halide_type_c type) {
  // std::cout << sizeof(halide_type_of<float>()) << std::endl;
  // std::cout
  //     << "================halide_type_to_halide_type_t====================="
  //     << std::endl;
  auto htt = halide_type_t(type.code, type.bits, type.lanes);
  // code_bits_lanes("htt", &htt);
  return htt;
}

union TypeUnion {
  halide_type_t htt;
  uint64_t as_uint64;
  TypeUnion() {}
  ~TypeUnion() {}
};

uint64_t halide_type_t_from(halide_type_c type) {
  TypeUnion tu;
  tu.htt = halide_type_t(type.code, type.bits, type.lanes);

  return tu.as_uint64;
  // // return reinterpret_cast<uint64_t>(htt);
  // return reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(&htt));
}
#endif
