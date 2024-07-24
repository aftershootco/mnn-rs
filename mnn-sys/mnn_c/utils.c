#include <stdlib.h>
#include "utils.h"
TensorInfoArray createTensorInfoArray(size_t count) {
    TensorInfoArray array;
    array.count = count;
    array.tensors = (TensorInfo*)malloc(count * sizeof(TensorInfo));
    return array;
}
  
void destroyTensorInfoArray(TensorInfoArray *array) {
    free(array->tensors);
    array->tensors = NULL;
    array->count = 0;
}
