#ifndef TENSOR_UTILS_C_H
#define TENSOR_UTILS_C_H
#ifdef __cplusplus
extern "C" {
#endif
#include "Tensor_c.h"
typedef struct {
    int32_t offset;
    int32_t stride[3];
} View;
typedef struct {
    View src;
    View dst;
    int32_t size[3];
    Tensor* origin;
} Region;
typedef struct {
    int32_t left;
    int32_t right;
    int32_t bottom;
    int32_t top;
} Pad;
typedef struct {
    bool isDynamicSize;
    bool isIdenticalShape;
    uint32_t arraySize;
    int** elemShape;
    size_t elemShapeSize;
} TensorArrayAttr;
typedef struct {
    float scale;
    float zero;
    float min;
    float max;
} QuantAttr;
typedef enum {
    MEMORY_BACKEND = 0,
    MEMORY_HOST,
    MEMORY_VIRTUAL,
    MEMORY_OUTSIDE,
} MemoryType;
typedef enum {
    NORMAL,
    INPUT,
    OUTPUT,
    CONSTANT,
    TRAINABLE,
} Usage;
typedef struct {
    MNN_DATA_FORMAT dimensionFormat;
    union {
        int offset;
        void (*handleFreeFunction)(void*);
    } extra;
    MemoryType memoryType;
    int useCount;
    Usage usage;
    Region* regions;
    size_t regionsSize;
    halide_dimension_t dims[MNN_MAX_TENSOR_DIM];
    TensorArrayAttr* tensorArrayAttr;
    QuantAttr* quantAttr;
    DataType type;
    bool isMutable;
    int index;
    int group;
    int channel_pack_num;
    bool support_pack16;
    Pad mPads;
    uint32_t stageMask;
} NativeInsideDescribe;
NativeInsideDescribe* TensorUtils_getDescribe(const Tensor* tensor);
void TensorUtils_copyShape(const Tensor* source, Tensor* dest, int copyFormat, int copyRef);
void TensorUtils_setShape(Tensor* dest, const int* alldims, size_t alldimsSize);
void TensorUtils_setLinearLayout(Tensor* tensor);
int TensorUtils_compareTensors(const Tensor* compareTensor, const Tensor* toTensor, float tolerance, int overall, int printsError, int printsTensors);
void TensorUtils_setupTensorInfo(const Tensor* tensor, Tensor* wrapTensor, MNN_DATA_FORMAT mMidFormat);
Region TensorUtils_makeFullSlice(Tensor* input);
int TensorUtils_regionIsFull(Tensor* input);
int TensorUtils_isCopyRegion(const Region* region);
int TensorUtils_isTransposeRegion(const Region* region);
int TensorUtils_isTileRegion(const Region* region);
int TensorUtils_isDepthToSpaceRegions(const Tensor* output);
int TensorUtils_reshapeSlice(Region* slice, int outside, int inside, int axis);
void TensorUtils_adjustTensorForCompability(Tensor* t);
DimensionType TensorUtils_getDimType(const Tensor* t);
float* TensorUtils_getQuantInfo(const Tensor* t, size_t* size);
size_t TensorUtils_getRawSize(const Tensor* t);
void TensorUtils_setRasterInputs(Command* cmd);
int TensorUtils_refTensorContent(Tensor* dst, const Tensor* src);
int TensorUtils_getTensorChannelPack(const Tensor* tensor);
void TensorUtils_setTensorChannelPack(const Tensor* tensor, int pack);
void TensorUtils_setTensorSupportPack(const Tensor* tensor, int flag);
void TensorUtils_setTensorPad(const Tensor* tensor, int left, int right, int bottom, int top);
#ifdef __cplusplus
}
#endif
#endif // TENSOR_UTILS_C_H

