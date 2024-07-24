// #include "Tensor_c.h"
// #include "MNN/Tensor.hpp"
// extern "C" {
// Tensor* Tensor_create(int dimSize, DimensionType type) {
//     return new MNN::Tensor(dimSize, static_cast<MNN::Tensor::DimensionType>(type));
// }
// Tensor* Tensor_createFromTensor(const Tensor* tensor, DimensionType type, int allocMemory) {
//     return new MNN::Tensor(tensor, static_cast<MNN::Tensor::DimensionType>(type), allocMemory);
// }
// void Tensor_destroy(Tensor* tensor) {
//     delete tensor;
// }
// Tensor* Tensor_createDevice(const int* shape, size_t shapeSize, halide_type_t typeCode, DimensionType dimType) {
//     std::vector<int> shapeVec(shape, shape + shapeSize);
//     return MNN::Tensor::createDevice(shapeVec, typeCode, static_cast<MNN::Tensor::DimensionType>(dimType));
// }
// Tensor* Tensor_createWith(const int* shape, size_t shapeSize, halide_type_t typeCode, void* data, DimensionType dimType) {
//     std::vector<int> shapeVec(shape, shape + shapeSize);
//     return MNN::Tensor::create(shapeVec, typeCode, data, static_cast<MNN::Tensor::DimensionType>(dimType));
// }
// int Tensor_copyFromHostTensor(Tensor* deviceTensor, const Tensor* hostTensor) {
//     return deviceTensor->copyFromHostTensor(hostTensor);
// }
// int Tensor_copyToHostTensor(const Tensor* deviceTensor, Tensor* hostTensor) {
//     return deviceTensor->copyToHostTensor(hostTensor);
// }
// Tensor* Tensor_createHostTensorFromDevice(const Tensor* deviceTensor, int copyData) {
//     return MNN::Tensor::createHostTensorFromDevice(deviceTensor, copyData);
// }
// const void* Tensor_host(const Tensor* tensor) {
//     return tensor->host<void>();
// }
// uint64_t Tensor_deviceId(const Tensor* tensor) {
//     return tensor->deviceId();
// }
// int Tensor_dimensions(const Tensor* tensor) {
//     return tensor->dimensions();
// }
// void Tensor_shape(const Tensor* tensor, int* shape) {
//     auto shapeVec = tensor->shape();
//     std::copy(shapeVec.begin(), shapeVec.end(), shape);
// }
// int Tensor_size(const Tensor* tensor) {
//     return tensor->size();
// }
// size_t Tensor_usize(const Tensor* tensor) {
//     return tensor->usize();
// }
// int Tensor_elementSize(const Tensor* tensor) {
//     return tensor->elementSize();
// }
// int Tensor_width(const Tensor* tensor) {
//     return tensor->width();
// }
// int Tensor_height(const Tensor* tensor) {
//     return tensor->height();
// }
// int Tensor_channel(const Tensor* tensor) {
//     return tensor->channel();
// }
// int Tensor_batch(const Tensor* tensor) {
//     return tensor->batch();
// }
// int Tensor_stride(const Tensor* tensor, int index) {
//     return tensor->stride(index);
// }
// int Tensor_length(const Tensor* tensor, int index) {
//     return tensor->length(index);
// }
// void Tensor_setStride(Tensor* tensor, int index, int stride) {
//     tensor->setStride(index, stride);
// }
// void Tensor_setLength(Tensor* tensor, int index, int length) {
//     tensor->setLength(index, length);
// }
// int Tensor_getDeviceInfo(const Tensor* tensor, void* dst, int forwardType) {
//     return tensor->getDeviceInfo(dst, forwardType);
// }
// void Tensor_print(const Tensor* tensor) {
//     tensor->print();
// }
// void Tensor_printShape(const Tensor* tensor) {
//     tensor->printShape();
// }
// void* Tensor_map(Tensor* tensor, MapType mtype, DimensionType dtype) {
//     return tensor->map(static_cast<MNN::Tensor::MapType>(mtype), static_cast<MNN::Tensor::DimensionType>(dtype));
// }
// void Tensor_unmap(Tensor* tensor, MapType mtype, DimensionType dtype, void* mapPtr) {
//     tensor->unmap(static_cast<MNN::Tensor::MapType>(mtype), static_cast<MNN::Tensor::DimensionType>(dtype), mapPtr);
// }
// int Tensor_wait(Tensor* tensor, MapType mtype, int finish) {
//     return tensor->wait(static_cast<MNN::Tensor::MapType>(mtype), finish);
// }
// int Tensor_setDevicePtr(Tensor* tensor, const void* devicePtr, int memoryType) {
//     return tensor->setDevicePtr(devicePtr, memoryType);
// }
// } // extern "C"
#include "Tensor_c.h"
#include "MNN/Tensor.hpp"
extern "C" {
Tensor* Tensor_create(int dimSize, DimensionType type) {
    return reinterpret_cast<Tensor*>(new MNN::Tensor(dimSize, static_cast<MNN::Tensor::DimensionType>(type)));
}
Tensor* Tensor_createFromTensor(const Tensor* tensor, DimensionType type, int allocMemory) {
    return reinterpret_cast<Tensor*>(new MNN::Tensor(reinterpret_cast<const MNN::Tensor*>(tensor), static_cast<MNN::Tensor::DimensionType>(type), allocMemory));
}
void Tensor_destroy(Tensor* tensor) {
    delete reinterpret_cast<MNN::Tensor*>(tensor);
}
Tensor* Tensor_createDevice(const int* shape, size_t shapeSize, halide_type_t typeCode, DimensionType dimType) {
    std::vector<int> shapeVec(shape, shape + shapeSize);
    return reinterpret_cast<Tensor*>(MNN::Tensor::createDevice(shapeVec, typeCode, static_cast<MNN::Tensor::DimensionType>(dimType)));
}
Tensor* Tensor_createWith(const int* shape, size_t shapeSize, halide_type_t typeCode, void* data, DimensionType dimType) {
    std::vector<int> shapeVec(shape, shape + shapeSize);
    return reinterpret_cast<Tensor*>(MNN::Tensor::create(shapeVec, typeCode, data, static_cast<MNN::Tensor::DimensionType>(dimType)));
}
int Tensor_copyFromHostTensor(Tensor* deviceTensor, const Tensor* hostTensor) {
    return reinterpret_cast<MNN::Tensor*>(deviceTensor)->copyFromHostTensor(reinterpret_cast<const MNN::Tensor*>(hostTensor));
}
int Tensor_copyToHostTensor(const Tensor* deviceTensor, Tensor* hostTensor) {
    return reinterpret_cast<const MNN::Tensor*>(deviceTensor)->copyToHostTensor(reinterpret_cast<MNN::Tensor*>(hostTensor));
}
Tensor* Tensor_createHostTensorFromDevice(const Tensor* deviceTensor, int copyData) {
    return reinterpret_cast<Tensor*>(MNN::Tensor::createHostTensorFromDevice(reinterpret_cast<const MNN::Tensor*>(deviceTensor), copyData));
}
const void* Tensor_host(const Tensor* tensor) {
    return reinterpret_cast<const MNN::Tensor*>(tensor)->host<void>();
}
uint64_t Tensor_deviceId(const Tensor* tensor) {
    return reinterpret_cast<const MNN::Tensor*>(tensor)->deviceId();
}
int Tensor_dimensions(const Tensor* tensor) {
    return reinterpret_cast<const MNN::Tensor*>(tensor)->dimensions();
}
void Tensor_shape(const Tensor* tensor, int* shape) {
    auto shapeVec = reinterpret_cast<const MNN::Tensor*>(tensor)->shape();
    std::copy(shapeVec.begin(), shapeVec.end(), shape);
}
int Tensor_size(const Tensor* tensor) {
    return reinterpret_cast<const MNN::Tensor*>(tensor)->size();
}
size_t Tensor_usize(const Tensor* tensor) {
    return reinterpret_cast<const MNN::Tensor*>(tensor)->usize();
}
int Tensor_elementSize(const Tensor* tensor) {
    return reinterpret_cast<const MNN::Tensor*>(tensor)->elementSize();
}
int Tensor_width(const Tensor* tensor) {
    return reinterpret_cast<const MNN::Tensor*>(tensor)->width();
}
int Tensor_height(const Tensor* tensor) {
    return reinterpret_cast<const MNN::Tensor*>(tensor)->height();
}
int Tensor_channel(const Tensor* tensor) {
    return reinterpret_cast<const MNN::Tensor*>(tensor)->channel();
}
int Tensor_batch(const Tensor* tensor) {
    return reinterpret_cast<const MNN::Tensor*>(tensor)->batch();
}
int Tensor_stride(const Tensor* tensor, int index) {
    return reinterpret_cast<const MNN::Tensor*>(tensor)->stride(index);
}
int Tensor_length(const Tensor* tensor, int index) {
    return reinterpret_cast<const MNN::Tensor*>(tensor)->length(index);
}
void Tensor_setStride(Tensor* tensor, int index, int stride) {
    reinterpret_cast<MNN::Tensor*>(tensor)->setStride(index, stride);
}
void Tensor_setLength(Tensor* tensor, int index, int length) {
    reinterpret_cast<MNN::Tensor*>(tensor)->setLength(index, length);
}
int Tensor_getDeviceInfo(const Tensor* tensor, void* dst, int forwardType) {
    return reinterpret_cast<const MNN::Tensor*>(tensor)->getDeviceInfo(dst, forwardType);
}
void Tensor_print(const Tensor* tensor) {
    reinterpret_cast<const MNN::Tensor*>(tensor)->print();
}
void Tensor_printShape(const Tensor* tensor) {
    reinterpret_cast<const MNN::Tensor*>(tensor)->printShape();
}
void* Tensor_map(Tensor* tensor, MapType mtype, DimensionType dtype) {
    return reinterpret_cast<MNN::Tensor*>(tensor)->map(static_cast<MNN::Tensor::MapType>(mtype), static_cast<MNN::Tensor::DimensionType>(dtype));
}
void Tensor_unmap(Tensor* tensor, MapType mtype, DimensionType dtype, void* mapPtr) {
    reinterpret_cast<MNN::Tensor*>(tensor)->unmap(static_cast<MNN::Tensor::MapType>(mtype), static_cast<MNN::Tensor::DimensionType>(dtype), mapPtr);
}
int Tensor_wait(Tensor* tensor, MapType mtype, int finish) {
    return reinterpret_cast<MNN::Tensor*>(tensor)->wait(static_cast<MNN::Tensor::MapType>(mtype), finish);
}
int Tensor_setDevicePtr(Tensor* tensor, const void* devicePtr, int memoryType) {
    return reinterpret_cast<MNN::Tensor*>(tensor)->setDevicePtr(devicePtr, memoryType);
}
} // extern "C"
