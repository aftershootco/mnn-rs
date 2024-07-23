use autocxx::prelude::*;

include_cpp! {
    #include "MNN/Interpreter.hpp"
    #include "MNN/MNNForwardType.h"
    #include "MNN/Tensor.hpp"
    #include "MNN/ErrorCode.hpp"
    #include "MNN/HalideRuntime.h"
    #include "glue/TensorGlue.hpp"
    safety!(unsafe_ffi)

    generate!("MNNForwardType")
    generate!("MNN::Tensor")
    generate!("MNN::Session")
    generate!("MNN::Runtime")
    generate!("MNN::ScheduleConfig")
    generate!("MNN::getVersion")
    generate!("MNN::Interpreter")
    // generate!("MNN::glueTensorCreateDevice")
    // extern_cpp_type!("MNN::TensorShape", crate::TensorShape)
    // extern_cpp_type!("MNN::HalideTypes", crate::HalideTypes)
}
pub use ffi::*;
