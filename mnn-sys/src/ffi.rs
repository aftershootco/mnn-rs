use autocxx::prelude::*;

include_cpp! {
    #include "MNN/Interpreter.hpp"
    #include "MNN/MNNForwardType.h"
    #include "MNN/Tensor.hpp"
    #include "MNN/ErrorCode.hpp"
    safety!(unsafe_ffi)

    generate!("MNNForwardType")
    generate!("MNN::Tensor")
    generate!("MNN::Session")
    generate!("MNN::Runtime")
    generate!("MNN::ScheduleConfig")
    generate!("MNN::getVersion")
    generate!("MNN::Interpreter")
}


pub use ffi::*;

