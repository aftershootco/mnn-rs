pub mod common;
use common::*;
use mnn::ForwardType;

#[test]
#[ignore = "takes too long"]
fn test_basic_cpu() {
    test_basic(ForwardType::CPU).unwrap();
}
#[cfg(feature = "metal")]
#[test]
fn test_basic_metal() {
    test_basic(ForwardType::Metal).unwrap();
}
#[cfg(feature = "opencl")]
#[test]
fn test_basic_opencl() {
    test_basic(ForwardType::OpenCL).unwrap();
}
#[cfg(feature = "coreml")]
#[test]
fn test_basic_coreml() {
    test_basic(ForwardType::CoreML).unwrap();
}
#[cfg(feature = "opengl")]
#[test]
fn test_basic_opengl() {
    test_basic(ForwardType::OpenGL).unwrap();
}

#[test]
#[ignore = "takes too long and unreliable on CI"]
fn test_multi_path_cpu_cpu() {
    test_multipath_session(ForwardType::CPU, ForwardType::CPU).unwrap();
}

#[cfg(feature = "opencl")]
#[test]
fn test_multi_path_opencl_cpu() {
    test_multipath_session(ForwardType::OpenCL, ForwardType::CPU).unwrap();
}
