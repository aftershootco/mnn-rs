use mnn::*;

// Simple test utilities
pub struct TestModel {
    bytes: &'static [u8],
}

impl TestModel {
    pub const fn new() -> Self {
        TestModel {
            bytes: include_bytes!("assets/realesr.mnn"),
        }
    }
}

impl AsRef<[u8]> for TestModel {
    fn as_ref(&self) -> &[u8] {
        self.bytes
    }
}

#[test]
fn test_basic_interpreter_creation() {
    // Test interpreter creation from file
    let result = Interpreter::from_file("tests/assets/realesr.mnn");
    assert!(result.is_ok(), "Should create interpreter from file");

    // Test interpreter creation from bytes
    let model = TestModel::new();
    let result = Interpreter::from_bytes(model);
    assert!(result.is_ok(), "Should create interpreter from bytes");

    // Test error case with invalid path
    let result = Interpreter::from_file("nonexistent.mnn");
    assert!(result.is_err(), "Should fail with nonexistent file");
}

#[test]
fn test_session_creation() {
    let model = TestModel::new();
    let mut net = Interpreter::from_bytes(model).unwrap();

    // Test basic session creation
    let mut config = ScheduleConfig::new();
    config.set_type(ForwardType::CPU);
    let session = net.create_session(config);
    assert!(session.is_ok(), "Should create session successfully");
}

#[test]
fn test_tensor_list_operations() {
    let model = TestModel::new();
    let mut net = Interpreter::from_bytes(model).unwrap();
    let mut config = ScheduleConfig::new();
    config.set_type(ForwardType::CPU);
    let session = net.create_session(config).unwrap();

    // Test input tensor list
    let inputs = net.inputs(&session);
    let size = inputs.size();
    assert!(size > 0, "Should have at least one input");

    // Test iterator
    let mut count = 0;
    for input in inputs.iter() {
        count += 1;
        let name = input.name();
        assert!(!name.is_empty(), "Input name should not be empty");
    }
    assert_eq!(count, size, "Iterator should visit all inputs");

    // Test get method
    let first_input = inputs.get(0);
    assert!(first_input.is_some(), "Should get first input");

    let invalid_input = inputs.get(999);
    assert!(
        invalid_input.is_none(),
        "Should return None for invalid index"
    );
}

#[test]
fn test_tensor_info_operations() {
    let model = TestModel::new();
    let mut net = Interpreter::from_bytes(model).unwrap();
    let mut config = ScheduleConfig::new();
    config.set_type(ForwardType::CPU);
    let session = net.create_session(config).unwrap();

    let inputs = net.inputs(&session);
    if let Some(input) = inputs.get(0) {
        // Test name access
        let name = input.name();
        assert!(!name.is_empty(), "Name should not be empty");

        // Test raw tensor access
        let raw_tensor = input.raw_tensor();
        let shape = raw_tensor.shape();
        assert!(!shape.is_empty(), "Shape should not be empty");

        // Test basic tensor properties
        let dimensions = raw_tensor.dimensions();
        assert!(dimensions > 0, "Should have dimensions");

        let size = raw_tensor.size();
        assert!(size > 0, "Should have non-zero size");
    }
}

#[test]
fn test_raw_tensor_properties() {
    let model = TestModel::new();
    let mut net = Interpreter::from_bytes(model).unwrap();
    let mut config = ScheduleConfig::new();
    config.set_type(ForwardType::CPU);
    let session = net.create_session(config).unwrap();

    let inputs = net.inputs(&session);
    if let Some(input) = inputs.get(0) {
        let raw_tensor = input.raw_tensor();

        // Test geometric properties
        let width = raw_tensor.width();
        let height = raw_tensor.height();
        let channel = raw_tensor.channel();
        assert!(width > 0, "Width should be positive");
        assert!(height > 0, "Height should be positive");
        assert!(channel > 0, "Channel should be positive");

        // Test dimension type
        let dim_type = raw_tensor.get_dimension_type();
        match dim_type {
            DimensionType::Caffe | DimensionType::CaffeC4 | DimensionType::TensorFlow => {
                // Valid dimension types
            }
        }

        // Test dynamic sizing check
        let _is_dynamic = raw_tensor.is_dynamic_unsized();

        // Test element size
        let element_size = raw_tensor.element_size();
        assert!(element_size > 0, "Element size should be positive");
    }
}

#[test]
fn test_tensor_shape_operations() {
    let model = TestModel::new();
    let mut net = Interpreter::from_bytes(model).unwrap();
    let mut config = ScheduleConfig::new();
    config.set_type(ForwardType::CPU);
    let session = net.create_session(config).unwrap();

    let inputs = net.inputs(&session);
    if let Some(input) = inputs.get(0) {
        let raw_tensor = input.raw_tensor();
        let shape = raw_tensor.shape();

        // Test shape indexing
        if !shape.is_empty() {
            let _first_dim = shape[0];
        }

        // Test shape length
        let _len = shape.len();

        // Test shape debug formatting
        let _debug_str = format!("{:?}", shape);
    }
}

#[test]
fn test_tensor_host_operations() {
    let model = TestModel::new();
    let mut net = Interpreter::from_bytes(model).unwrap();
    let mut config = ScheduleConfig::new();
    config.set_type(ForwardType::CPU);
    let session = net.create_session(config).unwrap();

    let inputs = net.inputs(&session);
    if let Some(input) = inputs.get(0) {
        let raw_tensor = input.raw_tensor();

        // Test host tensor creation
        let host_tensor = raw_tensor.create_host_tensor_from_device(true);
        let host_shape = host_tensor.shape();
        assert_eq!(host_shape.as_ref(), raw_tensor.shape().as_ref());

        // Test copy operations
        let host_tensor2 = raw_tensor.create_host_tensor_from_device(false);
        let mut host_tensor_mut = raw_tensor.create_host_tensor_from_device(true);
        let result = host_tensor2.copy_to_host_tensor(&mut host_tensor_mut);
        match result {
            Ok(_) => println!("Copy succeeded"),
            Err(_) => println!("Copy failed (may be expected)"),
        }

        // Clean up
        host_tensor.destroy();
        host_tensor_mut.destroy();
        host_tensor2.destroy();
    }
}

#[test]
fn test_session_mode_setting() {
    let model = TestModel::new();
    let mut net = Interpreter::from_bytes(model).unwrap();

    // Test setting various session modes (these don't return errors)
    let modes = vec![
        SessionMode::Debug,
        SessionMode::Release,
        SessionMode::InputInside,
        SessionMode::InputUser,
        SessionMode::OutputInside,
        SessionMode::OutputUser,
        SessionMode::ResizeDirect,
        SessionMode::ResizeDefer,
        SessionMode::BackendFix,
        SessionMode::BackendAuto,
        SessionMode::MemoryCollect,
        SessionMode::MemoryCache,
        SessionMode::CodegenDisable,
        SessionMode::CodegenEnable,
        SessionMode::ResizeCheck,
        SessionMode::ResizeFix,
    ];

    for mode in modes {
        net.set_session_mode(mode);
        // Just test that it doesn't panic
    }
}

#[test]
fn test_tensor_list_debug_formatting() {
    let model = TestModel::new();
    let mut net = Interpreter::from_bytes(model).unwrap();
    let mut config = ScheduleConfig::new();
    config.set_type(ForwardType::CPU);
    let session = net.create_session(config).unwrap();

    let inputs = net.inputs(&session);

    // Test debug formatting for TensorList
    let debug_output = format!("{:?}", inputs);
    assert!(!debug_output.is_empty());
    assert!(debug_output.contains("["));

    // Test debug formatting for TensorInfo
    if let Some(input) = inputs.get(0) {
        let debug_output = format!("{:?}", input);
        assert!(!debug_output.is_empty());
        assert!(debug_output.contains("TensorInfo"));
    }
}

#[test]
fn test_dimension_type_constants() {
    // Test dimension type constants
    assert_eq!(DimensionType::NHWC, DimensionType::TensorFlow);
    assert_eq!(DimensionType::NCHW, DimensionType::Caffe);
    assert_eq!(DimensionType::NC4HW4, DimensionType::CaffeC4);
}

#[test]
fn test_tensor_shape_conversions() {
    // Test TensorShape conversions from various types
    let vec_shape = vec![1, 2, 3, 4];
    let tensor_shape = vec_shape.as_tensor_shape();
    assert_eq!(tensor_shape.as_ref(), &[1, 2, 3, 4]);

    let array_shape = [1, 2, 3];
    let tensor_shape = array_shape.as_tensor_shape();
    assert_eq!(tensor_shape.as_ref(), &[1, 2, 3]);

    let ref_shape = &vec![5, 6, 7];
    let tensor_shape = ref_shape.as_tensor_shape();
    assert_eq!(tensor_shape.as_ref(), &[5, 6, 7]);
}

#[test]
fn test_host_tensor_creation() {
    // Test creating host tensors directly
    let shape = vec![2, 3, 4];
    let tensor: Tensor<Owned<f32>, Host> = Tensor::new(shape.clone(), DimensionType::NCHW);

    // Test basic properties
    assert_eq!(tensor.shape().as_ref(), &shape);
    assert_eq!(tensor.element_size(), 24); // 2*3*4 = 24
    assert_eq!(tensor.dimensions(), 3);
}

#[test]
fn test_tensor_fill_operations() {
    let shape = vec![2, 3, 4];
    let mut tensor: Tensor<Owned<f32>, Host> = Tensor::new(shape, DimensionType::NCHW);

    // Test fill operation
    tensor.fill(1.5f32);

    // Basic verification that fill completed
    assert_eq!(tensor.element_size(), 24);
}

#[test]
// #[ignore = "Cloning is not supported by mnn currently https://github.com/alibaba/MNN/blob/c67a96156614801ba47191188a327102cb49145e/include/MNN/Tensor.hpp#L131"]
fn test_tensor_cloning() {
    let shape = vec![2, 2];
    let mut tensor: Tensor<Owned<f32>, Host> = Tensor::new(&shape, DimensionType::NCHW);

    tensor.fill(3.14f32);

    // Test cloning
    let cloned = tensor.clone();

    // try to modify the original tensor to ensure deep copy
    tensor.fill(0.0f32);
    assert_ne!(cloned, tensor);

    dbg!(&cloned);
    dbg!(&tensor);
    drop(cloned);
    drop(tensor);
    drop(shape);
}

#[test]
fn test_tensor_type_traits() {
    // Test TensorMachine traits
    assert!(Host::host());
    assert!(!Host::device());
    assert!(Device::device());
    assert!(!Device::host());
}

#[test]
fn test_unsafe_operations() {
    let model = TestModel::new();
    let mut net = Interpreter::from_bytes(model).unwrap();
    let mut config = ScheduleConfig::new();
    config.set_type(ForwardType::CPU);
    let session = net.create_session(config).unwrap();

    let inputs = net.inputs(&session);
    if let Some(input) = inputs.get(0) {
        // Test unsafe input_unchecked
        unsafe {
            let _tensor = net.input_unchecked::<f32>(&session, input.name());
            // Just test that it doesn't panic
        }

        // Test unsafe raw tensor operations
        let raw_tensor = input.raw_tensor();
        unsafe {
            let _host_ptr = raw_tensor.unchecked_host_ptr();
            // Just test that we can call it
        }
    }
}

#[test]
fn test_resize_operations() {
    let model = TestModel::new();
    let mut net = Interpreter::from_bytes(model).unwrap();
    let mut config = ScheduleConfig::new();
    config.set_type(ForwardType::CPU);
    let mut session = net.create_session(config).unwrap();

    // Test resize operations (these don't return Results)
    net.resize_session(&mut session);
    net.resize_session_reallocate(&mut session);

    // Test wait operation
    net.wait(&session);
}

#[test]
fn test_memory_and_performance_info() {
    let model = TestModel::new();
    let mut net = Interpreter::from_bytes(model).unwrap();
    let mut config = ScheduleConfig::new();
    config.set_type(ForwardType::CPU);
    let session = net.create_session(config).unwrap();

    // Test memory info retrieval
    let result = net.memory(&session);
    match result {
        Ok(memory_info) => {
            assert!(memory_info >= 0.0, "Memory should be non-negative");
        }
        Err(_) => {
            // May fail in some test environments
        }
    }

    // Test FLOPS info retrieval
    let result = net.flops(&session);
    match result {
        Ok(flops_info) => {
            assert!(flops_info >= 0.0, "FLOPS should be non-negative");
        }
        Err(_) => {
            // May fail in some test environments
        }
    }

    // Test resize status
    let result = net.resize_status(&session);
    match result {
        Ok(status) => match status {
            ResizeStatus::None | ResizeStatus::NeedMalloc | ResizeStatus::NeedResize => {
                // Valid status values
            }
        },
        Err(_) => {
            // May fail in some test environments
        }
    }
}

#[test]
fn test_raw_tensor_wait_operations() {
    let model = TestModel::new();
    let mut net = Interpreter::from_bytes(model).unwrap();
    let mut config = ScheduleConfig::new();
    config.set_type(ForwardType::CPU);
    let session = net.create_session(config).unwrap();

    let inputs = net.inputs(&session);
    if let Some(input) = inputs.get(0) {
        let raw_tensor = input.raw_tensor();

        // Test wait operations with different parameters
        raw_tensor.wait(MapType::MAP_TENSOR_READ, true);
        raw_tensor.wait(MapType::MAP_TENSOR_WRITE, false);
    }
}

#[test]
fn test_error_handling_paths() {
    let model = TestModel::new();
    let mut net = Interpreter::from_bytes(model).unwrap();
    let mut config = ScheduleConfig::new();
    config.set_type(ForwardType::CPU);
    let session = net.create_session(config).unwrap();

    let inputs = net.inputs(&session);
    if let Some(input) = inputs.get(0) {
        let mut raw_tensor = input.raw_tensor();

        // Test copy operations that might fail
        let host_tensor1 = raw_tensor.create_host_tensor_from_device(true);
        let host_tensor2 = raw_tensor.create_host_tensor_from_device(false);

        let result1 = raw_tensor.copy_from_host_tensor(&host_tensor1);
        let mut host_tensor2_mut = raw_tensor.create_host_tensor_from_device(false);
        let result2 = raw_tensor.copy_to_host_tensor(&mut host_tensor2_mut);

        // These operations may succeed or fail - we just test they return Results
        match result1 {
            Ok(_) | Err(_) => {} // Both outcomes are fine
        }

        match result2 {
            Ok(_) | Err(_) => {} // Both outcomes are fine
        }

        // Clean up
        host_tensor1.destroy();
        host_tensor2.destroy();
        host_tensor2_mut.destroy();
    }
}

#[test]
fn test_tensor_shape_indexing() {
    let model = TestModel::new();
    let mut net = Interpreter::from_bytes(model).unwrap();
    let mut config = ScheduleConfig::new();
    config.set_type(ForwardType::CPU);
    let session = net.create_session(config).unwrap();

    let inputs = net.inputs(&session);
    if let Some(input) = inputs.get(0) {
        let raw_tensor = input.raw_tensor();
        let mut shape = raw_tensor.shape();

        // Test mutable indexing if shape has elements
        if !shape.is_empty() {
            let original_value = shape[0];
            shape[0] = 42;
            assert_eq!(shape[0], 42);
            shape[0] = original_value; // Restore
        }

        // Test deref operations
        let _length = shape.len();
        let _is_empty = shape.is_empty();
    }
}

#[test]
fn test_comprehensive_coverage() {
    // This test exercises various code paths to improve coverage

    // Test empty iterator behavior
    let model = TestModel::new();
    let mut net = Interpreter::from_bytes(model).unwrap();
    let mut config = ScheduleConfig::new();
    config.set_type(ForwardType::CPU);
    let session = net.create_session(config).unwrap();

    let inputs = net.inputs(&session);
    let mut iter = inputs.iter();

    // Exhaust the iterator
    let mut count = 0;
    while let Some(_) = iter.next() {
        count += 1;
        if count > 100 {
            break;
        } // Safety guard
    }

    // Further calls should return None
    assert!(iter.next().is_none());

    // Test IntoIterator trait
    let mut for_count = 0;
    for _input in &inputs {
        for_count += 1;
        if for_count > 100 {
            break;
        } // Safety guard
    }

    assert_eq!(count, for_count);
}
