use anyhow::Result;
use autocxx::WithinUniquePtr;
use core::marker::PhantomData;
use mnn_sys::*;
#[rustfmt::skip]
use ::tap::*;

/// A Tensor object from MNN
#[repr(transparent)]
pub struct Tensor<T, D> {
    tensor: *mut MNN::Tensor,
    __data_type: PhantomData<D>,
    __type: PhantomData<T>,
}

#[repr(u32)]
pub enum DimensionType {
    Tensorflow = MNN::Tensor_DimensionType::TENSORFLOW as u32,
    Caffe = MNN::Tensor_DimensionType::CAFFE as u32,
    CaffeC4 = MNN::Tensor_DimensionType::CAFFE_C4 as u32,
}
impl From<MNN::Tensor_DimensionType> for DimensionType {
    fn from(value: MNN::Tensor_DimensionType) -> Self {
        match value {
            MNN::Tensor_DimensionType::TENSORFLOW => DimensionType::Tensorflow,
            MNN::Tensor_DimensionType::CAFFE => DimensionType::Caffe,
            MNN::Tensor_DimensionType::CAFFE_C4 => DimensionType::CaffeC4,
        }
    }
}
impl From<DimensionType> for MNN::Tensor_DimensionType {
    fn from(value: DimensionType) -> Self {
        match value {
            DimensionType::Tensorflow => MNN::Tensor_DimensionType::TENSORFLOW,
            DimensionType::Caffe => MNN::Tensor_DimensionType::CAFFE,
            DimensionType::CaffeC4 => MNN::Tensor_DimensionType::CAFFE_C4,
        }
    }
}

pub struct Host;
pub struct Device;

pub trait TensorType {}

impl TensorType for Host {}
impl TensorType for Device {}

impl<D> Tensor<Host, D> {
    /// Create a new Tensor in the Host (CPU) from a Device Tensor (GPU/NPU)
    /// # Arguments
    /// * `device` - The device tensor to copy from
    /// * `copy` - If true, the data will be copied from the device to the host tensor
    pub fn create_host_tensor_from_device(device: &Tensor<Device, D>, copy: bool) -> Self {
        let tensor = unsafe { MNN::Tensor::createHostTensorFromDevice(device.tensor, copy) };
        Self {
            tensor,
            __data_type: PhantomData,
            __type: PhantomData,
        }
    }
}

impl<T, D> Tensor<T, D> {
    /// Create a new Tensor from a raw pointer
    pub unsafe fn from_raw(tensor: *mut MNN::Tensor) -> Self {
        Self {
            tensor,
            __data_type: PhantomData,
            __type: PhantomData,
        }
    }

    /// Get the raw pointer to the Tensor
    pub fn as_ptr(&self) -> *mut MNN::Tensor {
        self.tensor
    }

    pub unsafe fn as_reference(&self) -> &MNN::Tensor {
        core::mem::transmute(self.tensor)
    }

    pub fn copy_to_host_tensor(&self, host: &Tensor<Host, D>) -> Result<()> {
        let result = unsafe { MNN::Tensor::copyToHostTensor(self.as_reference(), host.tensor) };
        match result {
            true => Ok(()),
            _ => Err(anyhow::anyhow!("Error copying tensor to host tensor")),
        }
    }

    pub fn get_dimension_type(&self) -> DimensionType {
        unsafe { MNN::Tensor::getDimensionType(self.as_reference()) }.into()
    }

    pub fn create_device<HT>(shape: &[i32], dimension_type: DimensionType) -> Tensor<T, D>
    where
        D: HalideType,
    {
        let shape = TensorShape {
            dims: shape.to_vec(),
        };
        let tensor = MNN::glueTensorCreateDevice(
            &shape,
            dimension_type.into(),
        );
        Self {
            tensor,
            __data_type: PhantomData,
            __type: PhantomData,
        }
    }
}


pub trait HalideType {
    fn halide() -> HalideTypes;
}
macro_rules! halide_types {
    ($($t:ty => $ht:ident),*) => {
        $(
            impl HalideType for $t {
                fn halide() -> HalideTypes {
                    HalideTypes::$ht
                }
            }
        )*
    };
}

halide_types! {
    f32 => float,
    f64 => double,
    bool => bool,
    u8 => uint8_t,
    u16 => uint16_t,
    u32 => uint32_t,
    u64 => uint64_t,
    i8 => int8_t,
    i16 => int16_t,
    i32 => int32_t,
    i64 => int64_t
}
