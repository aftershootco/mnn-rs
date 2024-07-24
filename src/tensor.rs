use anyhow::Result;
use autocxx::WithinUniquePtr;
use core::marker::PhantomData;
use mnn_sys::*;
#[rustfmt::skip]
use ::tap::*;

/// A Tensor object from MNN
#[repr(transparent)]
pub struct Tensor<T = Host, D = f32> {
    tensor: *mut MNN::Tensor,
    __type: PhantomData<T>,
    __data_type: PhantomData<D>,
}

#[repr(u32)]
pub enum DimensionType {
    Tensorflow = MNN::Tensor_DimensionType::TENSORFLOW as u32,
    Caffe = MNN::Tensor_DimensionType::CAFFE as u32,
    CaffeC4 = MNN::Tensor_DimensionType::CAFFE_C4 as u32,
}

impl DimensionType {
    pub const NHWC: Self = Self::Tensorflow;
    pub const NCHW: Self = Self::Caffe;
    pub const NC4HW4: Self = Self::CaffeC4;
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

    // pub fn copy_from_device_tensor(&self, device: &Tensor<Device, D>) -> Result<()> {
    //     let result =
    //         unsafe { MNN::Tensor::copyFromDeviceTensor(self.as_reference(), device.tensor) };
    //     match result {
    //         true => Ok(()),
    //         _ => Err(anyhow::anyhow!("Error copying tensor from device tensor")),
    //     }
    // }
}

impl<D> Tensor<Device, D> {
    pub fn create_device(
        shape: impl AsRef<[i32]>,
        dimension_type: DimensionType,
    ) -> Tensor<Device, D>
    where
        D: HalideType,
    {
        let shape = TensorShape {
            dims: shape.as_ref().to_vec(),
        };
        let dimension_type: MNN::Tensor_DimensionType = dimension_type.into();
        let halide_type = D::halide();
        let tensor = MNN::glueTensorCreateDevice(&shape, &halide_type, dimension_type);
        Tensor {
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
        &*self.tensor
    }

    pub fn copy_to_host_tensor(&self, host: &mut Tensor<Host, D>) -> Result<()> {
        let result = unsafe { MNN::Tensor::copyToHostTensor(self.as_reference(), host.tensor) };
        match result {
            true => Ok(()),
            _ => Err(anyhow::anyhow!("Error copying tensor to host tensor")),
        }
    }

    pub fn copy_from_host_tensor(&mut self, host: &Tensor<Host, D>) -> Result<()> {
        use core::pin::Pin;
        let this_mut: Pin<&mut MNN::Tensor> = Pin::new(unsafe { &mut *self.tensor });
        let result = unsafe { MNN::Tensor::copyFromHostTensor(this_mut, host.tensor) };
        match result {
            true => Ok(()),
            _ => Err(anyhow::anyhow!("Error copying tensor from host tensor")),
        }
    }

    pub fn get_dimension_type(&self) -> DimensionType {
        unsafe { MNN::Tensor::getDimensionType(self.as_reference()) }.into()
    }

    pub fn print_shape(&self) {
        unsafe { MNN::Tensor::printShape(self.as_reference()) };
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
                    paste::paste! {
                        HalideTypes::[<halide_ $ht>]
                    }
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
