use anyhow::Result;
use core::marker::PhantomData;
use mnn_sys::*;

/// A Tensor object from MNN
#[repr(transparent)]
pub struct Tensor<T> {
    tensor: *mut MNN::Tensor,
    __marker: PhantomData<T>,
}

pub struct Host;
pub struct Device;

pub trait TensorType {}

impl TensorType for Host {}
impl TensorType for Device {}

impl Tensor<Host> {
    /// Create a new Tensor in the Host (CPU) from a Device Tensor (GPU/NPU)
    /// # Arguments
    /// * `device` - The device tensor to copy from
    /// * `copy` - If true, the data will be copied from the device to the host tensor
    pub fn create_host_tensor_from_device(device: &Tensor<Device>, copy: bool) -> Self {
        let tensor = unsafe { MNN::Tensor::createHostTensorFromDevice(device.tensor, copy) };
        Self {
            tensor,
            __marker: PhantomData,
        }
    }
}

impl<T> Tensor<T> {
    /// Create a new Tensor from a raw pointer
    pub unsafe fn from_raw(tensor: *mut MNN::Tensor) -> Self {
        Self {
            tensor,
            __marker: PhantomData,
        }
    }

    /// Get the raw pointer to the Tensor
    pub fn as_ptr(&self) -> *mut MNN::Tensor {
        self.tensor
    }

    pub unsafe fn as_reference(&self) -> &MNN::Tensor {
        core::mem::transmute(self.tensor)
    }

    pub fn copy_to_host_tensor(&self, host: &Tensor<Host>) -> Result<()> {
        let result = unsafe { MNN::Tensor::copyToHostTensor(self.as_reference(), host.tensor) };
        match result {
            true => Ok(()),
            _ => Err(anyhow::anyhow!("Error copying tensor to host tensor")),
        }
    }
}
