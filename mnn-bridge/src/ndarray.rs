use error_stack::*;
use ndarray::*;

#[derive(Debug)]
pub struct MnnBridge;
impl Context for MnnBridge {}
impl core::fmt::Display for MnnBridge {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "MnnBridgeError")
    }
}

pub trait MnnToNdarray {
    type H: mnn::HalideType;
    fn as_ndarray<D: Dimension>(&self) -> ndarray::ArrayView<Self::H, D> {
        self.try_as_ndarray::<D>()
            .expect("Failed to create ndarray::ArrayViewD from mnn::Tensor")
    }
    fn try_as_ndarray<D: Dimension>(&self) -> Result<ndarray::ArrayView<Self::H, D>, MnnBridge>;
}

pub trait MnnToNdarrayMut {
    type H: mnn::HalideType;
    fn as_ndarray_mut<D: Dimension>(&mut self) -> ndarray::ArrayViewMut<Self::H, D> {
        self.try_as_ndarray_mut::<D>()
            .expect("Failed to create ndarray::ArrayViewMutD from mnn::Tensor")
    }
    fn try_as_ndarray_mut<D: Dimension>(
        &mut self,
    ) -> Result<ndarray::ArrayViewMut<Self::H, D>, MnnBridge>;
}

pub trait NdarrayToMnn {
    type H: mnn::HalideType;
    fn as_mnn_tensor(&self) -> Option<mnn::Tensor<mnn::Ref<mnn::Host<Self::H>>>>;
}

pub trait NdarrayToMnnMut {
    type H: mnn::HalideType;
    fn as_mnn_tensor_mut(&mut self) -> Option<mnn::Tensor<mnn::RefMut<mnn::Host<Self::H>>>>;
}

const _: () = {
    impl<T> MnnToNdarray for mnn::Tensor<T>
    where
        T: mnn::TensorType + mnn::HostTensorType,
        T::H: mnn::HalideType,
    {
        type H = T::H;
        fn try_as_ndarray<D: Dimension>(
            &self,
        ) -> Result<ndarray::ArrayView<Self::H, D>, MnnBridge> {
            let shape = self
                .shape()
                .as_ref()
                .into_iter()
                .copied()
                .map(|i| i as usize)
                .collect::<Vec<_>>();
            let data = self.host();
            Ok(ndarray::ArrayViewD::from_shape(shape, data)
                .change_context(MnnBridge)?
                .into_dimensionality()
                .change_context(MnnBridge)?)
        }
    }

    impl<T> MnnToNdarrayMut for mnn::Tensor<T>
    where
        T: mnn::TensorType + mnn::MutableTensorType + mnn::HostTensorType,
        T::H: mnn::HalideType,
    {
        type H = T::H;
        fn try_as_ndarray_mut<D: Dimension>(
            &mut self,
        ) -> Result<ndarray::ArrayViewMut<Self::H, D>, MnnBridge> {
            let shape = self
                .shape()
                .as_ref()
                .into_iter()
                .copied()
                .map(|i| i as usize)
                .collect::<Vec<_>>();
            let data = self.host_mut();
            Ok(ndarray::ArrayViewMutD::from_shape(shape, data)
                .change_context(MnnBridge)?
                .into_dimensionality()
                .change_context(MnnBridge)?)
        }
    }

    impl<T, D, A> NdarrayToMnn for ndarray::ArrayBase<A, D>
    where
        A: ndarray::Data<Elem = T>,
        D: ndarray::Dimension,
        T: mnn::HalideType,
    {
        type H = T;
        fn as_mnn_tensor(&self) -> Option<mnn::Tensor<mnn::Ref<mnn::Host<Self::H>>>> {
            let shape = self.shape().iter().map(|i| *i as i32).collect::<Vec<_>>();
            let data = self.as_slice()?;
            Some(mnn::Tensor::borrowed(shape, data))
        }
    }

    impl<T, D, A> NdarrayToMnnMut for ndarray::ArrayBase<A, D>
    where
        A: ndarray::DataMut<Elem = T>,
        D: ndarray::Dimension,
        T: mnn::HalideType,
    {
        type H = T;
        fn as_mnn_tensor_mut(&mut self) -> Option<mnn::Tensor<mnn::RefMut<mnn::Host<Self::H>>>> {
            let shape = self.shape().iter().map(|i| *i as i32).collect::<Vec<_>>();
            let data = self.as_slice_mut()?;
            Some(mnn::Tensor::borrowed_mut(shape, data))
        }
    }
};
#[test]
pub fn test_tensor_to_ndarray_ref() {
    let mut tensor: mnn::Tensor<mnn::Host<i32>> =
        mnn::Tensor::new([1, 2, 3], mnn::DimensionType::Caffe);
    tensor.fill(64);
    let ndarr = tensor.as_ndarray();
    let ndarr_other = ndarray::Array3::from_shape_vec([1, 2, 3], [64; 6].to_vec()).unwrap();
    assert_eq!(ndarr, ndarr_other);
}
#[test]
pub fn test_tensor_to_ndarray_ref_mut() {
    let mut data = vec![100; 8 * 8 * 3];
    let mut tensor: mnn::Tensor<mnn::RefMut<mnn::Host<i16>>> =
        mnn::Tensor::borrowed_mut([8, 8, 3], &mut data);
    let mut ndarray = tensor.as_ndarray_mut::<Ix3>();
    ndarray.fill(600);
    assert_eq!(data, [600; 8 * 8 * 3]);
}
#[test]
pub fn test_ndarray_to_tensor_ref_mut() {
    let mut arr = ndarray::Array3::from_shape_vec([1, 2, 3], [64; 6].to_vec()).unwrap();
    arr.as_mnn_tensor_mut().unwrap().fill(600);
    assert_eq!(arr.as_slice().unwrap(), &[600; 6]);
}
#[test]
pub fn test_ndarray_to_tensor_ref() {
    let arr = ndarray::Array3::from_shape_vec([1, 2, 3], [64; 6].to_vec()).unwrap();
    let t = arr.as_mnn_tensor().unwrap();
    assert_eq!(t.host(), &[64; 6]);
}
