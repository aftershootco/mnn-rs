use error_stack::*;
use ndarray::*;
type Result<T, E> = core::result::Result<T, Report<E>>;

#[derive(Debug)]
pub struct MnnBridge;
impl core::error::Error for MnnBridge {}
impl core::fmt::Display for MnnBridge {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "MnnBridgeError")
    }
}

pub trait MnnToNdarray {
    type H: mnn::HalideType;
    fn as_ndarray<'a, D: Dimension>(&'a self) -> ndarray::ArrayView<'a, Self::H, D> {
        self.try_as_ndarray::<D>()
            .expect("Failed to create ndarray::ArrayView from mnn::Tensor")
    }
    fn try_as_ndarray<'a, D: Dimension>(
        &'a self,
    ) -> Result<ndarray::ArrayView<'a, Self::H, D>, MnnBridge>;
}

pub trait MnnToNdarrayMut {
    type H: mnn::HalideType;
    fn as_ndarray_mut<'a, D: Dimension>(&'a mut self) -> ndarray::ArrayViewMut<'a, Self::H, D> {
        self.try_as_ndarray_mut::<D>()
            .expect("Failed to create ndarray::ArrayViewMut from mnn::Tensor")
    }
    fn try_as_ndarray_mut<'a, D: Dimension>(
        &'a mut self,
    ) -> Result<ndarray::ArrayViewMut<'a, Self::H, D>, MnnBridge>;
}

use mnn::{Host, TensorView, TensorViewMut};

pub trait NdarrayAsMnn {
    type H: mnn::HalideType;
    fn as_mnn_tensor(&self) -> Result<TensorView<'_, Self::H, Host>, MnnBridge>;
}

pub trait NdarrayAsMnnMut {
    type H: mnn::HalideType;
    fn as_mnn_tensor_mut(&mut self) -> Result<TensorViewMut<'_, Self::H, Host>, MnnBridge>;
}

const _: () = {
    impl<T> MnnToNdarray for mnn::Tensor<T, Host>
    where
        T: mnn::TensorType,
        T::H: mnn::HalideType,
    {
        type H = T::H;
        fn try_as_ndarray<'a, D: Dimension>(
            &'a self,
        ) -> Result<ndarray::ArrayView<'a, Self::H, D>, MnnBridge> {
            let shape = self
                .shape()
                .as_ref()
                .iter()
                .copied()
                .map(|i| i as usize)
                .collect::<Vec<_>>();
            let data = self.host();
            ndarray::ArrayViewD::from_shape(shape, data)
                .change_context(MnnBridge)?
                .into_dimensionality()
                .change_context(MnnBridge)
        }
    }

    impl<T> MnnToNdarrayMut for mnn::Tensor<T, Host>
    where
        T: mnn::TensorType + mnn::MutableTensorType,
        T::H: mnn::HalideType,
    {
        type H = T::H;
        fn try_as_ndarray_mut<'a, D: Dimension>(
            &'a mut self,
        ) -> Result<ndarray::ArrayViewMut<'a, Self::H, D>, MnnBridge> {
            let shape = self
                .shape()
                .as_ref()
                .iter()
                .copied()
                .map(|i| i as usize)
                .collect::<Vec<_>>();
            let data = self.host_mut();
            ndarray::ArrayViewMutD::from_shape(shape, data)
                .change_context(MnnBridge)?
                .into_dimensionality()
                .change_context(MnnBridge)
        }
    }

    impl<T, D, A> NdarrayAsMnn for ndarray::ArrayBase<A, D>
    where
        A: ndarray::Data<Elem = T>,
        D: ndarray::Dimension,
        T: mnn::HalideType,
    {
        type H = T;
        fn as_mnn_tensor<'a>(&'a self) -> Result<TensorView<'a, T, Host>, MnnBridge> {
            let shape = self.shape().iter().map(|i| *i as i32).collect::<Vec<_>>();
            let data = self
                .as_slice()
                .ok_or(MnnBridge)
                .attach("Failed to get ndarray as a continuous slice")?;

            Ok(mnn::Tensor::borrowed(
                shape,
                mnn::DimensionType::default(),
                data,
            ))
        }
    }

    impl<T, D, A> NdarrayAsMnnMut for ndarray::ArrayBase<A, D>
    where
        A: ndarray::DataMut<Elem = T>,
        D: ndarray::Dimension,
        T: mnn::HalideType,
    {
        type H = T;
        fn as_mnn_tensor_mut(&mut self) -> Result<TensorViewMut<'_, T, Host>, MnnBridge> {
            let shape = self.shape().iter().map(|i| *i as i32).collect::<Vec<_>>();
            let data = self
                .as_slice_mut()
                .ok_or(MnnBridge)
                .attach("Failed to get ndarray as a continuous slice")?;
            Ok(mnn::Tensor::borrowed_mut(
                shape,
                mnn::DimensionType::default(),
                data,
            ))
        }
    }
};

#[test]
pub fn test_tensor_to_ndarray_ref() {
    let mut tensor: mnn::Tensor<mnn::Owned<i32>, Host> =
        mnn::Tensor::new([1, 2, 3], mnn::DimensionType::Caffe);
    tensor.fill(64).unwrap();
    let ndarr = tensor.as_ndarray();
    let ndarr_other = ndarray::Array3::from_shape_vec([1, 2, 3], [64; 6].to_vec()).unwrap();
    assert_eq!(ndarr, ndarr_other);
}

#[test]
pub fn test_tensor_to_ndarray_ref_mut() {
    let mut data = vec![100; 8 * 8 * 3];
    {
        let mut tensor: mnn::Tensor<mnn::View<&mut i32>, Host> =
            mnn::Tensor::borrowed_mut([8, 8, 3], mnn::DimensionType::default(), &mut data);
        let mut ndarray = tensor.as_ndarray_mut::<Ix3>();
        ndarray.fill(600);
    }
    assert_eq!(data, [600; 8 * 8 * 3]);
}

#[test]
pub fn test_ndarray_to_tensor_ref_mut() {
    let mut arr = ndarray::Array3::from_shape_vec([1, 2, 3], [64; 6].to_vec()).unwrap();
    arr.as_mnn_tensor_mut().unwrap().fill(600).unwrap();
    assert_eq!(arr.as_slice().unwrap(), &[600; 6]);
}

#[test]
pub fn test_ndarray_to_tensor_ref() {
    let arr = ndarray::Array3::from_shape_vec([1, 2, 3], [64; 6].to_vec()).unwrap();
    let t = arr.as_mnn_tensor().unwrap();
    assert_eq!(t.host(), &[64; 6]);
}
