pub trait MnnToNdarray {
    type H: mnn::HalideType;
    fn as_ndarray(&self) -> ndarray::ArrayViewD<Self::H> {
        self.try_as_ndarray()
            .expect("Failed to create ndarray::ArrayViewD from mnn::Tensor")
    }
    fn try_as_ndarray(&self) -> Option<ndarray::ArrayViewD<Self::H>>;
}

impl<T> MnnToNdarray for mnn::Tensor<T>
where
    T: mnn::TensorType + mnn::HostTensorType,
    T::H: mnn::HalideType,
{
    type H = T::H;
    fn try_as_ndarray(&self) -> Option<ndarray::ArrayViewD<Self::H>> {
        let shape = self
            .shape()
            .as_ref()
            .into_iter()
            .copied()
            .map(|i| i as usize)
            .collect::<Vec<_>>();
        let data = self.host();
        ndarray::ArrayViewD::from_shape(shape, data).ok()
    }
}

#[test]
pub fn test_tensor_to_ndarray_ref() {
    let mut tensor: mnn::Tensor<mnn::Host<i32>> =
        mnn::Tensor::new([1, 2, 3], mnn::DimensionType::Caffe);
    tensor.fill(64);
    let ndarr = tensor.as_ndarray();
    let ndarr_2 = ndarray::Array3::from_shape_vec([1, 2, 3], [64; 6].to_vec())
        .unwrap()
        .into_dyn();
    assert_eq!(ndarr, ndarr_2);
}

pub trait MnnToNdarrayMut {
    type H: mnn::HalideType;
    fn as_ndarray_mut(&mut self) -> ndarray::ArrayViewMutD<Self::H> {
        self.try_as_ndarray_mut()
            .expect("Failed to create ndarray::ArrayViewMutD from mnn::Tensor")
    }
    fn try_as_ndarray_mut(&mut self) -> Option<ndarray::ArrayViewMutD<Self::H>>;
}

impl<T> MnnToNdarrayMut for mnn::Tensor<T>
where
    T: mnn::TensorType + mnn::MutableTensorType + mnn::HostTensorType,
    T::H: mnn::HalideType,
{
    type H = T::H;
    fn try_as_ndarray_mut(&mut self) -> Option<ndarray::ArrayViewMutD<Self::H>> {
        let shape = self
            .shape()
            .as_ref()
            .into_iter()
            .copied()
            .map(|i| i as usize)
            .collect::<Vec<_>>();
        let data = self.host_mut();
        ndarray::ArrayViewMutD::from_shape(shape, data).ok()
    }
}

#[test]
pub fn test_tensor_to_ndarray_ref_mut() {
    let mut data = vec![100; 8 * 8 * 3];
    let mut tensor: mnn::Tensor<mnn::RefMut<mnn::Host<i16>>> =
        mnn::Tensor::borrowed_mut([8, 8, 3], &mut data);
    let mut ndarray = tensor.as_ndarray_mut();
    ndarray.fill(600);
    assert_eq!(data, [600; 8 * 8 * 3]);
}

pub trait NdarrayToMnn {
    type H: mnn::HalideType;
    fn as_mnn_tensor(&self) -> Option<mnn::Tensor<mnn::Ref<mnn::Host<Self::H>>>>;
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

#[test]
pub fn test_ndarray_to_tensor_ref() {
    let arr = ndarray::Array3::from_shape_vec([1, 2, 3], [64; 6].to_vec()).unwrap();
    let t = arr.as_mnn_tensor().unwrap();
    assert_eq!(t.host(), &[64; 6]);
}

pub trait NdarrayToMnnMut {
    type H: mnn::HalideType;
    fn as_mnn_tensor_mut(&mut self) -> Option<mnn::Tensor<mnn::RefMut<mnn::Host<Self::H>>>>;
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

#[test]
pub fn test_ndarray_to_tensor_ref_mut() {
    let mut arr = ndarray::Array3::from_shape_vec([1, 2, 3], [64; 6].to_vec()).unwrap();
    arr.as_mnn_tensor_mut().unwrap().fill(600);
    assert_eq!(arr.as_slice().unwrap(), &[600; 6]);
}
