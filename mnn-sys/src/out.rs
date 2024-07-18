#![feature(prelude_import)]
#[prelude_import]
use std::prelude::rust_2021::*;
#[macro_use]
extern crate std;
mod mnn {
    use autocxx::prelude::*;
    #[allow(non_snake_case)]
    #[allow(dead_code)]
    #[allow(non_upper_case_globals)]
    #[allow(non_camel_case_types)]
    mod ffi {
        pub trait ToCppString {
            fn into_cpp(self) -> cxx::UniquePtr<cxx::CxxString>;
        }
        impl ToCppString for &str {
            fn into_cpp(self) -> cxx::UniquePtr<cxx::CxxString> {
                make_string(self)
            }
        }
        impl ToCppString for String {
            fn into_cpp(self) -> cxx::UniquePtr<cxx::CxxString> {
                make_string(&self)
            }
        }
        impl ToCppString for &String {
            fn into_cpp(self) -> cxx::UniquePtr<cxx::CxxString> {
                make_string(self)
            }
        }
        impl ToCppString for cxx::UniquePtr<cxx::CxxString> {
            fn into_cpp(self) -> cxx::UniquePtr<cxx::CxxString> {
                self
            }
        }
        unsafe impl cxx::ExternType for bindgen::root::MNN::Tensor {
            type Id = (
                ::cxx::M,
                ::cxx::N,
                ::cxx::N,
                (),
                ::cxx::T,
                ::cxx::e,
                ::cxx::n,
                ::cxx::s,
                ::cxx::o,
                ::cxx::r,
            );
            type Kind = cxx::kind::Opaque;
        }
        unsafe impl cxx::ExternType for bindgen::root::halide_buffer_t {
            type Id = (
                ::cxx::h,
                ::cxx::a,
                ::cxx::l,
                ::cxx::i,
                ::cxx::d,
                ::cxx::e,
                ::cxx::__,
                ::cxx::b,
                ::cxx::u,
                ::cxx::f,
                ::cxx::f,
                ::cxx::e,
                ::cxx::r,
                ::cxx::__,
                ::cxx::t,
            );
            type Kind = cxx::kind::Opaque;
        }
        unsafe impl cxx::ExternType for bindgen::root::MNN::Tensor_DimensionType {
            type Id = (
                ::cxx::M,
                ::cxx::N,
                ::cxx::N,
                (),
                ::cxx::T,
                ::cxx::e,
                ::cxx::n,
                ::cxx::s,
                ::cxx::o,
                ::cxx::r,
                (),
                ::cxx::D,
                ::cxx::i,
                ::cxx::m,
                ::cxx::e,
                ::cxx::n,
                ::cxx::s,
                ::cxx::i,
                ::cxx::o,
                ::cxx::n,
                ::cxx::T,
                ::cxx::y,
                ::cxx::p,
                ::cxx::e,
            );
            type Kind = cxx::kind::Trivial;
        }
        unsafe impl cxx::ExternType for bindgen::root::MNN::Tensor_HandleDataType {
            type Id = (
                ::cxx::M,
                ::cxx::N,
                ::cxx::N,
                (),
                ::cxx::T,
                ::cxx::e,
                ::cxx::n,
                ::cxx::s,
                ::cxx::o,
                ::cxx::r,
                (),
                ::cxx::H,
                ::cxx::a,
                ::cxx::n,
                ::cxx::d,
                ::cxx::l,
                ::cxx::e,
                ::cxx::D,
                ::cxx::a,
                ::cxx::t,
                ::cxx::a,
                ::cxx::T,
                ::cxx::y,
                ::cxx::p,
                ::cxx::e,
            );
            type Kind = cxx::kind::Trivial;
        }
        unsafe impl cxx::ExternType for bindgen::root::halide_type_t {
            type Id = (
                ::cxx::h,
                ::cxx::a,
                ::cxx::l,
                ::cxx::i,
                ::cxx::d,
                ::cxx::e,
                ::cxx::__,
                ::cxx::t,
                ::cxx::y,
                ::cxx::p,
                ::cxx::e,
                ::cxx::__,
                ::cxx::t,
            );
            type Kind = cxx::kind::Opaque;
        }
        unsafe impl cxx::ExternType for bindgen::root::MNN::Tensor_MapType {
            type Id = (
                ::cxx::M,
                ::cxx::N,
                ::cxx::N,
                (),
                ::cxx::T,
                ::cxx::e,
                ::cxx::n,
                ::cxx::s,
                ::cxx::o,
                ::cxx::r,
                (),
                ::cxx::M,
                ::cxx::a,
                ::cxx::p,
                ::cxx::T,
                ::cxx::y,
                ::cxx::p,
                ::cxx::e,
            );
            type Kind = cxx::kind::Trivial;
        }
        mod bindgen {
            pub(super) mod root {
                /** \file

 This file declares the routines used by Halide internally in its
 runtime. On platforms that support weak linking, these can be
 replaced with user-defined versions by defining an extern "C"
 function with the same name and signature.

 When doing Just In Time (JIT) compilation methods on the Func being
 compiled must be called instead. The corresponding methods are
 documented below.

 All of these functions take a "void *user_context" parameter as their
 first argument; if the Halide kernel that calls back to any of these
 functions has been compiled with the UserContext feature set on its Target,
 then the value of that pointer passed from the code that calls the
 Halide kernel is piped through to the function.

 Some of these are also useful to call when using the default
 implementation. E.g. halide_shutdown_thread_pool.

 Note that even on platforms with weak linking, some linker setups
 may not respect the override you provide. E.g. if the override is
 in a shared library and the halide object files are linked directly
 into the output, the builtin versions of the runtime functions will
 be called. See your linker documentation for more details. On
 Linux, LD_DYNAMIC_WEAK=1 may help.
*/
                #[repr(C, align(8))]
                pub struct halide_buffer_t {
                    _pinned: core::marker::PhantomData<core::marker::PhantomPinned>,
                    _non_send_sync: core::marker::PhantomData<[*const u8; 0]>,
                    _data: ::core::cell::UnsafeCell<::core::mem::MaybeUninit<[u8; 64]>>,
                }
                /** A runtime tag for a type in the halide type system. Can be ints,
 unsigned ints, or floats of various bit-widths (the 'bits'
 field). Can also be vectors of the same (by setting the 'lanes'
 field to something larger than one). This struct should be
 exactly 32-bits in size.*/
                #[repr(C, align(4))]
                pub struct halide_type_t {
                    _pinned: core::marker::PhantomData<core::marker::PhantomPinned>,
                    _non_send_sync: core::marker::PhantomData<[*const u8; 0]>,
                    _data: ::core::cell::UnsafeCell<::core::mem::MaybeUninit<[u8; 8]>>,
                }
                unsafe impl autocxx::moveit::MakeCppStorage for root::halide_buffer_t {
                    unsafe fn allocate_uninitialized_cpp_storage() -> *mut root::halide_buffer_t {
                        cxxbridge::halide_buffer_t_autocxx_alloc_autocxx_wrapper_0xecc3669dd67dbc04()
                    }
                    unsafe fn free_uninitialized_cpp_storage(
                        arg0: *mut root::halide_buffer_t,
                    ) {
                        cxxbridge::halide_buffer_t_autocxx_free_autocxx_wrapper_0xecc3669dd67dbc04(
                            arg0,
                        )
                    }
                }
                unsafe impl autocxx::moveit::new::MoveNew for root::halide_buffer_t {
                    ///Synthesized move constructor.
                    unsafe fn move_new(
                        mut other: ::core::pin::Pin<
                            autocxx::moveit::MoveRef<'_, root::halide_buffer_t>,
                        >,
                        this: ::core::pin::Pin<
                            &mut ::core::mem::MaybeUninit<root::halide_buffer_t>,
                        >,
                    ) {
                        cxxbridge::halide_buffer_t_new_synthetic_move_ctor_0xecc3669dd67dbc04_autocxx_wrapper_0xecc3669dd67dbc04(
                            this.get_unchecked_mut().as_mut_ptr(),
                            {
                                let r: &mut _ = ::core::pin::Pin::into_inner_unchecked(
                                    other.as_mut(),
                                );
                                r
                            },
                        )
                    }
                }
                unsafe impl autocxx::moveit::new::CopyNew for root::halide_buffer_t {
                    ///Synthesized copy constructor.
                    unsafe fn copy_new(
                        other: &root::halide_buffer_t,
                        this: ::core::pin::Pin<
                            &mut ::core::mem::MaybeUninit<root::halide_buffer_t>,
                        >,
                    ) {
                        cxxbridge::halide_buffer_t_new_synthetic_const_copy_ctor_0xecc3669dd67dbc04_autocxx_wrapper_0xecc3669dd67dbc04(
                            this.get_unchecked_mut().as_mut_ptr(),
                            other,
                        )
                    }
                }
                unsafe impl autocxx::moveit::MakeCppStorage for root::halide_type_t {
                    unsafe fn allocate_uninitialized_cpp_storage() -> *mut root::halide_type_t {
                        cxxbridge::halide_type_t_autocxx_alloc_autocxx_wrapper_0xecc3669dd67dbc04()
                    }
                    unsafe fn free_uninitialized_cpp_storage(
                        arg0: *mut root::halide_type_t,
                    ) {
                        cxxbridge::halide_type_t_autocxx_free_autocxx_wrapper_0xecc3669dd67dbc04(
                            arg0,
                        )
                    }
                }
                unsafe impl autocxx::moveit::new::MoveNew for root::halide_type_t {
                    ///Synthesized move constructor.
                    unsafe fn move_new(
                        mut other: ::core::pin::Pin<
                            autocxx::moveit::MoveRef<'_, root::halide_type_t>,
                        >,
                        this: ::core::pin::Pin<
                            &mut ::core::mem::MaybeUninit<root::halide_type_t>,
                        >,
                    ) {
                        cxxbridge::new_synthetic_move_ctor_0xecc3669dd67dbc04_autocxx_wrapper_0xecc3669dd67dbc04(
                            this.get_unchecked_mut().as_mut_ptr(),
                            {
                                let r: &mut _ = ::core::pin::Pin::into_inner_unchecked(
                                    other.as_mut(),
                                );
                                r
                            },
                        )
                    }
                }
                unsafe impl autocxx::moveit::new::CopyNew for root::halide_type_t {
                    ///Synthesized copy constructor.
                    unsafe fn copy_new(
                        other: &root::halide_type_t,
                        this: ::core::pin::Pin<
                            &mut ::core::mem::MaybeUninit<root::halide_type_t>,
                        >,
                    ) {
                        cxxbridge::new_synthetic_const_copy_ctor_0xecc3669dd67dbc04_autocxx_wrapper_0xecc3669dd67dbc04(
                            this.get_unchecked_mut().as_mut_ptr(),
                            other,
                        )
                    }
                }
                pub mod MNN {
                    /** data container.
 data for host tensor is saved in `host` field. its memory is allocated malloc directly.
 data for device tensor is saved in `deviceId` field. its memory is allocated by session's backend.
 usually, device tensors are created by engine (like net, session).
 meanwhile, host tensors could be created by engine or user.*/
                    #[repr(C, align(8))]
                    pub struct Tensor {
                        _pinned: core::marker::PhantomData<core::marker::PhantomPinned>,
                        _non_send_sync: core::marker::PhantomData<[*const u8; 0]>,
                        _data: ::core::cell::UnsafeCell<
                            ::core::mem::MaybeUninit<[u8; 72]>,
                        >,
                    }
                    #[repr(u32)]
                    /// dimension type used to create tensor
                    pub enum Tensor_DimensionType {
                        /// for tensorflow net type. uses NHWC as data format.
                        TENSORFLOW = 0,
                        /// for caffe net type. uses NCHW as data format.
                        CAFFE = 1,
                        /// for caffe net type. uses NC4HW4 as data format.
                        CAFFE_C4 = 2,
                    }
                    #[automatically_derived]
                    impl ::core::clone::Clone for Tensor_DimensionType {
                        #[inline]
                        fn clone(&self) -> Tensor_DimensionType {
                            match self {
                                Tensor_DimensionType::TENSORFLOW => {
                                    Tensor_DimensionType::TENSORFLOW
                                }
                                Tensor_DimensionType::CAFFE => Tensor_DimensionType::CAFFE,
                                Tensor_DimensionType::CAFFE_C4 => {
                                    Tensor_DimensionType::CAFFE_C4
                                }
                            }
                        }
                    }
                    #[automatically_derived]
                    impl ::core::hash::Hash for Tensor_DimensionType {
                        #[inline]
                        fn hash<__H: ::core::hash::Hasher>(
                            &self,
                            state: &mut __H,
                        ) -> () {
                            let __self_discr = ::core::intrinsics::discriminant_value(
                                self,
                            );
                            ::core::hash::Hash::hash(&__self_discr, state)
                        }
                    }
                    #[automatically_derived]
                    impl ::core::marker::StructuralPartialEq for Tensor_DimensionType {}
                    #[automatically_derived]
                    impl ::core::cmp::PartialEq for Tensor_DimensionType {
                        #[inline]
                        fn eq(&self, other: &Tensor_DimensionType) -> bool {
                            let __self_discr = ::core::intrinsics::discriminant_value(
                                self,
                            );
                            let __arg1_discr = ::core::intrinsics::discriminant_value(
                                other,
                            );
                            __self_discr == __arg1_discr
                        }
                    }
                    #[automatically_derived]
                    impl ::core::cmp::Eq for Tensor_DimensionType {
                        #[inline]
                        #[doc(hidden)]
                        #[coverage(off)]
                        fn assert_receiver_is_total_eq(&self) -> () {}
                    }
                    #[repr(u32)]
                    /// handle type
                    pub enum Tensor_HandleDataType {
                        /// default handle type
                        HANDLE_NONE = 0,
                        /// string handle type
                        HANDLE_STRING = 1,
                    }
                    #[automatically_derived]
                    impl ::core::clone::Clone for Tensor_HandleDataType {
                        #[inline]
                        fn clone(&self) -> Tensor_HandleDataType {
                            match self {
                                Tensor_HandleDataType::HANDLE_NONE => {
                                    Tensor_HandleDataType::HANDLE_NONE
                                }
                                Tensor_HandleDataType::HANDLE_STRING => {
                                    Tensor_HandleDataType::HANDLE_STRING
                                }
                            }
                        }
                    }
                    #[automatically_derived]
                    impl ::core::hash::Hash for Tensor_HandleDataType {
                        #[inline]
                        fn hash<__H: ::core::hash::Hasher>(
                            &self,
                            state: &mut __H,
                        ) -> () {
                            let __self_discr = ::core::intrinsics::discriminant_value(
                                self,
                            );
                            ::core::hash::Hash::hash(&__self_discr, state)
                        }
                    }
                    #[automatically_derived]
                    impl ::core::marker::StructuralPartialEq for Tensor_HandleDataType {}
                    #[automatically_derived]
                    impl ::core::cmp::PartialEq for Tensor_HandleDataType {
                        #[inline]
                        fn eq(&self, other: &Tensor_HandleDataType) -> bool {
                            let __self_discr = ::core::intrinsics::discriminant_value(
                                self,
                            );
                            let __arg1_discr = ::core::intrinsics::discriminant_value(
                                other,
                            );
                            __self_discr == __arg1_discr
                        }
                    }
                    #[automatically_derived]
                    impl ::core::cmp::Eq for Tensor_HandleDataType {
                        #[inline]
                        #[doc(hidden)]
                        #[coverage(off)]
                        fn assert_receiver_is_total_eq(&self) -> () {}
                    }
                    #[repr(u32)]
                    /// Tensor map type : Read or Write
                    pub enum Tensor_MapType {
                        /// map Tensor for writing data
                        MAP_TENSOR_WRITE = 0,
                        /// map Tensor for writing data
                        MAP_TENSOR_READ = 1,
                    }
                    #[automatically_derived]
                    impl ::core::clone::Clone for Tensor_MapType {
                        #[inline]
                        fn clone(&self) -> Tensor_MapType {
                            match self {
                                Tensor_MapType::MAP_TENSOR_WRITE => {
                                    Tensor_MapType::MAP_TENSOR_WRITE
                                }
                                Tensor_MapType::MAP_TENSOR_READ => {
                                    Tensor_MapType::MAP_TENSOR_READ
                                }
                            }
                        }
                    }
                    #[automatically_derived]
                    impl ::core::hash::Hash for Tensor_MapType {
                        #[inline]
                        fn hash<__H: ::core::hash::Hasher>(
                            &self,
                            state: &mut __H,
                        ) -> () {
                            let __self_discr = ::core::intrinsics::discriminant_value(
                                self,
                            );
                            ::core::hash::Hash::hash(&__self_discr, state)
                        }
                    }
                    #[automatically_derived]
                    impl ::core::marker::StructuralPartialEq for Tensor_MapType {}
                    #[automatically_derived]
                    impl ::core::cmp::PartialEq for Tensor_MapType {
                        #[inline]
                        fn eq(&self, other: &Tensor_MapType) -> bool {
                            let __self_discr = ::core::intrinsics::discriminant_value(
                                self,
                            );
                            let __arg1_discr = ::core::intrinsics::discriminant_value(
                                other,
                            );
                            __self_discr == __arg1_discr
                        }
                    }
                    #[automatically_derived]
                    impl ::core::cmp::Eq for Tensor_MapType {
                        #[inline]
                        #[doc(hidden)]
                        #[coverage(off)]
                        fn assert_receiver_is_total_eq(&self) -> () {}
                    }
                    impl Tensor {
                        ///autocxx bindings couldn't be generated: Function operator_equals has a reference return value, but >1 input reference parameters, so the lifetime of the output reference cannot be deduced.
                        fn operator_equals(_uhoh: autocxx::BindingGenerationFailure) {}
                        ///autocxx bindings couldn't be generated: autocxx does not know how to generate bindings to operator=
                        fn operator_equals1(_uhoh: autocxx::BindingGenerationFailure) {}
                        ///autocxx bindings couldn't be generated: Problem handling function argument shape: A C++ std::vector was found containing some type that cxx can't accommodate as a vector element (int)
                        fn createDevice(_uhoh: autocxx::BindingGenerationFailure) {}
                        ///autocxx bindings couldn't be generated: Problem handling function argument shape: A C++ std::vector was found containing some type that cxx can't accommodate as a vector element (int)
                        fn create(_uhoh: autocxx::BindingGenerationFailure) {}
                        /** @brief copy tensor.
 @param src     tensor
 @param deepCopy whether create new content and copy, currently only support deepCopy = false*/
                        pub unsafe fn clone(
                            src: *const root::MNN::Tensor,
                            deepCopy: bool,
                        ) -> *mut root::MNN::Tensor {
                            cxxbridge::clone_autocxx_wrapper_0xecc3669dd67dbc04(
                                src,
                                deepCopy,
                            )
                        }
                        /** @brief delete tensor.
 @param src     tensor*/
                        pub unsafe fn destroy(tensor: *mut root::MNN::Tensor) {
                            cxxbridge::destroy_autocxx_wrapper_0xecc3669dd67dbc04(tensor)
                        }
                        /** @brief create HOST tensor from DEVICE tensor, with or without data copying.
 @param deviceTensor  given device tensor.
 @param copyData      copy data or not.
 @return created host tensor.*/
                        pub unsafe fn createHostTensorFromDevice(
                            deviceTensor: *const root::MNN::Tensor,
                            copyData: bool,
                        ) -> *mut root::MNN::Tensor {
                            cxxbridge::createHostTensorFromDevice_autocxx_wrapper_0xecc3669dd67dbc04(
                                deviceTensor,
                                copyData,
                            )
                        }
                        /** @brief get data type.
 @return data type.*/
                        pub fn getType<'a>(
                            self: &'a root::MNN::Tensor,
                        ) -> impl autocxx::moveit::new::New<
                            Output = root::halide_type_t,
                        > + 'a {
                            unsafe {
                                autocxx::moveit::new::by_raw(move |placement_return_type| {
                                    let placement_return_type = placement_return_type
                                        .get_unchecked_mut()
                                        .as_mut_ptr();
                                    cxxbridge::getType_autocxx_wrapper_0xecc3669dd67dbc04(
                                        self,
                                        placement_return_type,
                                    )
                                })
                            }
                        }
                        ///autocxx bindings couldn't be generated: A C++ std::vector was found containing some type that cxx can't accommodate as a vector element (int)
                        fn shape(_uhoh: autocxx::BindingGenerationFailure) {}
                        /** @brief create a tensor with dimension size and type without acquire memory for data.
 @param dimSize   dimension size.
 @param type      dimension type.*/
                        pub fn new(
                            dimSize: autocxx::c_int,
                            type_: root::MNN::Tensor_DimensionType,
                        ) -> impl autocxx::moveit::new::New<Output = Self> {
                            unsafe {
                                autocxx::moveit::new::by_raw(move |this| {
                                    let this = this.get_unchecked_mut().as_mut_ptr();
                                    cxxbridge::new_autocxx_autocxx_wrapper_0xecc3669dd67dbc04(
                                        this,
                                        dimSize,
                                        type_,
                                    )
                                })
                            }
                        }
                        /** @brief create a tensor with same shape as given tensor.
 @param tensor        shape provider.
 @param type          dimension type.
 @param allocMemory   acquire memory for data or not.
 @warning tensor data won't be copied.*/
                        pub unsafe fn new1(
                            tensor: *const root::MNN::Tensor,
                            type_: root::MNN::Tensor_DimensionType,
                            allocMemory: bool,
                        ) -> impl autocxx::moveit::new::New<Output = Self> {
                            autocxx::moveit::new::by_raw(move |this| {
                                let this = this.get_unchecked_mut().as_mut_ptr();
                                cxxbridge::new1_autocxx_wrapper_0xecc3669dd67dbc04(
                                    this,
                                    tensor,
                                    type_,
                                    allocMemory,
                                )
                            })
                        }
                        ///autocxx bindings couldn't be generated: This method is private
                        fn new2(_uhoh: autocxx::BindingGenerationFailure) {}
                        ///autocxx bindings couldn't be generated: This function was marked =delete
                        fn new3(_uhoh: autocxx::BindingGenerationFailure) {}
                        ///autocxx bindings couldn't be generated: This function was marked =delete
                        fn new4(_uhoh: autocxx::BindingGenerationFailure) {}
                    }
                    unsafe impl autocxx::moveit::MakeCppStorage for root::MNN::Tensor {
                        unsafe fn allocate_uninitialized_cpp_storage() -> *mut root::MNN::Tensor {
                            cxxbridge::Tensor_autocxx_alloc_autocxx_wrapper_0xecc3669dd67dbc04()
                        }
                        unsafe fn free_uninitialized_cpp_storage(
                            arg0: *mut root::MNN::Tensor,
                        ) {
                            cxxbridge::Tensor_autocxx_free_autocxx_wrapper_0xecc3669dd67dbc04(
                                arg0,
                            )
                        }
                    }
                    impl Drop for root::MNN::Tensor {
                        /// deinitializer
                        fn drop(self: &mut root::MNN::Tensor) {
                            unsafe {
                                cxxbridge::Tensor_destructor_autocxx_wrapper_0xecc3669dd67dbc04(
                                    self,
                                )
                            }
                        }
                    }
                    #[allow(unused_imports)]
                    use self::super::super::super::{cxxbridge, ToCppString};
                    #[allow(unused_imports)]
                    use self::super::super::root;
                }
                #[allow(unused_imports)]
                use self::super::super::{cxxbridge, ToCppString};
                #[allow(unused_imports)]
                use self::super::root;
            }
        }
        #[deny(improper_ctypes, improper_ctypes_definitions)]
        #[allow(clippy::unknown_clippy_lints)]
        #[allow(
            non_camel_case_types,
            non_snake_case,
            unused_unsafe,
            clippy::extra_unused_type_parameters,
            clippy::items_after_statements,
            clippy::no_effect_underscore_binding,
            clippy::ptr_as_ptr,
            clippy::ref_as_ptr,
            clippy::upper_case_acronyms,
            clippy::use_self,
        )]
        mod cxxbridge {
            pub fn autocxx_make_string_0xecc3669dd67dbc04(
                str_: &str,
            ) -> ::cxx::UniquePtr<::cxx::CxxString> {
                extern "C" {
                    #[link_name = "cxxbridge1$autocxx_make_string_0xecc3669dd67dbc04"]
                    fn __autocxx_make_string_0xecc3669dd67dbc04(
                        str_: ::cxx::private::RustStr,
                    ) -> *mut ::cxx::CxxString;
                }
                unsafe {
                    ::cxx::UniquePtr::from_raw(
                        __autocxx_make_string_0xecc3669dd67dbc04(
                            ::cxx::private::RustStr::from(str_),
                        ),
                    )
                }
            }
            pub unsafe fn Tensor_autocxx_alloc_autocxx_wrapper_0xecc3669dd67dbc04() -> *mut Tensor {
                extern "C" {
                    #[link_name = "cxxbridge1$Tensor_autocxx_alloc_autocxx_wrapper_0xecc3669dd67dbc04"]
                    fn __Tensor_autocxx_alloc_autocxx_wrapper_0xecc3669dd67dbc04() -> *mut ::cxx::core::ffi::c_void;
                }
                unsafe {
                    __Tensor_autocxx_alloc_autocxx_wrapper_0xecc3669dd67dbc04().cast()
                }
            }
            pub unsafe fn Tensor_autocxx_free_autocxx_wrapper_0xecc3669dd67dbc04(
                arg0: *mut Tensor,
            ) {
                extern "C" {
                    #[link_name = "cxxbridge1$Tensor_autocxx_free_autocxx_wrapper_0xecc3669dd67dbc04"]
                    fn __Tensor_autocxx_free_autocxx_wrapper_0xecc3669dd67dbc04(
                        arg0: *mut ::cxx::core::ffi::c_void,
                    );
                }
                unsafe {
                    __Tensor_autocxx_free_autocxx_wrapper_0xecc3669dd67dbc04(arg0.cast())
                }
            }
            /** data container.
 data for host tensor is saved in `host` field. its memory is allocated malloc directly.
 data for device tensor is saved in `deviceId` field. its memory is allocated by session's backend.
 usually, device tensors are created by engine (like net, session).
 meanwhile, host tensors could be created by engine or user.*/
            pub type Tensor = super::bindgen::root::MNN::Tensor;
            /** @brief copy tensor.
 @param src     tensor
 @param deepCopy whether create new content and copy, currently only support deepCopy = false*/
            pub unsafe fn clone_autocxx_wrapper_0xecc3669dd67dbc04(
                src: *const Tensor,
                deepCopy: bool,
            ) -> *mut Tensor {
                extern "C" {
                    #[link_name = "cxxbridge1$clone_autocxx_wrapper_0xecc3669dd67dbc04"]
                    fn __clone_autocxx_wrapper_0xecc3669dd67dbc04(
                        src: *const ::cxx::core::ffi::c_void,
                        deepCopy: bool,
                    ) -> *mut ::cxx::core::ffi::c_void;
                }
                unsafe {
                    __clone_autocxx_wrapper_0xecc3669dd67dbc04(src.cast(), deepCopy)
                        .cast()
                }
            }
            /** @brief delete tensor.
 @param src     tensor*/
            pub unsafe fn destroy_autocxx_wrapper_0xecc3669dd67dbc04(
                tensor: *mut Tensor,
            ) {
                extern "C" {
                    #[link_name = "cxxbridge1$destroy_autocxx_wrapper_0xecc3669dd67dbc04"]
                    fn __destroy_autocxx_wrapper_0xecc3669dd67dbc04(
                        tensor: *mut ::cxx::core::ffi::c_void,
                    );
                }
                unsafe { __destroy_autocxx_wrapper_0xecc3669dd67dbc04(tensor.cast()) }
            }
            impl Tensor {
                /** @brief for DEVICE tensor, copy data from given host tensor.
 @param hostTensor    host tensor, the data provider.
 @return true for DEVICE tensor, and false for HOST tensor.*/
                pub unsafe fn copyFromHostTensor(
                    self: ::cxx::core::pin::Pin<&mut Self>,
                    hostTensor: *const Tensor,
                ) -> bool {
                    extern "C" {
                        #[link_name = "MNN$cxxbridge1$Tensor$copyFromHostTensor"]
                        fn __copyFromHostTensor(
                            _: ::cxx::core::pin::Pin<&mut Tensor>,
                            hostTensor: *const ::cxx::core::ffi::c_void,
                        ) -> bool;
                    }
                    unsafe { __copyFromHostTensor(self, hostTensor.cast()) }
                }
            }
            impl Tensor {
                /** @brief for DEVICE tensor, copy data to given host tensor.
 @param hostTensor    host tensor, the data consumer.
 @return true for DEVICE tensor, and false for HOST tensor.*/
                pub unsafe fn copyToHostTensor(&self, hostTensor: *mut Tensor) -> bool {
                    extern "C" {
                        #[link_name = "MNN$cxxbridge1$Tensor$copyToHostTensor"]
                        fn __copyToHostTensor(
                            _: &Tensor,
                            hostTensor: *mut ::cxx::core::ffi::c_void,
                        ) -> bool;
                    }
                    unsafe { __copyToHostTensor(self, hostTensor.cast()) }
                }
            }
            /** @brief create HOST tensor from DEVICE tensor, with or without data copying.
 @param deviceTensor  given device tensor.
 @param copyData      copy data or not.
 @return created host tensor.*/
            pub unsafe fn createHostTensorFromDevice_autocxx_wrapper_0xecc3669dd67dbc04(
                deviceTensor: *const Tensor,
                copyData: bool,
            ) -> *mut Tensor {
                extern "C" {
                    #[link_name = "cxxbridge1$createHostTensorFromDevice_autocxx_wrapper_0xecc3669dd67dbc04"]
                    fn __createHostTensorFromDevice_autocxx_wrapper_0xecc3669dd67dbc04(
                        deviceTensor: *const ::cxx::core::ffi::c_void,
                        copyData: bool,
                    ) -> *mut ::cxx::core::ffi::c_void;
                }
                unsafe {
                    __createHostTensorFromDevice_autocxx_wrapper_0xecc3669dd67dbc04(
                            deviceTensor.cast(),
                            copyData,
                        )
                        .cast()
                }
            }
            impl Tensor {
                pub fn buffer(&self) -> &halide_buffer_t {
                    extern "C" {
                        #[link_name = "MNN$cxxbridge1$Tensor$buffer"]
                        fn __buffer(_: &Tensor) -> *const ::cxx::core::ffi::c_void;
                    }
                    unsafe { &*__buffer(self).cast() }
                }
            }
            impl<'a> Tensor {
                pub fn buffer1(
                    self: ::cxx::core::pin::Pin<&'a mut Self>,
                ) -> ::cxx::core::pin::Pin<&'a mut halide_buffer_t> {
                    extern "C" {
                        #[link_name = "MNN$cxxbridge1$Tensor$buffer1"]
                        fn __buffer1<'a>(
                            _: ::cxx::core::pin::Pin<&'a mut Tensor>,
                        ) -> *mut ::cxx::core::ffi::c_void;
                    }
                    unsafe {
                        ::cxx::core::pin::Pin::new_unchecked(
                            &mut *__buffer1(self).cast(),
                        )
                    }
                }
            }
            impl Tensor {
                /** @brief get dimension type.
 @return dimension type.*/
                pub fn getDimensionType(&self) -> Tensor_DimensionType {
                    extern "C" {
                        #[link_name = "MNN$cxxbridge1$Tensor$getDimensionType"]
                        fn __getDimensionType(
                            _: &Tensor,
                            __return: *mut Tensor_DimensionType,
                        );
                    }
                    unsafe {
                        let mut __return = ::cxx::core::mem::MaybeUninit::<
                            Tensor_DimensionType,
                        >::uninit();
                        __getDimensionType(self, __return.as_mut_ptr());
                        __return.assume_init()
                    }
                }
            }
            impl Tensor {
                /** @brief handle data type. used when data type code is halide_type_handle.
 @return handle data type.*/
                pub fn getHandleDataType(&self) -> Tensor_HandleDataType {
                    extern "C" {
                        #[link_name = "MNN$cxxbridge1$Tensor$getHandleDataType"]
                        fn __getHandleDataType(
                            _: &Tensor,
                            __return: *mut Tensor_HandleDataType,
                        );
                    }
                    unsafe {
                        let mut __return = ::cxx::core::mem::MaybeUninit::<
                            Tensor_HandleDataType,
                        >::uninit();
                        __getHandleDataType(self, __return.as_mut_ptr());
                        __return.assume_init()
                    }
                }
            }
            impl Tensor {
                /** @brief set data type.
 @param type data type defined in 'Type_generated.h'.*/
                pub fn setType(self: ::cxx::core::pin::Pin<&mut Self>, type_: c_int) {
                    extern "C" {
                        #[link_name = "MNN$cxxbridge1$Tensor$setType"]
                        fn __setType(
                            _: ::cxx::core::pin::Pin<&mut Tensor>,
                            type_: *mut c_int,
                        );
                    }
                    unsafe {
                        let mut type_ = ::cxx::core::mem::MaybeUninit::new(type_);
                        __setType(self, type_.as_mut_ptr())
                    }
                }
            }
            /** @brief get data type.
 @return data type.*/
            pub unsafe fn getType_autocxx_wrapper_0xecc3669dd67dbc04(
                autocxx_gen_this: &Tensor,
                placement_return_type: *mut halide_type_t,
            ) {
                extern "C" {
                    #[link_name = "cxxbridge1$getType_autocxx_wrapper_0xecc3669dd67dbc04"]
                    fn __getType_autocxx_wrapper_0xecc3669dd67dbc04(
                        autocxx_gen_this: *const ::cxx::core::ffi::c_void,
                        placement_return_type: *mut ::cxx::core::ffi::c_void,
                    );
                }
                unsafe {
                    __getType_autocxx_wrapper_0xecc3669dd67dbc04(
                        autocxx_gen_this as *const Tensor
                            as *const ::cxx::core::ffi::c_void,
                        placement_return_type.cast(),
                    )
                }
            }
            impl Tensor {
                /** @brief visit device memory.
 @return device data ID. what the ID means varies between backends.*/
                pub fn deviceId(&self) -> u64 {
                    extern "C" {
                        #[link_name = "MNN$cxxbridge1$Tensor$deviceId"]
                        fn __deviceId(_: &Tensor) -> u64;
                    }
                    unsafe { __deviceId(self) }
                }
            }
            impl Tensor {
                pub fn dimensions(&self) -> c_int {
                    extern "C" {
                        #[link_name = "MNN$cxxbridge1$Tensor$dimensions"]
                        fn __dimensions(_: &Tensor, __return: *mut c_int);
                    }
                    unsafe {
                        let mut __return = ::cxx::core::mem::MaybeUninit::<
                            c_int,
                        >::uninit();
                        __dimensions(self, __return.as_mut_ptr());
                        __return.assume_init()
                    }
                }
            }
            impl Tensor {
                /** @brief calculate number of bytes needed to store data taking reordering flag into account.
 @return bytes needed to store data*/
                pub fn size(&self) -> c_int {
                    extern "C" {
                        #[link_name = "MNN$cxxbridge1$Tensor$size"]
                        fn __size(_: &Tensor, __return: *mut c_int);
                    }
                    unsafe {
                        let mut __return = ::cxx::core::mem::MaybeUninit::<
                            c_int,
                        >::uninit();
                        __size(self, __return.as_mut_ptr());
                        __return.assume_init()
                    }
                }
            }
            impl Tensor {
                pub fn usize(&self) -> usize {
                    extern "C" {
                        #[link_name = "MNN$cxxbridge1$Tensor$usize"]
                        fn __usize(_: &Tensor) -> usize;
                    }
                    unsafe { __usize(self) }
                }
            }
            impl Tensor {
                /** @brief calculate number of elements needed to store data taking reordering flag into account.
 @return elements needed to store data*/
                pub fn elementSize(&self) -> c_int {
                    extern "C" {
                        #[link_name = "MNN$cxxbridge1$Tensor$elementSize"]
                        fn __elementSize(_: &Tensor, __return: *mut c_int);
                    }
                    unsafe {
                        let mut __return = ::cxx::core::mem::MaybeUninit::<
                            c_int,
                        >::uninit();
                        __elementSize(self, __return.as_mut_ptr());
                        __return.assume_init()
                    }
                }
            }
            impl Tensor {
                pub fn width(&self) -> c_int {
                    extern "C" {
                        #[link_name = "MNN$cxxbridge1$Tensor$width"]
                        fn __width(_: &Tensor, __return: *mut c_int);
                    }
                    unsafe {
                        let mut __return = ::cxx::core::mem::MaybeUninit::<
                            c_int,
                        >::uninit();
                        __width(self, __return.as_mut_ptr());
                        __return.assume_init()
                    }
                }
            }
            impl Tensor {
                pub fn height(&self) -> c_int {
                    extern "C" {
                        #[link_name = "MNN$cxxbridge1$Tensor$height"]
                        fn __height(_: &Tensor, __return: *mut c_int);
                    }
                    unsafe {
                        let mut __return = ::cxx::core::mem::MaybeUninit::<
                            c_int,
                        >::uninit();
                        __height(self, __return.as_mut_ptr());
                        __return.assume_init()
                    }
                }
            }
            impl Tensor {
                pub fn channel(&self) -> c_int {
                    extern "C" {
                        #[link_name = "MNN$cxxbridge1$Tensor$channel"]
                        fn __channel(_: &Tensor, __return: *mut c_int);
                    }
                    unsafe {
                        let mut __return = ::cxx::core::mem::MaybeUninit::<
                            c_int,
                        >::uninit();
                        __channel(self, __return.as_mut_ptr());
                        __return.assume_init()
                    }
                }
            }
            impl Tensor {
                pub fn batch(&self) -> c_int {
                    extern "C" {
                        #[link_name = "MNN$cxxbridge1$Tensor$batch"]
                        fn __batch(_: &Tensor, __return: *mut c_int);
                    }
                    unsafe {
                        let mut __return = ::cxx::core::mem::MaybeUninit::<
                            c_int,
                        >::uninit();
                        __batch(self, __return.as_mut_ptr());
                        __return.assume_init()
                    }
                }
            }
            impl Tensor {
                pub fn stride(&self, index: c_int) -> c_int {
                    extern "C" {
                        #[link_name = "MNN$cxxbridge1$Tensor$stride"]
                        fn __stride(_: &Tensor, index: *mut c_int, __return: *mut c_int);
                    }
                    unsafe {
                        let mut index = ::cxx::core::mem::MaybeUninit::new(index);
                        let mut __return = ::cxx::core::mem::MaybeUninit::<
                            c_int,
                        >::uninit();
                        __stride(self, index.as_mut_ptr(), __return.as_mut_ptr());
                        __return.assume_init()
                    }
                }
            }
            impl Tensor {
                pub fn length(&self, index: c_int) -> c_int {
                    extern "C" {
                        #[link_name = "MNN$cxxbridge1$Tensor$length"]
                        fn __length(_: &Tensor, index: *mut c_int, __return: *mut c_int);
                    }
                    unsafe {
                        let mut index = ::cxx::core::mem::MaybeUninit::new(index);
                        let mut __return = ::cxx::core::mem::MaybeUninit::<
                            c_int,
                        >::uninit();
                        __length(self, index.as_mut_ptr(), __return.as_mut_ptr());
                        __return.assume_init()
                    }
                }
            }
            impl Tensor {
                pub fn setStride(
                    self: ::cxx::core::pin::Pin<&mut Self>,
                    index: c_int,
                    stride: c_int,
                ) {
                    extern "C" {
                        #[link_name = "MNN$cxxbridge1$Tensor$setStride"]
                        fn __setStride(
                            _: ::cxx::core::pin::Pin<&mut Tensor>,
                            index: *mut c_int,
                            stride: *mut c_int,
                        );
                    }
                    unsafe {
                        let mut index = ::cxx::core::mem::MaybeUninit::new(index);
                        let mut stride = ::cxx::core::mem::MaybeUninit::new(stride);
                        __setStride(self, index.as_mut_ptr(), stride.as_mut_ptr())
                    }
                }
            }
            impl Tensor {
                pub fn setLength(
                    self: ::cxx::core::pin::Pin<&mut Self>,
                    index: c_int,
                    length: c_int,
                ) {
                    extern "C" {
                        #[link_name = "MNN$cxxbridge1$Tensor$setLength"]
                        fn __setLength(
                            _: ::cxx::core::pin::Pin<&mut Tensor>,
                            index: *mut c_int,
                            length: *mut c_int,
                        );
                    }
                    unsafe {
                        let mut index = ::cxx::core::mem::MaybeUninit::new(index);
                        let mut length = ::cxx::core::mem::MaybeUninit::new(length);
                        __setLength(self, index.as_mut_ptr(), length.as_mut_ptr())
                    }
                }
            }
            impl Tensor {
                /** @brief For GPU and Other Device, get memory directly, see MNNSharedContext for detail
 @return Success or not. If type != tensor's backend's type or type is cpu , return false*/
                pub unsafe fn getDeviceInfo(
                    &self,
                    dst: *mut c_void,
                    forwardType: c_int,
                ) -> bool {
                    extern "C" {
                        #[link_name = "MNN$cxxbridge1$Tensor$getDeviceInfo"]
                        fn __getDeviceInfo(
                            _: &Tensor,
                            dst: *mut ::cxx::core::ffi::c_void,
                            forwardType: *mut c_int,
                        ) -> bool;
                    }
                    unsafe {
                        let mut forwardType = ::cxx::core::mem::MaybeUninit::new(
                            forwardType,
                        );
                        __getDeviceInfo(self, dst.cast(), forwardType.as_mut_ptr())
                    }
                }
            }
            impl Tensor {
                /// @brief print tensor data. for DEBUG use only.
                pub fn print(&self) {
                    extern "C" {
                        #[link_name = "MNN$cxxbridge1$Tensor$print"]
                        fn __print(_: &Tensor);
                    }
                    unsafe { __print(self) }
                }
            }
            impl Tensor {
                ///@brief print tensor shape
                pub fn printShape(&self) {
                    extern "C" {
                        #[link_name = "MNN$cxxbridge1$Tensor$printShape"]
                        fn __printShape(_: &Tensor);
                    }
                    unsafe { __printShape(self) }
                }
            }
            impl Tensor {
                /// @brief map/umap GPU Tensor, to get host ptr
                pub fn map(
                    self: ::cxx::core::pin::Pin<&mut Self>,
                    mtype: Tensor_MapType,
                    dtype: Tensor_DimensionType,
                ) -> *mut c_void {
                    extern "C" {
                        #[link_name = "MNN$cxxbridge1$Tensor$map"]
                        fn __map(
                            _: ::cxx::core::pin::Pin<&mut Tensor>,
                            mtype: *mut Tensor_MapType,
                            dtype: *mut Tensor_DimensionType,
                        ) -> *mut ::cxx::core::ffi::c_void;
                    }
                    unsafe {
                        let mut mtype = ::cxx::core::mem::MaybeUninit::new(mtype);
                        let mut dtype = ::cxx::core::mem::MaybeUninit::new(dtype);
                        __map(self, mtype.as_mut_ptr(), dtype.as_mut_ptr()).cast()
                    }
                }
            }
            impl Tensor {
                pub unsafe fn unmap(
                    self: ::cxx::core::pin::Pin<&mut Self>,
                    mtype: Tensor_MapType,
                    dtype: Tensor_DimensionType,
                    mapPtr: *mut c_void,
                ) {
                    extern "C" {
                        #[link_name = "MNN$cxxbridge1$Tensor$unmap"]
                        fn __unmap(
                            _: ::cxx::core::pin::Pin<&mut Tensor>,
                            mtype: *mut Tensor_MapType,
                            dtype: *mut Tensor_DimensionType,
                            mapPtr: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        let mut mtype = ::cxx::core::mem::MaybeUninit::new(mtype);
                        let mut dtype = ::cxx::core::mem::MaybeUninit::new(dtype);
                        __unmap(
                            self,
                            mtype.as_mut_ptr(),
                            dtype.as_mut_ptr(),
                            mapPtr.cast(),
                        )
                    }
                }
            }
            impl Tensor {
                /** @brief wait until the tensor is ready to read / write
 @param mtype wait for read or write
 @param finish wait for command flush or finish*/
                pub fn wait(
                    self: ::cxx::core::pin::Pin<&mut Self>,
                    mtype: Tensor_MapType,
                    finish: bool,
                ) -> c_int {
                    extern "C" {
                        #[link_name = "MNN$cxxbridge1$Tensor$wait"]
                        fn __wait(
                            _: ::cxx::core::pin::Pin<&mut Tensor>,
                            mtype: *mut Tensor_MapType,
                            finish: bool,
                            __return: *mut c_int,
                        );
                    }
                    unsafe {
                        let mut mtype = ::cxx::core::mem::MaybeUninit::new(mtype);
                        let mut __return = ::cxx::core::mem::MaybeUninit::<
                            c_int,
                        >::uninit();
                        __wait(self, mtype.as_mut_ptr(), finish, __return.as_mut_ptr());
                        __return.assume_init()
                    }
                }
            }
            impl Tensor {
                /// @brief set GPU tensor device ptr, and inform memory type
                pub unsafe fn setDevicePtr(
                    self: ::cxx::core::pin::Pin<&mut Self>,
                    devicePtr: *const c_void,
                    memoryType: c_int,
                ) -> bool {
                    extern "C" {
                        #[link_name = "MNN$cxxbridge1$Tensor$setDevicePtr"]
                        fn __setDevicePtr(
                            _: ::cxx::core::pin::Pin<&mut Tensor>,
                            devicePtr: *const ::cxx::core::ffi::c_void,
                            memoryType: *mut c_int,
                        ) -> bool;
                    }
                    unsafe {
                        let mut memoryType = ::cxx::core::mem::MaybeUninit::new(
                            memoryType,
                        );
                        __setDevicePtr(self, devicePtr.cast(), memoryType.as_mut_ptr())
                    }
                }
            }
            /** @brief create a tensor with dimension size and type without acquire memory for data.
 @param dimSize   dimension size.
 @param type      dimension type.*/
            pub unsafe fn new_autocxx_autocxx_wrapper_0xecc3669dd67dbc04(
                autocxx_gen_this: *mut Tensor,
                dimSize: c_int,
                type_: Tensor_DimensionType,
            ) {
                extern "C" {
                    #[link_name = "cxxbridge1$new_autocxx_autocxx_wrapper_0xecc3669dd67dbc04"]
                    fn __new_autocxx_autocxx_wrapper_0xecc3669dd67dbc04(
                        autocxx_gen_this: *mut ::cxx::core::ffi::c_void,
                        dimSize: *mut c_int,
                        type_: *mut Tensor_DimensionType,
                    );
                }
                unsafe {
                    let mut dimSize = ::cxx::core::mem::MaybeUninit::new(dimSize);
                    let mut type_ = ::cxx::core::mem::MaybeUninit::new(type_);
                    __new_autocxx_autocxx_wrapper_0xecc3669dd67dbc04(
                        autocxx_gen_this.cast(),
                        dimSize.as_mut_ptr(),
                        type_.as_mut_ptr(),
                    )
                }
            }
            /** @brief create a tensor with same shape as given tensor.
 @param tensor        shape provider.
 @param type          dimension type.
 @param allocMemory   acquire memory for data or not.
 @warning tensor data won't be copied.*/
            pub unsafe fn new1_autocxx_wrapper_0xecc3669dd67dbc04(
                autocxx_gen_this: *mut Tensor,
                tensor: *const Tensor,
                type_: Tensor_DimensionType,
                allocMemory: bool,
            ) {
                extern "C" {
                    #[link_name = "cxxbridge1$new1_autocxx_wrapper_0xecc3669dd67dbc04"]
                    fn __new1_autocxx_wrapper_0xecc3669dd67dbc04(
                        autocxx_gen_this: *mut ::cxx::core::ffi::c_void,
                        tensor: *const ::cxx::core::ffi::c_void,
                        type_: *mut Tensor_DimensionType,
                        allocMemory: bool,
                    );
                }
                unsafe {
                    let mut type_ = ::cxx::core::mem::MaybeUninit::new(type_);
                    __new1_autocxx_wrapper_0xecc3669dd67dbc04(
                        autocxx_gen_this.cast(),
                        tensor.cast(),
                        type_.as_mut_ptr(),
                        allocMemory,
                    )
                }
            }
            /// deinitializer
            pub unsafe fn Tensor_destructor_autocxx_wrapper_0xecc3669dd67dbc04(
                autocxx_gen_this: *mut Tensor,
            ) {
                extern "C" {
                    #[link_name = "cxxbridge1$Tensor_destructor_autocxx_wrapper_0xecc3669dd67dbc04"]
                    fn __Tensor_destructor_autocxx_wrapper_0xecc3669dd67dbc04(
                        autocxx_gen_this: *mut ::cxx::core::ffi::c_void,
                    );
                }
                unsafe {
                    __Tensor_destructor_autocxx_wrapper_0xecc3669dd67dbc04(
                        autocxx_gen_this.cast(),
                    )
                }
            }
            /** \file

 This file declares the routines used by Halide internally in its
 runtime. On platforms that support weak linking, these can be
 replaced with user-defined versions by defining an extern "C"
 function with the same name and signature.

 When doing Just In Time (JIT) compilation methods on the Func being
 compiled must be called instead. The corresponding methods are
 documented below.

 All of these functions take a "void *user_context" parameter as their
 first argument; if the Halide kernel that calls back to any of these
 functions has been compiled with the UserContext feature set on its Target,
 then the value of that pointer passed from the code that calls the
 Halide kernel is piped through to the function.

 Some of these are also useful to call when using the default
 implementation. E.g. halide_shutdown_thread_pool.

 Note that even on platforms with weak linking, some linker setups
 may not respect the override you provide. E.g. if the override is
 in a shared library and the halide object files are linked directly
 into the output, the builtin versions of the runtime functions will
 be called. See your linker documentation for more details. On
 Linux, LD_DYNAMIC_WEAK=1 may help.
*/
            pub type halide_buffer_t = super::bindgen::root::halide_buffer_t;
            /// dimension type used to create tensor
            pub type Tensor_DimensionType = super::bindgen::root::MNN::Tensor_DimensionType;
            /// handle type
            pub type Tensor_HandleDataType = super::bindgen::root::MNN::Tensor_HandleDataType;
            /** A runtime tag for a type in the halide type system. Can be ints,
 unsigned ints, or floats of various bit-widths (the 'bits'
 field). Can also be vectors of the same (by setting the 'lanes'
 field to something larger than one). This struct should be
 exactly 32-bits in size.*/
            pub type halide_type_t = super::bindgen::root::halide_type_t;
            /// Tensor map type : Read or Write
            pub type Tensor_MapType = super::bindgen::root::MNN::Tensor_MapType;
            pub unsafe fn halide_buffer_t_autocxx_alloc_autocxx_wrapper_0xecc3669dd67dbc04() -> *mut halide_buffer_t {
                extern "C" {
                    #[link_name = "cxxbridge1$halide_buffer_t_autocxx_alloc_autocxx_wrapper_0xecc3669dd67dbc04"]
                    fn __halide_buffer_t_autocxx_alloc_autocxx_wrapper_0xecc3669dd67dbc04() -> *mut ::cxx::core::ffi::c_void;
                }
                unsafe {
                    __halide_buffer_t_autocxx_alloc_autocxx_wrapper_0xecc3669dd67dbc04()
                        .cast()
                }
            }
            pub unsafe fn halide_buffer_t_autocxx_free_autocxx_wrapper_0xecc3669dd67dbc04(
                arg0: *mut halide_buffer_t,
            ) {
                extern "C" {
                    #[link_name = "cxxbridge1$halide_buffer_t_autocxx_free_autocxx_wrapper_0xecc3669dd67dbc04"]
                    fn __halide_buffer_t_autocxx_free_autocxx_wrapper_0xecc3669dd67dbc04(
                        arg0: *mut ::cxx::core::ffi::c_void,
                    );
                }
                unsafe {
                    __halide_buffer_t_autocxx_free_autocxx_wrapper_0xecc3669dd67dbc04(
                        arg0.cast(),
                    )
                }
            }
            ///Synthesized move constructor.
            pub unsafe fn halide_buffer_t_new_synthetic_move_ctor_0xecc3669dd67dbc04_autocxx_wrapper_0xecc3669dd67dbc04(
                autocxx_gen_this: *mut halide_buffer_t,
                other: *mut halide_buffer_t,
            ) {
                extern "C" {
                    #[link_name = "cxxbridge1$halide_buffer_t_new_synthetic_move_ctor_0xecc3669dd67dbc04_autocxx_wrapper_0xecc3669dd67dbc04"]
                    fn __halide_buffer_t_new_synthetic_move_ctor_0xecc3669dd67dbc04_autocxx_wrapper_0xecc3669dd67dbc04(
                        autocxx_gen_this: *mut ::cxx::core::ffi::c_void,
                        other: *mut ::cxx::core::ffi::c_void,
                    );
                }
                unsafe {
                    __halide_buffer_t_new_synthetic_move_ctor_0xecc3669dd67dbc04_autocxx_wrapper_0xecc3669dd67dbc04(
                        autocxx_gen_this.cast(),
                        other.cast(),
                    )
                }
            }
            ///Synthesized copy constructor.
            pub unsafe fn halide_buffer_t_new_synthetic_const_copy_ctor_0xecc3669dd67dbc04_autocxx_wrapper_0xecc3669dd67dbc04(
                autocxx_gen_this: *mut halide_buffer_t,
                other: &halide_buffer_t,
            ) {
                extern "C" {
                    #[link_name = "cxxbridge1$halide_buffer_t_new_synthetic_const_copy_ctor_0xecc3669dd67dbc04_autocxx_wrapper_0xecc3669dd67dbc04"]
                    fn __halide_buffer_t_new_synthetic_const_copy_ctor_0xecc3669dd67dbc04_autocxx_wrapper_0xecc3669dd67dbc04(
                        autocxx_gen_this: *mut ::cxx::core::ffi::c_void,
                        other: *const ::cxx::core::ffi::c_void,
                    );
                }
                unsafe {
                    __halide_buffer_t_new_synthetic_const_copy_ctor_0xecc3669dd67dbc04_autocxx_wrapper_0xecc3669dd67dbc04(
                        autocxx_gen_this.cast(),
                        other as *const halide_buffer_t
                            as *const ::cxx::core::ffi::c_void,
                    )
                }
            }
            pub unsafe fn halide_type_t_autocxx_alloc_autocxx_wrapper_0xecc3669dd67dbc04() -> *mut halide_type_t {
                extern "C" {
                    #[link_name = "cxxbridge1$halide_type_t_autocxx_alloc_autocxx_wrapper_0xecc3669dd67dbc04"]
                    fn __halide_type_t_autocxx_alloc_autocxx_wrapper_0xecc3669dd67dbc04() -> *mut ::cxx::core::ffi::c_void;
                }
                unsafe {
                    __halide_type_t_autocxx_alloc_autocxx_wrapper_0xecc3669dd67dbc04()
                        .cast()
                }
            }
            pub unsafe fn halide_type_t_autocxx_free_autocxx_wrapper_0xecc3669dd67dbc04(
                arg0: *mut halide_type_t,
            ) {
                extern "C" {
                    #[link_name = "cxxbridge1$halide_type_t_autocxx_free_autocxx_wrapper_0xecc3669dd67dbc04"]
                    fn __halide_type_t_autocxx_free_autocxx_wrapper_0xecc3669dd67dbc04(
                        arg0: *mut ::cxx::core::ffi::c_void,
                    );
                }
                unsafe {
                    __halide_type_t_autocxx_free_autocxx_wrapper_0xecc3669dd67dbc04(
                        arg0.cast(),
                    )
                }
            }
            ///Synthesized move constructor.
            pub unsafe fn new_synthetic_move_ctor_0xecc3669dd67dbc04_autocxx_wrapper_0xecc3669dd67dbc04(
                autocxx_gen_this: *mut halide_type_t,
                other: *mut halide_type_t,
            ) {
                extern "C" {
                    #[link_name = "cxxbridge1$new_synthetic_move_ctor_0xecc3669dd67dbc04_autocxx_wrapper_0xecc3669dd67dbc04"]
                    fn __new_synthetic_move_ctor_0xecc3669dd67dbc04_autocxx_wrapper_0xecc3669dd67dbc04(
                        autocxx_gen_this: *mut ::cxx::core::ffi::c_void,
                        other: *mut ::cxx::core::ffi::c_void,
                    );
                }
                unsafe {
                    __new_synthetic_move_ctor_0xecc3669dd67dbc04_autocxx_wrapper_0xecc3669dd67dbc04(
                        autocxx_gen_this.cast(),
                        other.cast(),
                    )
                }
            }
            ///Synthesized copy constructor.
            pub unsafe fn new_synthetic_const_copy_ctor_0xecc3669dd67dbc04_autocxx_wrapper_0xecc3669dd67dbc04(
                autocxx_gen_this: *mut halide_type_t,
                other: &halide_type_t,
            ) {
                extern "C" {
                    #[link_name = "cxxbridge1$new_synthetic_const_copy_ctor_0xecc3669dd67dbc04_autocxx_wrapper_0xecc3669dd67dbc04"]
                    fn __new_synthetic_const_copy_ctor_0xecc3669dd67dbc04_autocxx_wrapper_0xecc3669dd67dbc04(
                        autocxx_gen_this: *mut ::cxx::core::ffi::c_void,
                        other: *const ::cxx::core::ffi::c_void,
                    );
                }
                unsafe {
                    __new_synthetic_const_copy_ctor_0xecc3669dd67dbc04_autocxx_wrapper_0xecc3669dd67dbc04(
                        autocxx_gen_this.cast(),
                        other as *const halide_type_t as *const ::cxx::core::ffi::c_void,
                    )
                }
            }
            pub type c_int = autocxx::c_int;
            pub type c_void = autocxx::c_void;
            unsafe impl ::cxx::private::UniquePtrTarget for Tensor {
                fn __typename(
                    f: &mut ::cxx::core::fmt::Formatter<'_>,
                ) -> ::cxx::core::fmt::Result {
                    f.write_str("Tensor")
                }
                fn __null() -> ::cxx::core::mem::MaybeUninit<
                    *mut ::cxx::core::ffi::c_void,
                > {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$MNN$Tensor$null"]
                        fn __null(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        );
                    }
                    let mut repr = ::cxx::core::mem::MaybeUninit::uninit();
                    unsafe {
                        __null(&mut repr);
                    }
                    repr
                }
                fn __new(
                    value: Self,
                ) -> ::cxx::core::mem::MaybeUninit<*mut ::cxx::core::ffi::c_void> {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$MNN$Tensor$uninit"]
                        fn __uninit(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        ) -> *mut ::cxx::core::ffi::c_void;
                    }
                    let mut repr = ::cxx::core::mem::MaybeUninit::uninit();
                    unsafe {
                        __uninit(&mut repr).cast::<Tensor>().write(value);
                    }
                    repr
                }
                unsafe fn __raw(
                    raw: *mut Self,
                ) -> ::cxx::core::mem::MaybeUninit<*mut ::cxx::core::ffi::c_void> {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$MNN$Tensor$raw"]
                        fn __raw(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                            raw: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    let mut repr = ::cxx::core::mem::MaybeUninit::uninit();
                    unsafe {
                        __raw(&mut repr, raw.cast());
                    }
                    repr
                }
                unsafe fn __get(
                    repr: ::cxx::core::mem::MaybeUninit<*mut ::cxx::core::ffi::c_void>,
                ) -> *const Self {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$MNN$Tensor$get"]
                        fn __get(
                            this: *const ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        ) -> *const ::cxx::core::ffi::c_void;
                    }
                    unsafe { __get(&repr).cast() }
                }
                unsafe fn __release(
                    mut repr: ::cxx::core::mem::MaybeUninit<
                        *mut ::cxx::core::ffi::c_void,
                    >,
                ) -> *mut Self {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$MNN$Tensor$release"]
                        fn __release(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        ) -> *mut ::cxx::core::ffi::c_void;
                    }
                    unsafe { __release(&mut repr).cast() }
                }
                unsafe fn __drop(
                    mut repr: ::cxx::core::mem::MaybeUninit<
                        *mut ::cxx::core::ffi::c_void,
                    >,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$MNN$Tensor$drop"]
                        fn __drop(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        );
                    }
                    unsafe {
                        __drop(&mut repr);
                    }
                }
            }
            unsafe impl ::cxx::private::SharedPtrTarget for Tensor {
                fn __typename(
                    f: &mut ::cxx::core::fmt::Formatter<'_>,
                ) -> ::cxx::core::fmt::Result {
                    f.write_str("Tensor")
                }
                unsafe fn __null(new: *mut ::cxx::core::ffi::c_void) {
                    extern "C" {
                        #[link_name = "cxxbridge1$shared_ptr$MNN$Tensor$null"]
                        fn __null(new: *mut ::cxx::core::ffi::c_void);
                    }
                    unsafe {
                        __null(new);
                    }
                }
                unsafe fn __new(value: Self, new: *mut ::cxx::core::ffi::c_void) {
                    extern "C" {
                        #[link_name = "cxxbridge1$shared_ptr$MNN$Tensor$uninit"]
                        fn __uninit(
                            new: *mut ::cxx::core::ffi::c_void,
                        ) -> *mut ::cxx::core::ffi::c_void;
                    }
                    unsafe {
                        __uninit(new).cast::<Tensor>().write(value);
                    }
                }
                unsafe fn __clone(
                    this: *const ::cxx::core::ffi::c_void,
                    new: *mut ::cxx::core::ffi::c_void,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$shared_ptr$MNN$Tensor$clone"]
                        fn __clone(
                            this: *const ::cxx::core::ffi::c_void,
                            new: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __clone(this, new);
                    }
                }
                unsafe fn __get(this: *const ::cxx::core::ffi::c_void) -> *const Self {
                    extern "C" {
                        #[link_name = "cxxbridge1$shared_ptr$MNN$Tensor$get"]
                        fn __get(
                            this: *const ::cxx::core::ffi::c_void,
                        ) -> *const ::cxx::core::ffi::c_void;
                    }
                    unsafe { __get(this).cast() }
                }
                unsafe fn __drop(this: *mut ::cxx::core::ffi::c_void) {
                    extern "C" {
                        #[link_name = "cxxbridge1$shared_ptr$MNN$Tensor$drop"]
                        fn __drop(this: *mut ::cxx::core::ffi::c_void);
                    }
                    unsafe {
                        __drop(this);
                    }
                }
            }
            unsafe impl ::cxx::private::WeakPtrTarget for Tensor {
                fn __typename(
                    f: &mut ::cxx::core::fmt::Formatter<'_>,
                ) -> ::cxx::core::fmt::Result {
                    f.write_str("Tensor")
                }
                unsafe fn __null(new: *mut ::cxx::core::ffi::c_void) {
                    extern "C" {
                        #[link_name = "cxxbridge1$weak_ptr$MNN$Tensor$null"]
                        fn __null(new: *mut ::cxx::core::ffi::c_void);
                    }
                    unsafe {
                        __null(new);
                    }
                }
                unsafe fn __clone(
                    this: *const ::cxx::core::ffi::c_void,
                    new: *mut ::cxx::core::ffi::c_void,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$weak_ptr$MNN$Tensor$clone"]
                        fn __clone(
                            this: *const ::cxx::core::ffi::c_void,
                            new: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __clone(this, new);
                    }
                }
                unsafe fn __downgrade(
                    shared: *const ::cxx::core::ffi::c_void,
                    weak: *mut ::cxx::core::ffi::c_void,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$weak_ptr$MNN$Tensor$downgrade"]
                        fn __downgrade(
                            shared: *const ::cxx::core::ffi::c_void,
                            weak: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __downgrade(shared, weak);
                    }
                }
                unsafe fn __upgrade(
                    weak: *const ::cxx::core::ffi::c_void,
                    shared: *mut ::cxx::core::ffi::c_void,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$weak_ptr$MNN$Tensor$upgrade"]
                        fn __upgrade(
                            weak: *const ::cxx::core::ffi::c_void,
                            shared: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __upgrade(weak, shared);
                    }
                }
                unsafe fn __drop(this: *mut ::cxx::core::ffi::c_void) {
                    extern "C" {
                        #[link_name = "cxxbridge1$weak_ptr$MNN$Tensor$drop"]
                        fn __drop(this: *mut ::cxx::core::ffi::c_void);
                    }
                    unsafe {
                        __drop(this);
                    }
                }
            }
            unsafe impl ::cxx::private::UniquePtrTarget for halide_buffer_t {
                fn __typename(
                    f: &mut ::cxx::core::fmt::Formatter<'_>,
                ) -> ::cxx::core::fmt::Result {
                    f.write_str("halide_buffer_t")
                }
                fn __null() -> ::cxx::core::mem::MaybeUninit<
                    *mut ::cxx::core::ffi::c_void,
                > {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$halide_buffer_t$null"]
                        fn __null(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        );
                    }
                    let mut repr = ::cxx::core::mem::MaybeUninit::uninit();
                    unsafe {
                        __null(&mut repr);
                    }
                    repr
                }
                fn __new(
                    value: Self,
                ) -> ::cxx::core::mem::MaybeUninit<*mut ::cxx::core::ffi::c_void> {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$halide_buffer_t$uninit"]
                        fn __uninit(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        ) -> *mut ::cxx::core::ffi::c_void;
                    }
                    let mut repr = ::cxx::core::mem::MaybeUninit::uninit();
                    unsafe {
                        __uninit(&mut repr).cast::<halide_buffer_t>().write(value);
                    }
                    repr
                }
                unsafe fn __raw(
                    raw: *mut Self,
                ) -> ::cxx::core::mem::MaybeUninit<*mut ::cxx::core::ffi::c_void> {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$halide_buffer_t$raw"]
                        fn __raw(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                            raw: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    let mut repr = ::cxx::core::mem::MaybeUninit::uninit();
                    unsafe {
                        __raw(&mut repr, raw.cast());
                    }
                    repr
                }
                unsafe fn __get(
                    repr: ::cxx::core::mem::MaybeUninit<*mut ::cxx::core::ffi::c_void>,
                ) -> *const Self {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$halide_buffer_t$get"]
                        fn __get(
                            this: *const ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        ) -> *const ::cxx::core::ffi::c_void;
                    }
                    unsafe { __get(&repr).cast() }
                }
                unsafe fn __release(
                    mut repr: ::cxx::core::mem::MaybeUninit<
                        *mut ::cxx::core::ffi::c_void,
                    >,
                ) -> *mut Self {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$halide_buffer_t$release"]
                        fn __release(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        ) -> *mut ::cxx::core::ffi::c_void;
                    }
                    unsafe { __release(&mut repr).cast() }
                }
                unsafe fn __drop(
                    mut repr: ::cxx::core::mem::MaybeUninit<
                        *mut ::cxx::core::ffi::c_void,
                    >,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$halide_buffer_t$drop"]
                        fn __drop(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        );
                    }
                    unsafe {
                        __drop(&mut repr);
                    }
                }
            }
            unsafe impl ::cxx::private::SharedPtrTarget for halide_buffer_t {
                fn __typename(
                    f: &mut ::cxx::core::fmt::Formatter<'_>,
                ) -> ::cxx::core::fmt::Result {
                    f.write_str("halide_buffer_t")
                }
                unsafe fn __null(new: *mut ::cxx::core::ffi::c_void) {
                    extern "C" {
                        #[link_name = "cxxbridge1$shared_ptr$halide_buffer_t$null"]
                        fn __null(new: *mut ::cxx::core::ffi::c_void);
                    }
                    unsafe {
                        __null(new);
                    }
                }
                unsafe fn __new(value: Self, new: *mut ::cxx::core::ffi::c_void) {
                    extern "C" {
                        #[link_name = "cxxbridge1$shared_ptr$halide_buffer_t$uninit"]
                        fn __uninit(
                            new: *mut ::cxx::core::ffi::c_void,
                        ) -> *mut ::cxx::core::ffi::c_void;
                    }
                    unsafe {
                        __uninit(new).cast::<halide_buffer_t>().write(value);
                    }
                }
                unsafe fn __clone(
                    this: *const ::cxx::core::ffi::c_void,
                    new: *mut ::cxx::core::ffi::c_void,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$shared_ptr$halide_buffer_t$clone"]
                        fn __clone(
                            this: *const ::cxx::core::ffi::c_void,
                            new: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __clone(this, new);
                    }
                }
                unsafe fn __get(this: *const ::cxx::core::ffi::c_void) -> *const Self {
                    extern "C" {
                        #[link_name = "cxxbridge1$shared_ptr$halide_buffer_t$get"]
                        fn __get(
                            this: *const ::cxx::core::ffi::c_void,
                        ) -> *const ::cxx::core::ffi::c_void;
                    }
                    unsafe { __get(this).cast() }
                }
                unsafe fn __drop(this: *mut ::cxx::core::ffi::c_void) {
                    extern "C" {
                        #[link_name = "cxxbridge1$shared_ptr$halide_buffer_t$drop"]
                        fn __drop(this: *mut ::cxx::core::ffi::c_void);
                    }
                    unsafe {
                        __drop(this);
                    }
                }
            }
            unsafe impl ::cxx::private::WeakPtrTarget for halide_buffer_t {
                fn __typename(
                    f: &mut ::cxx::core::fmt::Formatter<'_>,
                ) -> ::cxx::core::fmt::Result {
                    f.write_str("halide_buffer_t")
                }
                unsafe fn __null(new: *mut ::cxx::core::ffi::c_void) {
                    extern "C" {
                        #[link_name = "cxxbridge1$weak_ptr$halide_buffer_t$null"]
                        fn __null(new: *mut ::cxx::core::ffi::c_void);
                    }
                    unsafe {
                        __null(new);
                    }
                }
                unsafe fn __clone(
                    this: *const ::cxx::core::ffi::c_void,
                    new: *mut ::cxx::core::ffi::c_void,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$weak_ptr$halide_buffer_t$clone"]
                        fn __clone(
                            this: *const ::cxx::core::ffi::c_void,
                            new: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __clone(this, new);
                    }
                }
                unsafe fn __downgrade(
                    shared: *const ::cxx::core::ffi::c_void,
                    weak: *mut ::cxx::core::ffi::c_void,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$weak_ptr$halide_buffer_t$downgrade"]
                        fn __downgrade(
                            shared: *const ::cxx::core::ffi::c_void,
                            weak: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __downgrade(shared, weak);
                    }
                }
                unsafe fn __upgrade(
                    weak: *const ::cxx::core::ffi::c_void,
                    shared: *mut ::cxx::core::ffi::c_void,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$weak_ptr$halide_buffer_t$upgrade"]
                        fn __upgrade(
                            weak: *const ::cxx::core::ffi::c_void,
                            shared: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __upgrade(weak, shared);
                    }
                }
                unsafe fn __drop(this: *mut ::cxx::core::ffi::c_void) {
                    extern "C" {
                        #[link_name = "cxxbridge1$weak_ptr$halide_buffer_t$drop"]
                        fn __drop(this: *mut ::cxx::core::ffi::c_void);
                    }
                    unsafe {
                        __drop(this);
                    }
                }
            }
            unsafe impl ::cxx::private::VectorElement for halide_buffer_t {
                fn __typename(
                    f: &mut ::cxx::core::fmt::Formatter<'_>,
                ) -> ::cxx::core::fmt::Result {
                    f.write_str("halide_buffer_t")
                }
                fn __vector_new() -> *mut ::cxx::CxxVector<Self> {
                    extern "C" {
                        #[link_name = "cxxbridge1$std$vector$halide_buffer_t$new"]
                        fn __vector_new() -> *mut ::cxx::CxxVector<halide_buffer_t>;
                    }
                    unsafe { __vector_new() }
                }
                fn __vector_size(v: &::cxx::CxxVector<Self>) -> usize {
                    extern "C" {
                        #[link_name = "cxxbridge1$std$vector$halide_buffer_t$size"]
                        fn __vector_size(_: &::cxx::CxxVector<halide_buffer_t>) -> usize;
                    }
                    unsafe { __vector_size(v) }
                }
                unsafe fn __get_unchecked(
                    v: *mut ::cxx::CxxVector<Self>,
                    pos: usize,
                ) -> *mut Self {
                    extern "C" {
                        #[link_name = "cxxbridge1$std$vector$halide_buffer_t$get_unchecked"]
                        fn __get_unchecked(
                            v: *mut ::cxx::CxxVector<halide_buffer_t>,
                            pos: usize,
                        ) -> *mut ::cxx::core::ffi::c_void;
                    }
                    unsafe { __get_unchecked(v, pos) as *mut Self }
                }
                unsafe fn __push_back(
                    this: ::cxx::core::pin::Pin<&mut ::cxx::CxxVector<Self>>,
                    value: &mut ::cxx::core::mem::ManuallyDrop<Self>,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$std$vector$halide_buffer_t$push_back"]
                        fn __push_back(
                            this: ::cxx::core::pin::Pin<
                                &mut ::cxx::CxxVector<halide_buffer_t>,
                            >,
                            value: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __push_back(
                            this,
                            value as *mut ::cxx::core::mem::ManuallyDrop<Self>
                                as *mut ::cxx::core::ffi::c_void,
                        );
                    }
                }
                unsafe fn __pop_back(
                    this: ::cxx::core::pin::Pin<&mut ::cxx::CxxVector<Self>>,
                    out: &mut ::cxx::core::mem::MaybeUninit<Self>,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$std$vector$halide_buffer_t$pop_back"]
                        fn __pop_back(
                            this: ::cxx::core::pin::Pin<
                                &mut ::cxx::CxxVector<halide_buffer_t>,
                            >,
                            out: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __pop_back(
                            this,
                            out as *mut ::cxx::core::mem::MaybeUninit<Self>
                                as *mut ::cxx::core::ffi::c_void,
                        );
                    }
                }
                fn __unique_ptr_null() -> ::cxx::core::mem::MaybeUninit<
                    *mut ::cxx::core::ffi::c_void,
                > {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$std$vector$halide_buffer_t$null"]
                        fn __unique_ptr_null(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        );
                    }
                    let mut repr = ::cxx::core::mem::MaybeUninit::uninit();
                    unsafe {
                        __unique_ptr_null(&mut repr);
                    }
                    repr
                }
                unsafe fn __unique_ptr_raw(
                    raw: *mut ::cxx::CxxVector<Self>,
                ) -> ::cxx::core::mem::MaybeUninit<*mut ::cxx::core::ffi::c_void> {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$std$vector$halide_buffer_t$raw"]
                        fn __unique_ptr_raw(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                            raw: *mut ::cxx::CxxVector<halide_buffer_t>,
                        );
                    }
                    let mut repr = ::cxx::core::mem::MaybeUninit::uninit();
                    unsafe {
                        __unique_ptr_raw(&mut repr, raw);
                    }
                    repr
                }
                unsafe fn __unique_ptr_get(
                    repr: ::cxx::core::mem::MaybeUninit<*mut ::cxx::core::ffi::c_void>,
                ) -> *const ::cxx::CxxVector<Self> {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$std$vector$halide_buffer_t$get"]
                        fn __unique_ptr_get(
                            this: *const ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        ) -> *const ::cxx::CxxVector<halide_buffer_t>;
                    }
                    unsafe { __unique_ptr_get(&repr) }
                }
                unsafe fn __unique_ptr_release(
                    mut repr: ::cxx::core::mem::MaybeUninit<
                        *mut ::cxx::core::ffi::c_void,
                    >,
                ) -> *mut ::cxx::CxxVector<Self> {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$std$vector$halide_buffer_t$release"]
                        fn __unique_ptr_release(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        ) -> *mut ::cxx::CxxVector<halide_buffer_t>;
                    }
                    unsafe { __unique_ptr_release(&mut repr) }
                }
                unsafe fn __unique_ptr_drop(
                    mut repr: ::cxx::core::mem::MaybeUninit<
                        *mut ::cxx::core::ffi::c_void,
                    >,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$std$vector$halide_buffer_t$drop"]
                        fn __unique_ptr_drop(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        );
                    }
                    unsafe {
                        __unique_ptr_drop(&mut repr);
                    }
                }
            }
            unsafe impl ::cxx::private::UniquePtrTarget for Tensor_DimensionType {
                fn __typename(
                    f: &mut ::cxx::core::fmt::Formatter<'_>,
                ) -> ::cxx::core::fmt::Result {
                    f.write_str("Tensor_DimensionType")
                }
                fn __null() -> ::cxx::core::mem::MaybeUninit<
                    *mut ::cxx::core::ffi::c_void,
                > {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$MNN$Tensor$DimensionType$null"]
                        fn __null(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        );
                    }
                    let mut repr = ::cxx::core::mem::MaybeUninit::uninit();
                    unsafe {
                        __null(&mut repr);
                    }
                    repr
                }
                fn __new(
                    value: Self,
                ) -> ::cxx::core::mem::MaybeUninit<*mut ::cxx::core::ffi::c_void> {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$MNN$Tensor$DimensionType$uninit"]
                        fn __uninit(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        ) -> *mut ::cxx::core::ffi::c_void;
                    }
                    let mut repr = ::cxx::core::mem::MaybeUninit::uninit();
                    unsafe {
                        __uninit(&mut repr).cast::<Tensor_DimensionType>().write(value);
                    }
                    repr
                }
                unsafe fn __raw(
                    raw: *mut Self,
                ) -> ::cxx::core::mem::MaybeUninit<*mut ::cxx::core::ffi::c_void> {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$MNN$Tensor$DimensionType$raw"]
                        fn __raw(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                            raw: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    let mut repr = ::cxx::core::mem::MaybeUninit::uninit();
                    unsafe {
                        __raw(&mut repr, raw.cast());
                    }
                    repr
                }
                unsafe fn __get(
                    repr: ::cxx::core::mem::MaybeUninit<*mut ::cxx::core::ffi::c_void>,
                ) -> *const Self {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$MNN$Tensor$DimensionType$get"]
                        fn __get(
                            this: *const ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        ) -> *const ::cxx::core::ffi::c_void;
                    }
                    unsafe { __get(&repr).cast() }
                }
                unsafe fn __release(
                    mut repr: ::cxx::core::mem::MaybeUninit<
                        *mut ::cxx::core::ffi::c_void,
                    >,
                ) -> *mut Self {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$MNN$Tensor$DimensionType$release"]
                        fn __release(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        ) -> *mut ::cxx::core::ffi::c_void;
                    }
                    unsafe { __release(&mut repr).cast() }
                }
                unsafe fn __drop(
                    mut repr: ::cxx::core::mem::MaybeUninit<
                        *mut ::cxx::core::ffi::c_void,
                    >,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$MNN$Tensor$DimensionType$drop"]
                        fn __drop(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        );
                    }
                    unsafe {
                        __drop(&mut repr);
                    }
                }
            }
            unsafe impl ::cxx::private::SharedPtrTarget for Tensor_DimensionType {
                fn __typename(
                    f: &mut ::cxx::core::fmt::Formatter<'_>,
                ) -> ::cxx::core::fmt::Result {
                    f.write_str("Tensor_DimensionType")
                }
                unsafe fn __null(new: *mut ::cxx::core::ffi::c_void) {
                    extern "C" {
                        #[link_name = "cxxbridge1$shared_ptr$MNN$Tensor$DimensionType$null"]
                        fn __null(new: *mut ::cxx::core::ffi::c_void);
                    }
                    unsafe {
                        __null(new);
                    }
                }
                unsafe fn __new(value: Self, new: *mut ::cxx::core::ffi::c_void) {
                    extern "C" {
                        #[link_name = "cxxbridge1$shared_ptr$MNN$Tensor$DimensionType$uninit"]
                        fn __uninit(
                            new: *mut ::cxx::core::ffi::c_void,
                        ) -> *mut ::cxx::core::ffi::c_void;
                    }
                    unsafe {
                        __uninit(new).cast::<Tensor_DimensionType>().write(value);
                    }
                }
                unsafe fn __clone(
                    this: *const ::cxx::core::ffi::c_void,
                    new: *mut ::cxx::core::ffi::c_void,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$shared_ptr$MNN$Tensor$DimensionType$clone"]
                        fn __clone(
                            this: *const ::cxx::core::ffi::c_void,
                            new: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __clone(this, new);
                    }
                }
                unsafe fn __get(this: *const ::cxx::core::ffi::c_void) -> *const Self {
                    extern "C" {
                        #[link_name = "cxxbridge1$shared_ptr$MNN$Tensor$DimensionType$get"]
                        fn __get(
                            this: *const ::cxx::core::ffi::c_void,
                        ) -> *const ::cxx::core::ffi::c_void;
                    }
                    unsafe { __get(this).cast() }
                }
                unsafe fn __drop(this: *mut ::cxx::core::ffi::c_void) {
                    extern "C" {
                        #[link_name = "cxxbridge1$shared_ptr$MNN$Tensor$DimensionType$drop"]
                        fn __drop(this: *mut ::cxx::core::ffi::c_void);
                    }
                    unsafe {
                        __drop(this);
                    }
                }
            }
            unsafe impl ::cxx::private::WeakPtrTarget for Tensor_DimensionType {
                fn __typename(
                    f: &mut ::cxx::core::fmt::Formatter<'_>,
                ) -> ::cxx::core::fmt::Result {
                    f.write_str("Tensor_DimensionType")
                }
                unsafe fn __null(new: *mut ::cxx::core::ffi::c_void) {
                    extern "C" {
                        #[link_name = "cxxbridge1$weak_ptr$MNN$Tensor$DimensionType$null"]
                        fn __null(new: *mut ::cxx::core::ffi::c_void);
                    }
                    unsafe {
                        __null(new);
                    }
                }
                unsafe fn __clone(
                    this: *const ::cxx::core::ffi::c_void,
                    new: *mut ::cxx::core::ffi::c_void,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$weak_ptr$MNN$Tensor$DimensionType$clone"]
                        fn __clone(
                            this: *const ::cxx::core::ffi::c_void,
                            new: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __clone(this, new);
                    }
                }
                unsafe fn __downgrade(
                    shared: *const ::cxx::core::ffi::c_void,
                    weak: *mut ::cxx::core::ffi::c_void,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$weak_ptr$MNN$Tensor$DimensionType$downgrade"]
                        fn __downgrade(
                            shared: *const ::cxx::core::ffi::c_void,
                            weak: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __downgrade(shared, weak);
                    }
                }
                unsafe fn __upgrade(
                    weak: *const ::cxx::core::ffi::c_void,
                    shared: *mut ::cxx::core::ffi::c_void,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$weak_ptr$MNN$Tensor$DimensionType$upgrade"]
                        fn __upgrade(
                            weak: *const ::cxx::core::ffi::c_void,
                            shared: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __upgrade(weak, shared);
                    }
                }
                unsafe fn __drop(this: *mut ::cxx::core::ffi::c_void) {
                    extern "C" {
                        #[link_name = "cxxbridge1$weak_ptr$MNN$Tensor$DimensionType$drop"]
                        fn __drop(this: *mut ::cxx::core::ffi::c_void);
                    }
                    unsafe {
                        __drop(this);
                    }
                }
            }
            unsafe impl ::cxx::private::VectorElement for Tensor_DimensionType {
                fn __typename(
                    f: &mut ::cxx::core::fmt::Formatter<'_>,
                ) -> ::cxx::core::fmt::Result {
                    f.write_str("Tensor_DimensionType")
                }
                fn __vector_new() -> *mut ::cxx::CxxVector<Self> {
                    extern "C" {
                        #[link_name = "cxxbridge1$std$vector$MNN$Tensor$DimensionType$new"]
                        fn __vector_new() -> *mut ::cxx::CxxVector<Tensor_DimensionType>;
                    }
                    unsafe { __vector_new() }
                }
                fn __vector_size(v: &::cxx::CxxVector<Self>) -> usize {
                    extern "C" {
                        #[link_name = "cxxbridge1$std$vector$MNN$Tensor$DimensionType$size"]
                        fn __vector_size(
                            _: &::cxx::CxxVector<Tensor_DimensionType>,
                        ) -> usize;
                    }
                    unsafe { __vector_size(v) }
                }
                unsafe fn __get_unchecked(
                    v: *mut ::cxx::CxxVector<Self>,
                    pos: usize,
                ) -> *mut Self {
                    extern "C" {
                        #[link_name = "cxxbridge1$std$vector$MNN$Tensor$DimensionType$get_unchecked"]
                        fn __get_unchecked(
                            v: *mut ::cxx::CxxVector<Tensor_DimensionType>,
                            pos: usize,
                        ) -> *mut ::cxx::core::ffi::c_void;
                    }
                    unsafe { __get_unchecked(v, pos) as *mut Self }
                }
                unsafe fn __push_back(
                    this: ::cxx::core::pin::Pin<&mut ::cxx::CxxVector<Self>>,
                    value: &mut ::cxx::core::mem::ManuallyDrop<Self>,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$std$vector$MNN$Tensor$DimensionType$push_back"]
                        fn __push_back(
                            this: ::cxx::core::pin::Pin<
                                &mut ::cxx::CxxVector<Tensor_DimensionType>,
                            >,
                            value: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __push_back(
                            this,
                            value as *mut ::cxx::core::mem::ManuallyDrop<Self>
                                as *mut ::cxx::core::ffi::c_void,
                        );
                    }
                }
                unsafe fn __pop_back(
                    this: ::cxx::core::pin::Pin<&mut ::cxx::CxxVector<Self>>,
                    out: &mut ::cxx::core::mem::MaybeUninit<Self>,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$std$vector$MNN$Tensor$DimensionType$pop_back"]
                        fn __pop_back(
                            this: ::cxx::core::pin::Pin<
                                &mut ::cxx::CxxVector<Tensor_DimensionType>,
                            >,
                            out: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __pop_back(
                            this,
                            out as *mut ::cxx::core::mem::MaybeUninit<Self>
                                as *mut ::cxx::core::ffi::c_void,
                        );
                    }
                }
                fn __unique_ptr_null() -> ::cxx::core::mem::MaybeUninit<
                    *mut ::cxx::core::ffi::c_void,
                > {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$std$vector$MNN$Tensor$DimensionType$null"]
                        fn __unique_ptr_null(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        );
                    }
                    let mut repr = ::cxx::core::mem::MaybeUninit::uninit();
                    unsafe {
                        __unique_ptr_null(&mut repr);
                    }
                    repr
                }
                unsafe fn __unique_ptr_raw(
                    raw: *mut ::cxx::CxxVector<Self>,
                ) -> ::cxx::core::mem::MaybeUninit<*mut ::cxx::core::ffi::c_void> {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$std$vector$MNN$Tensor$DimensionType$raw"]
                        fn __unique_ptr_raw(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                            raw: *mut ::cxx::CxxVector<Tensor_DimensionType>,
                        );
                    }
                    let mut repr = ::cxx::core::mem::MaybeUninit::uninit();
                    unsafe {
                        __unique_ptr_raw(&mut repr, raw);
                    }
                    repr
                }
                unsafe fn __unique_ptr_get(
                    repr: ::cxx::core::mem::MaybeUninit<*mut ::cxx::core::ffi::c_void>,
                ) -> *const ::cxx::CxxVector<Self> {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$std$vector$MNN$Tensor$DimensionType$get"]
                        fn __unique_ptr_get(
                            this: *const ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        ) -> *const ::cxx::CxxVector<Tensor_DimensionType>;
                    }
                    unsafe { __unique_ptr_get(&repr) }
                }
                unsafe fn __unique_ptr_release(
                    mut repr: ::cxx::core::mem::MaybeUninit<
                        *mut ::cxx::core::ffi::c_void,
                    >,
                ) -> *mut ::cxx::CxxVector<Self> {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$std$vector$MNN$Tensor$DimensionType$release"]
                        fn __unique_ptr_release(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        ) -> *mut ::cxx::CxxVector<Tensor_DimensionType>;
                    }
                    unsafe { __unique_ptr_release(&mut repr) }
                }
                unsafe fn __unique_ptr_drop(
                    mut repr: ::cxx::core::mem::MaybeUninit<
                        *mut ::cxx::core::ffi::c_void,
                    >,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$std$vector$MNN$Tensor$DimensionType$drop"]
                        fn __unique_ptr_drop(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        );
                    }
                    unsafe {
                        __unique_ptr_drop(&mut repr);
                    }
                }
            }
            unsafe impl ::cxx::private::UniquePtrTarget for Tensor_HandleDataType {
                fn __typename(
                    f: &mut ::cxx::core::fmt::Formatter<'_>,
                ) -> ::cxx::core::fmt::Result {
                    f.write_str("Tensor_HandleDataType")
                }
                fn __null() -> ::cxx::core::mem::MaybeUninit<
                    *mut ::cxx::core::ffi::c_void,
                > {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$MNN$Tensor$HandleDataType$null"]
                        fn __null(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        );
                    }
                    let mut repr = ::cxx::core::mem::MaybeUninit::uninit();
                    unsafe {
                        __null(&mut repr);
                    }
                    repr
                }
                fn __new(
                    value: Self,
                ) -> ::cxx::core::mem::MaybeUninit<*mut ::cxx::core::ffi::c_void> {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$MNN$Tensor$HandleDataType$uninit"]
                        fn __uninit(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        ) -> *mut ::cxx::core::ffi::c_void;
                    }
                    let mut repr = ::cxx::core::mem::MaybeUninit::uninit();
                    unsafe {
                        __uninit(&mut repr).cast::<Tensor_HandleDataType>().write(value);
                    }
                    repr
                }
                unsafe fn __raw(
                    raw: *mut Self,
                ) -> ::cxx::core::mem::MaybeUninit<*mut ::cxx::core::ffi::c_void> {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$MNN$Tensor$HandleDataType$raw"]
                        fn __raw(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                            raw: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    let mut repr = ::cxx::core::mem::MaybeUninit::uninit();
                    unsafe {
                        __raw(&mut repr, raw.cast());
                    }
                    repr
                }
                unsafe fn __get(
                    repr: ::cxx::core::mem::MaybeUninit<*mut ::cxx::core::ffi::c_void>,
                ) -> *const Self {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$MNN$Tensor$HandleDataType$get"]
                        fn __get(
                            this: *const ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        ) -> *const ::cxx::core::ffi::c_void;
                    }
                    unsafe { __get(&repr).cast() }
                }
                unsafe fn __release(
                    mut repr: ::cxx::core::mem::MaybeUninit<
                        *mut ::cxx::core::ffi::c_void,
                    >,
                ) -> *mut Self {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$MNN$Tensor$HandleDataType$release"]
                        fn __release(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        ) -> *mut ::cxx::core::ffi::c_void;
                    }
                    unsafe { __release(&mut repr).cast() }
                }
                unsafe fn __drop(
                    mut repr: ::cxx::core::mem::MaybeUninit<
                        *mut ::cxx::core::ffi::c_void,
                    >,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$MNN$Tensor$HandleDataType$drop"]
                        fn __drop(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        );
                    }
                    unsafe {
                        __drop(&mut repr);
                    }
                }
            }
            unsafe impl ::cxx::private::SharedPtrTarget for Tensor_HandleDataType {
                fn __typename(
                    f: &mut ::cxx::core::fmt::Formatter<'_>,
                ) -> ::cxx::core::fmt::Result {
                    f.write_str("Tensor_HandleDataType")
                }
                unsafe fn __null(new: *mut ::cxx::core::ffi::c_void) {
                    extern "C" {
                        #[link_name = "cxxbridge1$shared_ptr$MNN$Tensor$HandleDataType$null"]
                        fn __null(new: *mut ::cxx::core::ffi::c_void);
                    }
                    unsafe {
                        __null(new);
                    }
                }
                unsafe fn __new(value: Self, new: *mut ::cxx::core::ffi::c_void) {
                    extern "C" {
                        #[link_name = "cxxbridge1$shared_ptr$MNN$Tensor$HandleDataType$uninit"]
                        fn __uninit(
                            new: *mut ::cxx::core::ffi::c_void,
                        ) -> *mut ::cxx::core::ffi::c_void;
                    }
                    unsafe {
                        __uninit(new).cast::<Tensor_HandleDataType>().write(value);
                    }
                }
                unsafe fn __clone(
                    this: *const ::cxx::core::ffi::c_void,
                    new: *mut ::cxx::core::ffi::c_void,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$shared_ptr$MNN$Tensor$HandleDataType$clone"]
                        fn __clone(
                            this: *const ::cxx::core::ffi::c_void,
                            new: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __clone(this, new);
                    }
                }
                unsafe fn __get(this: *const ::cxx::core::ffi::c_void) -> *const Self {
                    extern "C" {
                        #[link_name = "cxxbridge1$shared_ptr$MNN$Tensor$HandleDataType$get"]
                        fn __get(
                            this: *const ::cxx::core::ffi::c_void,
                        ) -> *const ::cxx::core::ffi::c_void;
                    }
                    unsafe { __get(this).cast() }
                }
                unsafe fn __drop(this: *mut ::cxx::core::ffi::c_void) {
                    extern "C" {
                        #[link_name = "cxxbridge1$shared_ptr$MNN$Tensor$HandleDataType$drop"]
                        fn __drop(this: *mut ::cxx::core::ffi::c_void);
                    }
                    unsafe {
                        __drop(this);
                    }
                }
            }
            unsafe impl ::cxx::private::WeakPtrTarget for Tensor_HandleDataType {
                fn __typename(
                    f: &mut ::cxx::core::fmt::Formatter<'_>,
                ) -> ::cxx::core::fmt::Result {
                    f.write_str("Tensor_HandleDataType")
                }
                unsafe fn __null(new: *mut ::cxx::core::ffi::c_void) {
                    extern "C" {
                        #[link_name = "cxxbridge1$weak_ptr$MNN$Tensor$HandleDataType$null"]
                        fn __null(new: *mut ::cxx::core::ffi::c_void);
                    }
                    unsafe {
                        __null(new);
                    }
                }
                unsafe fn __clone(
                    this: *const ::cxx::core::ffi::c_void,
                    new: *mut ::cxx::core::ffi::c_void,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$weak_ptr$MNN$Tensor$HandleDataType$clone"]
                        fn __clone(
                            this: *const ::cxx::core::ffi::c_void,
                            new: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __clone(this, new);
                    }
                }
                unsafe fn __downgrade(
                    shared: *const ::cxx::core::ffi::c_void,
                    weak: *mut ::cxx::core::ffi::c_void,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$weak_ptr$MNN$Tensor$HandleDataType$downgrade"]
                        fn __downgrade(
                            shared: *const ::cxx::core::ffi::c_void,
                            weak: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __downgrade(shared, weak);
                    }
                }
                unsafe fn __upgrade(
                    weak: *const ::cxx::core::ffi::c_void,
                    shared: *mut ::cxx::core::ffi::c_void,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$weak_ptr$MNN$Tensor$HandleDataType$upgrade"]
                        fn __upgrade(
                            weak: *const ::cxx::core::ffi::c_void,
                            shared: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __upgrade(weak, shared);
                    }
                }
                unsafe fn __drop(this: *mut ::cxx::core::ffi::c_void) {
                    extern "C" {
                        #[link_name = "cxxbridge1$weak_ptr$MNN$Tensor$HandleDataType$drop"]
                        fn __drop(this: *mut ::cxx::core::ffi::c_void);
                    }
                    unsafe {
                        __drop(this);
                    }
                }
            }
            unsafe impl ::cxx::private::VectorElement for Tensor_HandleDataType {
                fn __typename(
                    f: &mut ::cxx::core::fmt::Formatter<'_>,
                ) -> ::cxx::core::fmt::Result {
                    f.write_str("Tensor_HandleDataType")
                }
                fn __vector_new() -> *mut ::cxx::CxxVector<Self> {
                    extern "C" {
                        #[link_name = "cxxbridge1$std$vector$MNN$Tensor$HandleDataType$new"]
                        fn __vector_new() -> *mut ::cxx::CxxVector<
                            Tensor_HandleDataType,
                        >;
                    }
                    unsafe { __vector_new() }
                }
                fn __vector_size(v: &::cxx::CxxVector<Self>) -> usize {
                    extern "C" {
                        #[link_name = "cxxbridge1$std$vector$MNN$Tensor$HandleDataType$size"]
                        fn __vector_size(
                            _: &::cxx::CxxVector<Tensor_HandleDataType>,
                        ) -> usize;
                    }
                    unsafe { __vector_size(v) }
                }
                unsafe fn __get_unchecked(
                    v: *mut ::cxx::CxxVector<Self>,
                    pos: usize,
                ) -> *mut Self {
                    extern "C" {
                        #[link_name = "cxxbridge1$std$vector$MNN$Tensor$HandleDataType$get_unchecked"]
                        fn __get_unchecked(
                            v: *mut ::cxx::CxxVector<Tensor_HandleDataType>,
                            pos: usize,
                        ) -> *mut ::cxx::core::ffi::c_void;
                    }
                    unsafe { __get_unchecked(v, pos) as *mut Self }
                }
                unsafe fn __push_back(
                    this: ::cxx::core::pin::Pin<&mut ::cxx::CxxVector<Self>>,
                    value: &mut ::cxx::core::mem::ManuallyDrop<Self>,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$std$vector$MNN$Tensor$HandleDataType$push_back"]
                        fn __push_back(
                            this: ::cxx::core::pin::Pin<
                                &mut ::cxx::CxxVector<Tensor_HandleDataType>,
                            >,
                            value: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __push_back(
                            this,
                            value as *mut ::cxx::core::mem::ManuallyDrop<Self>
                                as *mut ::cxx::core::ffi::c_void,
                        );
                    }
                }
                unsafe fn __pop_back(
                    this: ::cxx::core::pin::Pin<&mut ::cxx::CxxVector<Self>>,
                    out: &mut ::cxx::core::mem::MaybeUninit<Self>,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$std$vector$MNN$Tensor$HandleDataType$pop_back"]
                        fn __pop_back(
                            this: ::cxx::core::pin::Pin<
                                &mut ::cxx::CxxVector<Tensor_HandleDataType>,
                            >,
                            out: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __pop_back(
                            this,
                            out as *mut ::cxx::core::mem::MaybeUninit<Self>
                                as *mut ::cxx::core::ffi::c_void,
                        );
                    }
                }
                fn __unique_ptr_null() -> ::cxx::core::mem::MaybeUninit<
                    *mut ::cxx::core::ffi::c_void,
                > {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$std$vector$MNN$Tensor$HandleDataType$null"]
                        fn __unique_ptr_null(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        );
                    }
                    let mut repr = ::cxx::core::mem::MaybeUninit::uninit();
                    unsafe {
                        __unique_ptr_null(&mut repr);
                    }
                    repr
                }
                unsafe fn __unique_ptr_raw(
                    raw: *mut ::cxx::CxxVector<Self>,
                ) -> ::cxx::core::mem::MaybeUninit<*mut ::cxx::core::ffi::c_void> {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$std$vector$MNN$Tensor$HandleDataType$raw"]
                        fn __unique_ptr_raw(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                            raw: *mut ::cxx::CxxVector<Tensor_HandleDataType>,
                        );
                    }
                    let mut repr = ::cxx::core::mem::MaybeUninit::uninit();
                    unsafe {
                        __unique_ptr_raw(&mut repr, raw);
                    }
                    repr
                }
                unsafe fn __unique_ptr_get(
                    repr: ::cxx::core::mem::MaybeUninit<*mut ::cxx::core::ffi::c_void>,
                ) -> *const ::cxx::CxxVector<Self> {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$std$vector$MNN$Tensor$HandleDataType$get"]
                        fn __unique_ptr_get(
                            this: *const ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        ) -> *const ::cxx::CxxVector<Tensor_HandleDataType>;
                    }
                    unsafe { __unique_ptr_get(&repr) }
                }
                unsafe fn __unique_ptr_release(
                    mut repr: ::cxx::core::mem::MaybeUninit<
                        *mut ::cxx::core::ffi::c_void,
                    >,
                ) -> *mut ::cxx::CxxVector<Self> {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$std$vector$MNN$Tensor$HandleDataType$release"]
                        fn __unique_ptr_release(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        ) -> *mut ::cxx::CxxVector<Tensor_HandleDataType>;
                    }
                    unsafe { __unique_ptr_release(&mut repr) }
                }
                unsafe fn __unique_ptr_drop(
                    mut repr: ::cxx::core::mem::MaybeUninit<
                        *mut ::cxx::core::ffi::c_void,
                    >,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$std$vector$MNN$Tensor$HandleDataType$drop"]
                        fn __unique_ptr_drop(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        );
                    }
                    unsafe {
                        __unique_ptr_drop(&mut repr);
                    }
                }
            }
            unsafe impl ::cxx::private::UniquePtrTarget for halide_type_t {
                fn __typename(
                    f: &mut ::cxx::core::fmt::Formatter<'_>,
                ) -> ::cxx::core::fmt::Result {
                    f.write_str("halide_type_t")
                }
                fn __null() -> ::cxx::core::mem::MaybeUninit<
                    *mut ::cxx::core::ffi::c_void,
                > {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$halide_type_t$null"]
                        fn __null(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        );
                    }
                    let mut repr = ::cxx::core::mem::MaybeUninit::uninit();
                    unsafe {
                        __null(&mut repr);
                    }
                    repr
                }
                fn __new(
                    value: Self,
                ) -> ::cxx::core::mem::MaybeUninit<*mut ::cxx::core::ffi::c_void> {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$halide_type_t$uninit"]
                        fn __uninit(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        ) -> *mut ::cxx::core::ffi::c_void;
                    }
                    let mut repr = ::cxx::core::mem::MaybeUninit::uninit();
                    unsafe {
                        __uninit(&mut repr).cast::<halide_type_t>().write(value);
                    }
                    repr
                }
                unsafe fn __raw(
                    raw: *mut Self,
                ) -> ::cxx::core::mem::MaybeUninit<*mut ::cxx::core::ffi::c_void> {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$halide_type_t$raw"]
                        fn __raw(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                            raw: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    let mut repr = ::cxx::core::mem::MaybeUninit::uninit();
                    unsafe {
                        __raw(&mut repr, raw.cast());
                    }
                    repr
                }
                unsafe fn __get(
                    repr: ::cxx::core::mem::MaybeUninit<*mut ::cxx::core::ffi::c_void>,
                ) -> *const Self {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$halide_type_t$get"]
                        fn __get(
                            this: *const ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        ) -> *const ::cxx::core::ffi::c_void;
                    }
                    unsafe { __get(&repr).cast() }
                }
                unsafe fn __release(
                    mut repr: ::cxx::core::mem::MaybeUninit<
                        *mut ::cxx::core::ffi::c_void,
                    >,
                ) -> *mut Self {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$halide_type_t$release"]
                        fn __release(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        ) -> *mut ::cxx::core::ffi::c_void;
                    }
                    unsafe { __release(&mut repr).cast() }
                }
                unsafe fn __drop(
                    mut repr: ::cxx::core::mem::MaybeUninit<
                        *mut ::cxx::core::ffi::c_void,
                    >,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$halide_type_t$drop"]
                        fn __drop(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        );
                    }
                    unsafe {
                        __drop(&mut repr);
                    }
                }
            }
            unsafe impl ::cxx::private::SharedPtrTarget for halide_type_t {
                fn __typename(
                    f: &mut ::cxx::core::fmt::Formatter<'_>,
                ) -> ::cxx::core::fmt::Result {
                    f.write_str("halide_type_t")
                }
                unsafe fn __null(new: *mut ::cxx::core::ffi::c_void) {
                    extern "C" {
                        #[link_name = "cxxbridge1$shared_ptr$halide_type_t$null"]
                        fn __null(new: *mut ::cxx::core::ffi::c_void);
                    }
                    unsafe {
                        __null(new);
                    }
                }
                unsafe fn __new(value: Self, new: *mut ::cxx::core::ffi::c_void) {
                    extern "C" {
                        #[link_name = "cxxbridge1$shared_ptr$halide_type_t$uninit"]
                        fn __uninit(
                            new: *mut ::cxx::core::ffi::c_void,
                        ) -> *mut ::cxx::core::ffi::c_void;
                    }
                    unsafe {
                        __uninit(new).cast::<halide_type_t>().write(value);
                    }
                }
                unsafe fn __clone(
                    this: *const ::cxx::core::ffi::c_void,
                    new: *mut ::cxx::core::ffi::c_void,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$shared_ptr$halide_type_t$clone"]
                        fn __clone(
                            this: *const ::cxx::core::ffi::c_void,
                            new: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __clone(this, new);
                    }
                }
                unsafe fn __get(this: *const ::cxx::core::ffi::c_void) -> *const Self {
                    extern "C" {
                        #[link_name = "cxxbridge1$shared_ptr$halide_type_t$get"]
                        fn __get(
                            this: *const ::cxx::core::ffi::c_void,
                        ) -> *const ::cxx::core::ffi::c_void;
                    }
                    unsafe { __get(this).cast() }
                }
                unsafe fn __drop(this: *mut ::cxx::core::ffi::c_void) {
                    extern "C" {
                        #[link_name = "cxxbridge1$shared_ptr$halide_type_t$drop"]
                        fn __drop(this: *mut ::cxx::core::ffi::c_void);
                    }
                    unsafe {
                        __drop(this);
                    }
                }
            }
            unsafe impl ::cxx::private::WeakPtrTarget for halide_type_t {
                fn __typename(
                    f: &mut ::cxx::core::fmt::Formatter<'_>,
                ) -> ::cxx::core::fmt::Result {
                    f.write_str("halide_type_t")
                }
                unsafe fn __null(new: *mut ::cxx::core::ffi::c_void) {
                    extern "C" {
                        #[link_name = "cxxbridge1$weak_ptr$halide_type_t$null"]
                        fn __null(new: *mut ::cxx::core::ffi::c_void);
                    }
                    unsafe {
                        __null(new);
                    }
                }
                unsafe fn __clone(
                    this: *const ::cxx::core::ffi::c_void,
                    new: *mut ::cxx::core::ffi::c_void,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$weak_ptr$halide_type_t$clone"]
                        fn __clone(
                            this: *const ::cxx::core::ffi::c_void,
                            new: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __clone(this, new);
                    }
                }
                unsafe fn __downgrade(
                    shared: *const ::cxx::core::ffi::c_void,
                    weak: *mut ::cxx::core::ffi::c_void,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$weak_ptr$halide_type_t$downgrade"]
                        fn __downgrade(
                            shared: *const ::cxx::core::ffi::c_void,
                            weak: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __downgrade(shared, weak);
                    }
                }
                unsafe fn __upgrade(
                    weak: *const ::cxx::core::ffi::c_void,
                    shared: *mut ::cxx::core::ffi::c_void,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$weak_ptr$halide_type_t$upgrade"]
                        fn __upgrade(
                            weak: *const ::cxx::core::ffi::c_void,
                            shared: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __upgrade(weak, shared);
                    }
                }
                unsafe fn __drop(this: *mut ::cxx::core::ffi::c_void) {
                    extern "C" {
                        #[link_name = "cxxbridge1$weak_ptr$halide_type_t$drop"]
                        fn __drop(this: *mut ::cxx::core::ffi::c_void);
                    }
                    unsafe {
                        __drop(this);
                    }
                }
            }
            unsafe impl ::cxx::private::VectorElement for halide_type_t {
                fn __typename(
                    f: &mut ::cxx::core::fmt::Formatter<'_>,
                ) -> ::cxx::core::fmt::Result {
                    f.write_str("halide_type_t")
                }
                fn __vector_new() -> *mut ::cxx::CxxVector<Self> {
                    extern "C" {
                        #[link_name = "cxxbridge1$std$vector$halide_type_t$new"]
                        fn __vector_new() -> *mut ::cxx::CxxVector<halide_type_t>;
                    }
                    unsafe { __vector_new() }
                }
                fn __vector_size(v: &::cxx::CxxVector<Self>) -> usize {
                    extern "C" {
                        #[link_name = "cxxbridge1$std$vector$halide_type_t$size"]
                        fn __vector_size(_: &::cxx::CxxVector<halide_type_t>) -> usize;
                    }
                    unsafe { __vector_size(v) }
                }
                unsafe fn __get_unchecked(
                    v: *mut ::cxx::CxxVector<Self>,
                    pos: usize,
                ) -> *mut Self {
                    extern "C" {
                        #[link_name = "cxxbridge1$std$vector$halide_type_t$get_unchecked"]
                        fn __get_unchecked(
                            v: *mut ::cxx::CxxVector<halide_type_t>,
                            pos: usize,
                        ) -> *mut ::cxx::core::ffi::c_void;
                    }
                    unsafe { __get_unchecked(v, pos) as *mut Self }
                }
                unsafe fn __push_back(
                    this: ::cxx::core::pin::Pin<&mut ::cxx::CxxVector<Self>>,
                    value: &mut ::cxx::core::mem::ManuallyDrop<Self>,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$std$vector$halide_type_t$push_back"]
                        fn __push_back(
                            this: ::cxx::core::pin::Pin<
                                &mut ::cxx::CxxVector<halide_type_t>,
                            >,
                            value: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __push_back(
                            this,
                            value as *mut ::cxx::core::mem::ManuallyDrop<Self>
                                as *mut ::cxx::core::ffi::c_void,
                        );
                    }
                }
                unsafe fn __pop_back(
                    this: ::cxx::core::pin::Pin<&mut ::cxx::CxxVector<Self>>,
                    out: &mut ::cxx::core::mem::MaybeUninit<Self>,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$std$vector$halide_type_t$pop_back"]
                        fn __pop_back(
                            this: ::cxx::core::pin::Pin<
                                &mut ::cxx::CxxVector<halide_type_t>,
                            >,
                            out: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __pop_back(
                            this,
                            out as *mut ::cxx::core::mem::MaybeUninit<Self>
                                as *mut ::cxx::core::ffi::c_void,
                        );
                    }
                }
                fn __unique_ptr_null() -> ::cxx::core::mem::MaybeUninit<
                    *mut ::cxx::core::ffi::c_void,
                > {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$std$vector$halide_type_t$null"]
                        fn __unique_ptr_null(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        );
                    }
                    let mut repr = ::cxx::core::mem::MaybeUninit::uninit();
                    unsafe {
                        __unique_ptr_null(&mut repr);
                    }
                    repr
                }
                unsafe fn __unique_ptr_raw(
                    raw: *mut ::cxx::CxxVector<Self>,
                ) -> ::cxx::core::mem::MaybeUninit<*mut ::cxx::core::ffi::c_void> {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$std$vector$halide_type_t$raw"]
                        fn __unique_ptr_raw(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                            raw: *mut ::cxx::CxxVector<halide_type_t>,
                        );
                    }
                    let mut repr = ::cxx::core::mem::MaybeUninit::uninit();
                    unsafe {
                        __unique_ptr_raw(&mut repr, raw);
                    }
                    repr
                }
                unsafe fn __unique_ptr_get(
                    repr: ::cxx::core::mem::MaybeUninit<*mut ::cxx::core::ffi::c_void>,
                ) -> *const ::cxx::CxxVector<Self> {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$std$vector$halide_type_t$get"]
                        fn __unique_ptr_get(
                            this: *const ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        ) -> *const ::cxx::CxxVector<halide_type_t>;
                    }
                    unsafe { __unique_ptr_get(&repr) }
                }
                unsafe fn __unique_ptr_release(
                    mut repr: ::cxx::core::mem::MaybeUninit<
                        *mut ::cxx::core::ffi::c_void,
                    >,
                ) -> *mut ::cxx::CxxVector<Self> {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$std$vector$halide_type_t$release"]
                        fn __unique_ptr_release(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        ) -> *mut ::cxx::CxxVector<halide_type_t>;
                    }
                    unsafe { __unique_ptr_release(&mut repr) }
                }
                unsafe fn __unique_ptr_drop(
                    mut repr: ::cxx::core::mem::MaybeUninit<
                        *mut ::cxx::core::ffi::c_void,
                    >,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$std$vector$halide_type_t$drop"]
                        fn __unique_ptr_drop(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        );
                    }
                    unsafe {
                        __unique_ptr_drop(&mut repr);
                    }
                }
            }
            unsafe impl ::cxx::private::UniquePtrTarget for Tensor_MapType {
                fn __typename(
                    f: &mut ::cxx::core::fmt::Formatter<'_>,
                ) -> ::cxx::core::fmt::Result {
                    f.write_str("Tensor_MapType")
                }
                fn __null() -> ::cxx::core::mem::MaybeUninit<
                    *mut ::cxx::core::ffi::c_void,
                > {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$MNN$Tensor$MapType$null"]
                        fn __null(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        );
                    }
                    let mut repr = ::cxx::core::mem::MaybeUninit::uninit();
                    unsafe {
                        __null(&mut repr);
                    }
                    repr
                }
                fn __new(
                    value: Self,
                ) -> ::cxx::core::mem::MaybeUninit<*mut ::cxx::core::ffi::c_void> {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$MNN$Tensor$MapType$uninit"]
                        fn __uninit(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        ) -> *mut ::cxx::core::ffi::c_void;
                    }
                    let mut repr = ::cxx::core::mem::MaybeUninit::uninit();
                    unsafe {
                        __uninit(&mut repr).cast::<Tensor_MapType>().write(value);
                    }
                    repr
                }
                unsafe fn __raw(
                    raw: *mut Self,
                ) -> ::cxx::core::mem::MaybeUninit<*mut ::cxx::core::ffi::c_void> {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$MNN$Tensor$MapType$raw"]
                        fn __raw(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                            raw: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    let mut repr = ::cxx::core::mem::MaybeUninit::uninit();
                    unsafe {
                        __raw(&mut repr, raw.cast());
                    }
                    repr
                }
                unsafe fn __get(
                    repr: ::cxx::core::mem::MaybeUninit<*mut ::cxx::core::ffi::c_void>,
                ) -> *const Self {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$MNN$Tensor$MapType$get"]
                        fn __get(
                            this: *const ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        ) -> *const ::cxx::core::ffi::c_void;
                    }
                    unsafe { __get(&repr).cast() }
                }
                unsafe fn __release(
                    mut repr: ::cxx::core::mem::MaybeUninit<
                        *mut ::cxx::core::ffi::c_void,
                    >,
                ) -> *mut Self {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$MNN$Tensor$MapType$release"]
                        fn __release(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        ) -> *mut ::cxx::core::ffi::c_void;
                    }
                    unsafe { __release(&mut repr).cast() }
                }
                unsafe fn __drop(
                    mut repr: ::cxx::core::mem::MaybeUninit<
                        *mut ::cxx::core::ffi::c_void,
                    >,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$MNN$Tensor$MapType$drop"]
                        fn __drop(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        );
                    }
                    unsafe {
                        __drop(&mut repr);
                    }
                }
            }
            unsafe impl ::cxx::private::SharedPtrTarget for Tensor_MapType {
                fn __typename(
                    f: &mut ::cxx::core::fmt::Formatter<'_>,
                ) -> ::cxx::core::fmt::Result {
                    f.write_str("Tensor_MapType")
                }
                unsafe fn __null(new: *mut ::cxx::core::ffi::c_void) {
                    extern "C" {
                        #[link_name = "cxxbridge1$shared_ptr$MNN$Tensor$MapType$null"]
                        fn __null(new: *mut ::cxx::core::ffi::c_void);
                    }
                    unsafe {
                        __null(new);
                    }
                }
                unsafe fn __new(value: Self, new: *mut ::cxx::core::ffi::c_void) {
                    extern "C" {
                        #[link_name = "cxxbridge1$shared_ptr$MNN$Tensor$MapType$uninit"]
                        fn __uninit(
                            new: *mut ::cxx::core::ffi::c_void,
                        ) -> *mut ::cxx::core::ffi::c_void;
                    }
                    unsafe {
                        __uninit(new).cast::<Tensor_MapType>().write(value);
                    }
                }
                unsafe fn __clone(
                    this: *const ::cxx::core::ffi::c_void,
                    new: *mut ::cxx::core::ffi::c_void,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$shared_ptr$MNN$Tensor$MapType$clone"]
                        fn __clone(
                            this: *const ::cxx::core::ffi::c_void,
                            new: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __clone(this, new);
                    }
                }
                unsafe fn __get(this: *const ::cxx::core::ffi::c_void) -> *const Self {
                    extern "C" {
                        #[link_name = "cxxbridge1$shared_ptr$MNN$Tensor$MapType$get"]
                        fn __get(
                            this: *const ::cxx::core::ffi::c_void,
                        ) -> *const ::cxx::core::ffi::c_void;
                    }
                    unsafe { __get(this).cast() }
                }
                unsafe fn __drop(this: *mut ::cxx::core::ffi::c_void) {
                    extern "C" {
                        #[link_name = "cxxbridge1$shared_ptr$MNN$Tensor$MapType$drop"]
                        fn __drop(this: *mut ::cxx::core::ffi::c_void);
                    }
                    unsafe {
                        __drop(this);
                    }
                }
            }
            unsafe impl ::cxx::private::WeakPtrTarget for Tensor_MapType {
                fn __typename(
                    f: &mut ::cxx::core::fmt::Formatter<'_>,
                ) -> ::cxx::core::fmt::Result {
                    f.write_str("Tensor_MapType")
                }
                unsafe fn __null(new: *mut ::cxx::core::ffi::c_void) {
                    extern "C" {
                        #[link_name = "cxxbridge1$weak_ptr$MNN$Tensor$MapType$null"]
                        fn __null(new: *mut ::cxx::core::ffi::c_void);
                    }
                    unsafe {
                        __null(new);
                    }
                }
                unsafe fn __clone(
                    this: *const ::cxx::core::ffi::c_void,
                    new: *mut ::cxx::core::ffi::c_void,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$weak_ptr$MNN$Tensor$MapType$clone"]
                        fn __clone(
                            this: *const ::cxx::core::ffi::c_void,
                            new: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __clone(this, new);
                    }
                }
                unsafe fn __downgrade(
                    shared: *const ::cxx::core::ffi::c_void,
                    weak: *mut ::cxx::core::ffi::c_void,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$weak_ptr$MNN$Tensor$MapType$downgrade"]
                        fn __downgrade(
                            shared: *const ::cxx::core::ffi::c_void,
                            weak: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __downgrade(shared, weak);
                    }
                }
                unsafe fn __upgrade(
                    weak: *const ::cxx::core::ffi::c_void,
                    shared: *mut ::cxx::core::ffi::c_void,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$weak_ptr$MNN$Tensor$MapType$upgrade"]
                        fn __upgrade(
                            weak: *const ::cxx::core::ffi::c_void,
                            shared: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __upgrade(weak, shared);
                    }
                }
                unsafe fn __drop(this: *mut ::cxx::core::ffi::c_void) {
                    extern "C" {
                        #[link_name = "cxxbridge1$weak_ptr$MNN$Tensor$MapType$drop"]
                        fn __drop(this: *mut ::cxx::core::ffi::c_void);
                    }
                    unsafe {
                        __drop(this);
                    }
                }
            }
            unsafe impl ::cxx::private::VectorElement for Tensor_MapType {
                fn __typename(
                    f: &mut ::cxx::core::fmt::Formatter<'_>,
                ) -> ::cxx::core::fmt::Result {
                    f.write_str("Tensor_MapType")
                }
                fn __vector_new() -> *mut ::cxx::CxxVector<Self> {
                    extern "C" {
                        #[link_name = "cxxbridge1$std$vector$MNN$Tensor$MapType$new"]
                        fn __vector_new() -> *mut ::cxx::CxxVector<Tensor_MapType>;
                    }
                    unsafe { __vector_new() }
                }
                fn __vector_size(v: &::cxx::CxxVector<Self>) -> usize {
                    extern "C" {
                        #[link_name = "cxxbridge1$std$vector$MNN$Tensor$MapType$size"]
                        fn __vector_size(_: &::cxx::CxxVector<Tensor_MapType>) -> usize;
                    }
                    unsafe { __vector_size(v) }
                }
                unsafe fn __get_unchecked(
                    v: *mut ::cxx::CxxVector<Self>,
                    pos: usize,
                ) -> *mut Self {
                    extern "C" {
                        #[link_name = "cxxbridge1$std$vector$MNN$Tensor$MapType$get_unchecked"]
                        fn __get_unchecked(
                            v: *mut ::cxx::CxxVector<Tensor_MapType>,
                            pos: usize,
                        ) -> *mut ::cxx::core::ffi::c_void;
                    }
                    unsafe { __get_unchecked(v, pos) as *mut Self }
                }
                unsafe fn __push_back(
                    this: ::cxx::core::pin::Pin<&mut ::cxx::CxxVector<Self>>,
                    value: &mut ::cxx::core::mem::ManuallyDrop<Self>,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$std$vector$MNN$Tensor$MapType$push_back"]
                        fn __push_back(
                            this: ::cxx::core::pin::Pin<
                                &mut ::cxx::CxxVector<Tensor_MapType>,
                            >,
                            value: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __push_back(
                            this,
                            value as *mut ::cxx::core::mem::ManuallyDrop<Self>
                                as *mut ::cxx::core::ffi::c_void,
                        );
                    }
                }
                unsafe fn __pop_back(
                    this: ::cxx::core::pin::Pin<&mut ::cxx::CxxVector<Self>>,
                    out: &mut ::cxx::core::mem::MaybeUninit<Self>,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$std$vector$MNN$Tensor$MapType$pop_back"]
                        fn __pop_back(
                            this: ::cxx::core::pin::Pin<
                                &mut ::cxx::CxxVector<Tensor_MapType>,
                            >,
                            out: *mut ::cxx::core::ffi::c_void,
                        );
                    }
                    unsafe {
                        __pop_back(
                            this,
                            out as *mut ::cxx::core::mem::MaybeUninit<Self>
                                as *mut ::cxx::core::ffi::c_void,
                        );
                    }
                }
                fn __unique_ptr_null() -> ::cxx::core::mem::MaybeUninit<
                    *mut ::cxx::core::ffi::c_void,
                > {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$std$vector$MNN$Tensor$MapType$null"]
                        fn __unique_ptr_null(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        );
                    }
                    let mut repr = ::cxx::core::mem::MaybeUninit::uninit();
                    unsafe {
                        __unique_ptr_null(&mut repr);
                    }
                    repr
                }
                unsafe fn __unique_ptr_raw(
                    raw: *mut ::cxx::CxxVector<Self>,
                ) -> ::cxx::core::mem::MaybeUninit<*mut ::cxx::core::ffi::c_void> {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$std$vector$MNN$Tensor$MapType$raw"]
                        fn __unique_ptr_raw(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                            raw: *mut ::cxx::CxxVector<Tensor_MapType>,
                        );
                    }
                    let mut repr = ::cxx::core::mem::MaybeUninit::uninit();
                    unsafe {
                        __unique_ptr_raw(&mut repr, raw);
                    }
                    repr
                }
                unsafe fn __unique_ptr_get(
                    repr: ::cxx::core::mem::MaybeUninit<*mut ::cxx::core::ffi::c_void>,
                ) -> *const ::cxx::CxxVector<Self> {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$std$vector$MNN$Tensor$MapType$get"]
                        fn __unique_ptr_get(
                            this: *const ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        ) -> *const ::cxx::CxxVector<Tensor_MapType>;
                    }
                    unsafe { __unique_ptr_get(&repr) }
                }
                unsafe fn __unique_ptr_release(
                    mut repr: ::cxx::core::mem::MaybeUninit<
                        *mut ::cxx::core::ffi::c_void,
                    >,
                ) -> *mut ::cxx::CxxVector<Self> {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$std$vector$MNN$Tensor$MapType$release"]
                        fn __unique_ptr_release(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        ) -> *mut ::cxx::CxxVector<Tensor_MapType>;
                    }
                    unsafe { __unique_ptr_release(&mut repr) }
                }
                unsafe fn __unique_ptr_drop(
                    mut repr: ::cxx::core::mem::MaybeUninit<
                        *mut ::cxx::core::ffi::c_void,
                    >,
                ) {
                    extern "C" {
                        #[link_name = "cxxbridge1$unique_ptr$std$vector$MNN$Tensor$MapType$drop"]
                        fn __unique_ptr_drop(
                            this: *mut ::cxx::core::mem::MaybeUninit<
                                *mut ::cxx::core::ffi::c_void,
                            >,
                        );
                    }
                    unsafe {
                        __unique_ptr_drop(&mut repr);
                    }
                }
            }
            #[doc(hidden)]
            const _: () = {
                const _: fn() = ::cxx::private::verify_extern_type::<
                    Tensor,
                    (
                        ::cxx::M,
                        ::cxx::N,
                        ::cxx::N,
                        (),
                        ::cxx::T,
                        ::cxx::e,
                        ::cxx::n,
                        ::cxx::s,
                        ::cxx::o,
                        ::cxx::r,
                    ),
                >;
                const _: fn() = ::cxx::private::verify_extern_type::<
                    halide_buffer_t,
                    (
                        ::cxx::h,
                        ::cxx::a,
                        ::cxx::l,
                        ::cxx::i,
                        ::cxx::d,
                        ::cxx::e,
                        ::cxx::__,
                        ::cxx::b,
                        ::cxx::u,
                        ::cxx::f,
                        ::cxx::f,
                        ::cxx::e,
                        ::cxx::r,
                        ::cxx::__,
                        ::cxx::t,
                    ),
                >;
                const _: fn() = ::cxx::private::verify_extern_type::<
                    Tensor_DimensionType,
                    (
                        ::cxx::M,
                        ::cxx::N,
                        ::cxx::N,
                        (),
                        ::cxx::T,
                        ::cxx::e,
                        ::cxx::n,
                        ::cxx::s,
                        ::cxx::o,
                        ::cxx::r,
                        (),
                        ::cxx::D,
                        ::cxx::i,
                        ::cxx::m,
                        ::cxx::e,
                        ::cxx::n,
                        ::cxx::s,
                        ::cxx::i,
                        ::cxx::o,
                        ::cxx::n,
                        ::cxx::T,
                        ::cxx::y,
                        ::cxx::p,
                        ::cxx::e,
                    ),
                >;
                const _: fn() = ::cxx::private::verify_extern_kind::<
                    Tensor_DimensionType,
                    ::cxx::kind::Trivial,
                >;
                const _: fn() = ::cxx::private::verify_extern_type::<
                    Tensor_HandleDataType,
                    (
                        ::cxx::M,
                        ::cxx::N,
                        ::cxx::N,
                        (),
                        ::cxx::T,
                        ::cxx::e,
                        ::cxx::n,
                        ::cxx::s,
                        ::cxx::o,
                        ::cxx::r,
                        (),
                        ::cxx::H,
                        ::cxx::a,
                        ::cxx::n,
                        ::cxx::d,
                        ::cxx::l,
                        ::cxx::e,
                        ::cxx::D,
                        ::cxx::a,
                        ::cxx::t,
                        ::cxx::a,
                        ::cxx::T,
                        ::cxx::y,
                        ::cxx::p,
                        ::cxx::e,
                    ),
                >;
                const _: fn() = ::cxx::private::verify_extern_kind::<
                    Tensor_HandleDataType,
                    ::cxx::kind::Trivial,
                >;
                const _: fn() = ::cxx::private::verify_extern_type::<
                    halide_type_t,
                    (
                        ::cxx::h,
                        ::cxx::a,
                        ::cxx::l,
                        ::cxx::i,
                        ::cxx::d,
                        ::cxx::e,
                        ::cxx::__,
                        ::cxx::t,
                        ::cxx::y,
                        ::cxx::p,
                        ::cxx::e,
                        ::cxx::__,
                        ::cxx::t,
                    ),
                >;
                const _: fn() = ::cxx::private::verify_extern_type::<
                    Tensor_MapType,
                    (
                        ::cxx::M,
                        ::cxx::N,
                        ::cxx::N,
                        (),
                        ::cxx::T,
                        ::cxx::e,
                        ::cxx::n,
                        ::cxx::s,
                        ::cxx::o,
                        ::cxx::r,
                        (),
                        ::cxx::M,
                        ::cxx::a,
                        ::cxx::p,
                        ::cxx::T,
                        ::cxx::y,
                        ::cxx::p,
                        ::cxx::e,
                    ),
                >;
                const _: fn() = ::cxx::private::verify_extern_kind::<
                    Tensor_MapType,
                    ::cxx::kind::Trivial,
                >;
                const _: fn() = ::cxx::private::verify_extern_type::<
                    c_int,
                    (::cxx::c, ::cxx::__, ::cxx::i, ::cxx::n, ::cxx::t),
                >;
                const _: fn() = ::cxx::private::verify_extern_kind::<
                    c_int,
                    ::cxx::kind::Trivial,
                >;
                const _: fn() = ::cxx::private::verify_extern_type::<
                    c_void,
                    (::cxx::c, ::cxx::__, ::cxx::v, ::cxx::o, ::cxx::i, ::cxx::d),
                >;
            };
        }
        #[allow(unused_imports)]
        use bindgen::root;
        pub use cxxbridge::autocxx_make_string_0xecc3669dd67dbc04 as make_string;
        #[allow(unused_imports)]
        pub use bindgen::root::halide_buffer_t;
        #[allow(unused_imports)]
        pub use bindgen::root::halide_type_t;
        pub mod MNN {
            #[allow(unused_imports)]
            pub use super::bindgen::root::MNN::Tensor;
            #[allow(unused_imports)]
            pub use super::bindgen::root::MNN::Tensor_DimensionType;
            #[allow(unused_imports)]
            pub use super::bindgen::root::MNN::Tensor_HandleDataType;
            #[allow(unused_imports)]
            pub use super::bindgen::root::MNN::Tensor_MapType;
        }
    }
    pub use ffi::*;
}
use mnn::*;
