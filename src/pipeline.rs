use std::path::Path;

use crate::prelude::*;
pub struct Pipeline {
    inner: *mut c_void,
    __marker: PhantomData<()>,
}

impl Pipeline {
    pub fn new(path: impl AsRef<Path>, model_type: i32) {
        
    }
}
