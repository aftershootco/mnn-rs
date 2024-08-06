use std::sync::Arc;

use libc::c_void;

pub struct Expr;
pub struct BufferStorage;
pub struct OpT;
pub struct Op;
pub struct NetT;
pub struct Variable;

pub struct Varp {
    inner: *mut c_void,
    __phantom: std::marker::PhantomData<Arc<Variable>>,
}
