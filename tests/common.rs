pub type Error = Box<dyn std::error::Error + Send + Sync + 'static>;
pub type Result<T, E = Error> = std::result::Result<T, E>;
pub struct Model {
    bytes: &'static [u8],
}

impl Model {
    pub const fn new() -> Self {
        Model {
            bytes: include_bytes!("assets/realesr.mnn"),
        }
    }
}

impl AsRef<[u8]> for Model {
    fn as_ref(&self) -> &[u8] {
        self.bytes
    }
}
