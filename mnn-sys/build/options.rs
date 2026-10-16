use std::sync::LazyLock;

pub const VENDOR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/vendor");
pub const MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");

#[cfg(feature = "download")]
pub static TARGET_ARCH: LazyLock<build_target::Arch> = LazyLock::new(build_target::target_arch);
pub static TARGET_OS: LazyLock<build_target::Os> = LazyLock::new(build_target::target_os);

pub static MNN_COMPILE: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("MNN_COMPILE")
        .ok()
        .and_then(|v| match v.as_str() {
            "1" | "true" | "yes" => Some(true),
            "0" | "false" | "no" => Some(false),
            _ => None,
        })
        .unwrap_or(true)
});

pub const HALIDE_SEARCH: &str =
    r#"HALIDE_ATTRIBUTE_ALIGN(1) halide_type_code_t code; // halide_type_code_t"#;

pub const TRACING_SEARCH: &str =
    "#define MNN_PRINT(format, ...) printf(format, ##__VA_ARGS__)\n#define MNN_ERROR(format, ...) printf(format, ##__VA_ARGS__)";

pub const TRACING_REPLACE: &str = r#"
enum class Level {
  Info = 0,
  Error = 1,
};
extern "C" {
void mnn_ffi_emit(const char *file, size_t line, Level level,
                  const char *message);
}
#define MNN_PRINT(format, ...)                                                 \
  {                                                                            \
    char logtmp[4096];                                                         \
    snprintf(logtmp, 4096, format, ##__VA_ARGS__);                             \
    mnn_ffi_emit(__FILE__, __LINE__, Level::Info, logtmp);                     \
  }

#define MNN_ERROR(format, ...)                                                 \
  {                                                                            \
    char logtmp[4096];                                                         \
    snprintf(logtmp, 4096, format, ##__VA_ARGS__);                             \
    mnn_ffi_emit(__FILE__, __LINE__, Level::Error, logtmp);                    \
  }
"#;

#[derive(Debug, Clone, Copy)]
pub enum CxxOptionValue {
    On,
    Off,
    Value(&'static str),
}

impl From<bool> for CxxOptionValue {
    fn from(b: bool) -> Self {
        if b {
            Self::On
        } else {
            Self::Off
        }
    }
}

impl CxxOptionValue {
    pub const fn from_bool(value: bool) -> Self {
        match value {
            true => Self::On,
            false => Self::Off,
        }
    }
}

impl From<&'static str> for CxxOptionValue {
    fn from(s: &'static str) -> Self {
        match s {
            "ON" => Self::On,
            "OFF" => Self::Off,
            _ => Self::Value(s),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CxxOption {
    pub name: &'static str,
    pub value: CxxOptionValue,
}

macro_rules! cxx_option_from_feature {
    ($feature:literal, $cxx:literal) => {{
        CxxOption::from_bool($cxx, cfg!(feature = $feature))
    }};
}

impl CxxOption {
    pub const fn from_bool(name: &'static str, value: bool) -> Self {
        Self {
            name,
            value: CxxOptionValue::from_bool(value),
        }
    }

    pub const VULKAN: CxxOption = cxx_option_from_feature!("vulkan", "MNN_VULKAN");
    pub const CUDA: CxxOption = cxx_option_from_feature!("cuda", "MNN_CUDA");
    pub const METAL: CxxOption = cxx_option_from_feature!("metal", "MNN_METAL");
    pub const COREML: CxxOption = cxx_option_from_feature!("coreml", "MNN_COREML");
    pub const OPENCL: CxxOption = cxx_option_from_feature!("opencl", "MNN_OPENCL");
    pub const OPENMP: CxxOption = cxx_option_from_feature!("openmp", "MNN_OPENMP");
    pub const OPENGL: CxxOption = cxx_option_from_feature!("opengl", "MNN_OPENGL");
    pub const CRT_STATIC: CxxOption = cxx_option_from_feature!("crt_static", "MNN_WIN_RUNTIME_MT");
    pub const THREADPOOL: CxxOption =
        cxx_option_from_feature!("mnn-threadpool", "MNN_USE_THREAD_POOL");

    // pub fn new(name: &'static str, value: impl Into<CxxOptionValue>) -> Self {
    //     Self {
    //         name,
    //         value: value.into(),
    //     }
    // }
    //
    // pub fn on(mut self) -> Self {
    //     self.value = CxxOptionValue::On;
    //     self
    // }
    //
    // pub fn off(mut self) -> Self {
    //     self.value = CxxOptionValue::Off;
    //     self
    // }
    //
    // pub fn with_value(mut self, value: &'static str) -> Self {
    //     self.value = CxxOptionValue::Value(value);
    //     self
    // }
    //
    // pub fn cmake(&self) -> String {
    //     match &self.value {
    //         CxxOptionValue::On => format!("-D{}=ON", self.name),
    //         CxxOptionValue::Off => format!("-D{}=OFF", self.name),
    //         CxxOptionValue::Value(v) => format!("-D{}={}", self.name, v),
    //     }
    // }

    pub fn cmake_value(&self) -> &'static str {
        match &self.value {
            CxxOptionValue::On => "ON",
            CxxOptionValue::Off => "OFF",
            CxxOptionValue::Value(v) => v,
        }
    }

    pub fn cxx(&self) -> String {
        match &self.value {
            CxxOptionValue::On => format!("-D{}=1", self.name),
            CxxOptionValue::Off => format!("-D{}=0", self.name),
            CxxOptionValue::Value(v) => format!("-D{}={}", self.name, v),
        }
    }

    // pub fn enabled(&self) -> bool {
    //     match self.value {
    //         CxxOptionValue::On => true,
    //         CxxOptionValue::Off => false,
    //         CxxOptionValue::Value(_) => true,
    //     }
    // }
}
