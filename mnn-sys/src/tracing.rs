// This is mostly adapted from tracing-gstreamer crate's implementation
use once_cell::sync::OnceCell;
use std::sync::atomic::AtomicUsize;
use std::sync::{PoisonError, RwLock};
use std::{collections::BTreeMap, ffi::c_char};
use tracing_core::{field::FieldSet, identify_callsite, Callsite, Interest, Kind, Metadata};

pub const CALLSITE_INTEREST_NEVER: usize = 1;
pub const CALLSITE_INTEREST_SOMETIMES: usize = 2;
pub const CALLSITE_INTEREST_ALWAYS: usize = 3;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
pub enum Level {
    Info = 0,
    Error = 1,
}

impl From<Level> for tracing_core::Level {
    fn from(value: Level) -> Self {
        match value {
            Level::Info => tracing_core::Level::INFO,
            Level::Error => tracing_core::Level::ERROR,
        }
    }
}

pub struct DynamicCallsites {
    callsites: RwLock<Map>,
}

type Map = BTreeMap<Key<'static>, &'static MnnCallsite>;

impl DynamicCallsites {
    pub(crate) fn get() -> &'static Self {
        static MAP: OnceCell<DynamicCallsites> = OnceCell::new();
        MAP.get_or_init(|| DynamicCallsites {
            callsites: RwLock::new(Map::new()),
        })
    }

    fn callsite_for(
        &'static self,
        level: Level,
        line: Option<u32>,
        file: Option<&'static str>,
    ) -> &'static MnnCallsite {
        let mut guard = self
            .callsites
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        let lookup_key = Key { level, line, file };
        if let Some(callsite) = guard.get(&lookup_key) {
            return callsite;
        }
        let callsite = MnnCallsite::make_static(&lookup_key);
        let key = Key::<'static> {
            level,
            line,
            file: callsite.metadata.file(),
        };
        guard.insert(key, callsite);
        tracing_core::callsite::register(callsite);
        callsite
    }
}

impl Callsite for MnnCallsite {
    fn set_interest(&self, interest: Interest) {
        self.interest.store(
            match () {
                _ if interest.is_never() => CALLSITE_INTEREST_NEVER,
                _ if interest.is_always() => CALLSITE_INTEREST_ALWAYS,
                _ => CALLSITE_INTEREST_SOMETIMES,
            },
            std::sync::atomic::Ordering::Release,
        );
    }

    fn metadata(&self) -> &Metadata<'_> {
        &self.metadata
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Key<'k> {
    level: Level,
    line: Option<u32>,
    file: Option<&'k str>,
}

impl DynamicCallsites {}

pub struct MnnCallsite {
    interest: AtomicUsize,
    metadata: Metadata<'static>,
}

impl MnnCallsite {
    pub fn make_static(key: &Key<'static>) -> &'static Self {
        unsafe {
            use std::alloc::GlobalAlloc as _;
            let callsite_layout = std::alloc::Layout::new::<MnnCallsite>();
            let alloc = std::alloc::System.alloc(callsite_layout);
            let callsite = alloc as *mut MnnCallsite;
            // No allocation for string required as they are static by default
            callsite.write(MnnCallsite {
                interest: AtomicUsize::new(0),
                metadata: Metadata::new(
                    "",
                    "mnn_ffi_emit",
                    key.level.into(),
                    key.file,
                    key.line,
                    None,
                    FieldSet::new(&["message"], identify_callsite!(&*callsite)),
                    Kind::EVENT,
                ),
            });
            &*callsite
        }
    }

    pub(crate) fn interest(&self) -> Interest {
        match self.interest.load(std::sync::atomic::Ordering::Acquire) {
            CALLSITE_INTEREST_NEVER => Interest::never(),
            CALLSITE_INTEREST_SOMETIMES => Interest::sometimes(),
            CALLSITE_INTEREST_ALWAYS => Interest::always(),
            _ => panic!("attempting to obtain callsite's interest before its been set"),
        }
    }
}

#[no_mangle]
extern "C" fn mnn_ffi_emit(
    file: *const c_char,
    line: libc::size_t,
    level: Level,
    message: *const c_char,
) {
    std::panic::catch_unwind(|| {
        let file: &'static str = unsafe {
            core::ffi::CStr::from_ptr(file)
                .to_str()
                .expect("Invalid filename for C file")
        };

        let callsite = DynamicCallsites::get().callsite_for(level, Some(line as u32), Some(file));
        // let interest = callsite.interest
        let interest = callsite.interest();
        if interest.is_never() {
            return;
        }
        let meta = callsite.metadata();
        tracing_core::dispatcher::get_default(move |dispatcher| {
            if !dispatcher.enabled(meta) {
                return;
            }
            let fields = meta.fields();
            let message = unsafe {
                std::ffi::CStr::from_ptr(message)
                    .to_str()
                    .expect("Invalid message for C message")
            };

            let message_value = &message as &dyn tracing_core::field::Value;
            let message_field = fields
                .into_iter()
                .next()
                .expect("Failed to get message field");
            let values = &[(&message_field, Some(message_value))];
            let valueset = fields.value_set(values);

            let event = tracing_core::Event::new(meta, &valueset);

            dispatcher.event(&event);
        });
    })
    .unwrap_or_else(|_e| {
        eprintln!("Panic in mnn_ffi_emit aborting");
        // Cannot let the panic escape the ffi boundary
        std::process::abort();
    })
}
