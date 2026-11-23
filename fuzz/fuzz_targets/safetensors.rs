#![no_main]

use libfuzzer_sys::fuzz_target;
use mtab_core::safetensors::{load_weights_bytes, WeightsInfo};

fuzz_target!(|data: &[u8]| {
    // A bounded image avoids resource exhaustion from inputs supplied by the
    // fuzzer itself; declared shapes and offsets remain unconstrained.
    if data.len() > 1024 * 1024 {
        return;
    }
    let parsed = WeightsInfo::from_bytes(data);
    let loaded = load_weights_bytes(data);
    if let Ok(tensors) = loaded {
        assert!(parsed.is_ok());
        assert_eq!(parsed.unwrap().tensor_count(), tensors.len());
    }
});
