use pyo3::prelude::*;

mod async_spec_decoding_engine;
mod spec_decoding_engine;

/// Initialise the logger from the ``SPEC_LOG`` environment variable.
///
/// Called automatically when the Python module is imported. Examples:
///
/// ```bash
/// SPEC_LOG=info python run.py          # acceptance stats + model loading
/// SPEC_LOG=debug python run.py         # + rollback events, per-step detail
/// SPEC_LOG=trace python run.py         # + every accepted token
/// SPEC_LOG=spec_decode=debug python …  # target only the decode crate
/// ```
fn init_logger() {
    let _ = env_logger::Builder::from_env(env_logger::Env::new().filter_or("SPEC_LOG", "warn"))
        .try_init();
}

/// Python module definition.
#[pymodule]
#[pyo3(name = "SPEC")]
mod spec_module {
    use super::*;

    #[pymodule_init]
    fn init(_m: &Bound<'_, PyModule>) -> PyResult<()> {
        init_logger();

        Ok(())
    }

    #[pymodule]
    mod engines {
        #[pymodule_export]
        use crate::spec_decoding_engine::SpecDecodingEngine;

        #[pymodule_export]
        use crate::async_spec_decoding_engine::AsyncSpecDecodingEngine;
    }
}
