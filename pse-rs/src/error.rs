//! Error types for the PSE Rust bindings

use thiserror::Error;

/// Result type for PSE operations
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur when using PSE
#[derive(Error, Debug)]
pub enum Error {
    /// Python interpreter error
    #[error("Python error: {0}")]
    Python(#[from] pyo3::PyErr),

    /// NumPy array conversion error
    #[error("NumPy error: {0}")]
    NumPy(#[from] numpy::FromVecError),

    /// Invalid expression
    #[error("Invalid expression: {0}")]
    InvalidExpression(String),

    /// Fitting failed
    #[error("Fitting failed: {0}")]
    FitFailed(String),

    /// PSE module not found
    #[error("PSE Python module not found. Set PSE_PATH environment variable to the PSE directory.")]
    ModuleNotFound,

    /// Shape mismatch
    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    /// Downcasting error
    #[error("Type error: {0}")]
    DowncastError(String),
}

impl From<pyo3::DowncastError<'_, '_>> for Error {
    fn from(err: pyo3::DowncastError) -> Self {
        Error::DowncastError(err.to_string())
    }
}

impl<'py> From<pyo3::DowncastIntoError<'py>> for Error {
    fn from(err: pyo3::DowncastIntoError<'py>) -> Self {
        Error::DowncastError(err.to_string())
    }
}
