//! Rust bindings for PSE (PSRN Symbolic Regression)
//!
//! This crate provides a Rust interface to the PSE Python library for
//! symbolic regression using PSRN (Physical Symbol Regression Network).
//!
//! # Example
//!
//! ```no_run
//! use pse::{PSRNRegressor, PSRNConfig, FitConfig};
//! use ndarray::Array2;
//!
//! let config = PSRNConfig::default();
//! let mut regressor = PSRNRegressor::new(config).unwrap();
//!
//! let x = Array2::from_shape_vec((100, 2), (0..200).map(|i| i as f64 * 0.1).collect()).unwrap();
//! let y: Vec<f64> = x.rows().into_iter().map(|row| row[0] * row[1]).collect();
//!
//! let fit_config = FitConfig::default();
//! let result = regressor.fit(x.view(), &y, fit_config).unwrap();
//!
//! if let Some(expr) = result.best_expression() {
//!     println!("Best expression: {}", expr);
//! }
//! ```

mod error;

pub use error::{Error, Result};

use ndarray::ArrayView2;
use numpy::{PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::path::PathBuf;

/// Token generator algorithm for expression search
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TokenGenerator {
    /// Genetic Programming (default)
    #[default]
    GP,
    /// Monte Carlo Tree Search
    MCTS,
    /// Random search
    Random,
}

impl TokenGenerator {
    fn as_str(&self) -> &'static str {
        match self {
            TokenGenerator::GP => "GP",
            TokenGenerator::MCTS => "MCTS",
            TokenGenerator::Random => "Random",
        }
    }
}

/// Sort criteria for Pareto frontier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SortBy {
    /// Sort by reward (descending)
    #[default]
    Reward,
    /// Sort by MSE (ascending)
    Mse,
    /// Sort by complexity (ascending)
    Complexity,
}

impl SortBy {
    fn as_str(&self) -> &'static str {
        match self {
            SortBy::Reward => "reward",
            SortBy::Mse => "mse",
            SortBy::Complexity => "complexity",
        }
    }
}

/// Configuration for PSRN Regressor initialization
#[derive(Debug, Clone)]
pub struct PSRNConfig {
    /// Variable names (default: ["x"])
    pub variables: Vec<String>,
    /// Operators to use (default: from stage config)
    pub operators: Option<Vec<String>>,
    /// Number of symbol layers (default: 3)
    pub n_symbol_layers: usize,
    /// Number of inputs (default: from stage config)
    pub n_inputs: Option<usize>,
    /// Use dimensionality reduction mask (default: true)
    pub use_dr_mask: bool,
    /// Directory for DR mask files
    pub dr_mask_dir: PathBuf,
    /// Use constants in expressions (default: false)
    pub use_const: bool,
    /// Use extra constants (default: false)
    pub use_extra_const: bool,
    /// Number of sample variables
    pub n_sample_variables: Option<usize>,
    /// Path to stage config YAML
    pub stage_config: PathBuf,
    /// Path to token generator config YAML
    pub token_generator_config: PathBuf,
    /// Token generator algorithm
    pub token_generator: TokenGenerator,
    /// Device to use ("cuda" or "cpu")
    pub device: String,
}

impl Default for PSRNConfig {
    fn default() -> Self {
        Self {
            variables: vec!["x".to_string()],
            operators: None,
            n_symbol_layers: 3,
            n_inputs: None,
            use_dr_mask: true,
            dr_mask_dir: PathBuf::from("./dr_mask"),
            use_const: false,
            use_extra_const: false,
            n_sample_variables: None,
            stage_config: PathBuf::from("model/stages_config/benchmark.yaml"),
            token_generator_config: PathBuf::from("token_generator_config.yaml"),
            token_generator: TokenGenerator::GP,
            device: "cuda".to_string(),
        }
    }
}

/// Configuration for the fit method
#[derive(Debug, Clone)]
pub struct FitConfig {
    /// Number of samples for downsampling during PSRN forward (default: 20)
    pub n_down_sample: usize,
    /// Eta parameter for reward calculation (default: 0.99)
    pub eta: f64,
    /// Use MSE threshold for early stopping (default: true)
    pub use_threshold: bool,
    /// MSE threshold value (default: 1e-10)
    pub threshold: f64,
    /// Prune constants to fewer digits (default: true)
    pub prun_const: bool,
    /// Number of digits for constant pruning (default: 6)
    pub prun_ndigit: usize,
    /// Display real-time progress (default: true)
    pub real_time_display: bool,
    /// Frequency of real-time display updates (default: 1)
    pub real_time_display_freq: usize,
    /// Number of top expressions to display (default: 20)
    pub real_time_display_ntop: usize,
    /// Add bias term to expressions (default: true)
    pub add_bias: bool,
    /// Simplify expressions together (default: false)
    pub together: bool,
    /// Top-k expressions to consider (default: 30)
    pub top_k: usize,
    /// Use strict Pareto dominance (default: true)
    pub use_strict_pareto: bool,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            n_down_sample: 20,
            eta: 0.99,
            use_threshold: true,
            threshold: 1e-10,
            prun_const: true,
            prun_ndigit: 6,
            real_time_display: true,
            real_time_display_freq: 1,
            real_time_display_ntop: 20,
            add_bias: true,
            together: false,
            top_k: 30,
            use_strict_pareto: true,
        }
    }
}

/// An expression on the Pareto frontier
#[derive(Debug, Clone)]
pub struct ParetoExpression {
    /// The symbolic expression string
    pub expression: String,
    /// Reward value
    pub reward: f64,
    /// Mean squared error
    pub mse: f64,
    /// Expression complexity
    pub complexity: f64,
}

/// Result of fitting the regressor
#[derive(Debug, Clone)]
pub struct FitResult {
    /// Whether the threshold was reached
    pub converged: bool,
    /// Pareto frontier of expressions
    pub pareto_frontier: Vec<ParetoExpression>,
}

impl FitResult {
    /// Get the best expression by MSE
    pub fn best_expression(&self) -> Option<&str> {
        self.pareto_frontier
            .iter()
            .min_by(|a, b| a.mse.partial_cmp(&b.mse).unwrap())
            .map(|e| e.expression.as_str())
    }

    /// Get the simplest expression
    pub fn simplest_expression(&self) -> Option<&str> {
        self.pareto_frontier
            .iter()
            .min_by(|a, b| a.complexity.partial_cmp(&b.complexity).unwrap())
            .map(|e| e.expression.as_str())
    }

    /// Get the expression with highest reward
    pub fn highest_reward_expression(&self) -> Option<&str> {
        self.pareto_frontier
            .iter()
            .max_by(|a, b| a.reward.partial_cmp(&b.reward).unwrap())
            .map(|e| e.expression.as_str())
    }
}

/// PSRN Symbolic Regressor
///
/// Wraps the Python PSRN_Regressor class for use from Rust.
pub struct PSRNRegressor {
    regressor: Py<PyAny>,
    variables: Vec<String>,
}

impl PSRNRegressor {
    /// Create a new PSRN Regressor with the given configuration
    pub fn new(config: PSRNConfig) -> Result<Self> {
        Python::with_gil(|py| {
            Self::add_pse_to_path(py)?;

            let regressor_module = py.import_bound("model.regressor")?;
            let regressor_class = regressor_module.getattr("PSRN_Regressor")?;

            let kwargs = PyDict::new_bound(py);
            let variables_list = PyList::new_bound(py, &config.variables);
            kwargs.set_item("variables", variables_list)?;

            if let Some(ref ops) = config.operators {
                let ops_list = PyList::new_bound(py, ops);
                kwargs.set_item("operators", ops_list)?;
            }

            kwargs.set_item("n_symbol_layers", config.n_symbol_layers)?;

            if let Some(n) = config.n_inputs {
                kwargs.set_item("n_inputs", n)?;
            }

            kwargs.set_item("use_dr_mask", config.use_dr_mask)?;
            kwargs.set_item("dr_mask_dir", config.dr_mask_dir.to_string_lossy().as_ref())?;
            kwargs.set_item("use_const", config.use_const)?;
            kwargs.set_item("use_extra_const", config.use_extra_const)?;

            if let Some(n) = config.n_sample_variables {
                kwargs.set_item("n_sample_variables", n)?;
            }

            kwargs.set_item(
                "stage_config",
                config.stage_config.to_string_lossy().as_ref(),
            )?;
            kwargs.set_item(
                "token_generator_config",
                config.token_generator_config.to_string_lossy().as_ref(),
            )?;
            kwargs.set_item("token_generator", config.token_generator.as_str())?;
            kwargs.set_item("device", &config.device)?;

            let regressor = regressor_class.call((), Some(&kwargs))?;

            Ok(Self {
                regressor: regressor.into(),
                variables: config.variables,
            })
        })
    }

    /// Add PSE directory to Python path
    fn add_pse_to_path(py: Python<'_>) -> Result<()> {
        let sys = py.import_bound("sys")?;
        let path = sys.getattr("path")?;

        // Try to find PSE directory relative to current working directory
        // or use environment variable PSE_PATH
        let pse_path = std::env::var("PSE_PATH").unwrap_or_else(|_| ".".to_string());

        path.call_method1("insert", (0, pse_path))?;
        Ok(())
    }

    /// Fit the regressor to data
    ///
    /// # Arguments
    /// * `x` - Input features of shape (n_samples, n_features)
    /// * `y` - Target values of shape (n_samples,)
    /// * `config` - Fit configuration options
    ///
    /// # Returns
    /// * `FitResult` containing convergence status and Pareto frontier
    pub fn fit(&mut self, x: ArrayView2<f64>, y: &[f64], config: FitConfig) -> Result<FitResult> {
        Python::with_gil(|py| {
            let x_py = x.to_pyarray_bound(py);
            let y_py = PyArray1::from_slice_bound(py, y);

            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("n_down_sample", config.n_down_sample)?;
            kwargs.set_item("eta", config.eta)?;
            kwargs.set_item("use_threshold", config.use_threshold)?;
            kwargs.set_item("threshold", config.threshold)?;
            kwargs.set_item("prun_const", config.prun_const)?;
            kwargs.set_item("prun_ndigit", config.prun_ndigit)?;
            kwargs.set_item("real_time_display", config.real_time_display)?;
            kwargs.set_item("real_time_display_freq", config.real_time_display_freq)?;
            kwargs.set_item("real_time_display_ntop", config.real_time_display_ntop)?;
            kwargs.set_item("add_bias", config.add_bias)?;
            kwargs.set_item("together", config.together)?;
            kwargs.set_item("top_k", config.top_k)?;
            kwargs.set_item("use_strict_pareto", config.use_strict_pareto)?;

            let result = self
                .regressor
                .bind(py)
                .call_method("fit", (x_py, y_py), Some(&kwargs))?;

            let tuple = result.downcast::<pyo3::types::PyTuple>()?;
            let converged: bool = tuple.get_item(0)?.extract()?;
            let pareto_list = tuple.get_item(1)?;

            let pareto_frontier = self.extract_pareto_frontier(&pareto_list)?;

            Ok(FitResult {
                converged,
                pareto_frontier,
            })
        })
    }

    /// Extract Pareto frontier from Python list
    fn extract_pareto_frontier(&self, pareto_list: &Bound<'_, PyAny>) -> Result<Vec<ParetoExpression>> {
        let mut frontier = Vec::new();

        let iter = pareto_list.iter()?;
        for item in iter {
            let item = item?;
            let tuple = item.downcast::<pyo3::types::PyTuple>()?;

            let expression: String = tuple.get_item(0)?.extract()?;
            let reward: f64 = tuple.get_item(1)?.extract()?;
            let mse: f64 = tuple.get_item(2)?.extract()?;
            let complexity: f64 = tuple.get_item(3)?.extract()?;

            frontier.push(ParetoExpression {
                expression,
                reward,
                mse,
                complexity,
            });
        }

        Ok(frontier)
    }

    /// Predict target values for new input data
    ///
    /// Uses the best expression (by MSE) from the Pareto frontier.
    ///
    /// # Arguments
    /// * `x` - Input features of shape (n_samples, n_features)
    ///
    /// # Returns
    /// * Predicted values as a Vec<f64>
    pub fn predict(&self, x: ArrayView2<f64>) -> Result<Vec<f64>> {
        Python::with_gil(|py| {
            let x_py = x.to_pyarray_bound(py);

            let result = self.regressor.bind(py).call_method1("predict", (x_py,))?;

            // Result is a 2D array of shape (n_samples, 1), flatten to 1D
            let array = result.downcast::<PyArray2<f64>>()?;
            let owned = array.to_owned_array();
            let (vec, _offset) = owned.into_raw_vec_and_offset();
            Ok(vec)
        })
    }

    /// Get the Pareto frontier sorted by the specified criterion
    pub fn get_pareto_frontier(&self, sort_by: SortBy) -> Result<Vec<ParetoExpression>> {
        Python::with_gil(|py| {
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("sort_by", sort_by.as_str())?;

            let result = self
                .regressor
                .bind(py)
                .call_method("get_pf", (), Some(&kwargs))?;

            self.extract_pareto_frontier(&result)
        })
    }

    /// Get the variable names used by this regressor
    pub fn variables(&self) -> &[String] {
        &self.variables
    }

    /// Get regressor parameters
    pub fn get_params(&self) -> Result<RegressorParams> {
        Python::with_gil(|py| {
            let result = self.regressor.bind(py).call_method0("get_params")?;
            let dict = result.downcast::<PyDict>()?;

            let variables: Vec<String> = dict.get_item("variables")?.unwrap().extract()?;
            let operators: Vec<String> = dict.get_item("operators")?.unwrap().extract()?;
            let n_symbol_layers: usize = dict.get_item("n_symbol_layers")?.unwrap().extract()?;
            let n_inputs: usize = dict.get_item("n_inputs")?.unwrap().extract()?;

            let const_range = dict.get_item("trying_const_range")?.unwrap();
            let const_range_list = const_range.downcast::<PyList>()?;
            let trying_const_range = (
                const_range_list.get_item(0)?.extract::<f64>()?,
                const_range_list.get_item(1)?.extract::<f64>()?,
            );

            Ok(RegressorParams {
                variables,
                operators,
                n_symbol_layers,
                n_inputs,
                trying_const_range,
            })
        })
    }
}

/// Parameters of a fitted regressor
#[derive(Debug, Clone)]
pub struct RegressorParams {
    pub variables: Vec<String>,
    pub operators: Vec<String>,
    pub n_symbol_layers: usize,
    pub n_inputs: usize,
    pub trying_const_range: (f64, f64),
}

/// Evaluate an expression string on input data
///
/// Utility function for evaluating symbolic expressions without
/// needing a full regressor instance.
pub fn evaluate_expression(expr: &str, x: ArrayView2<f64>, variables: &[&str]) -> Result<Vec<f64>> {
    Python::with_gil(|py| {
        PSRNRegressor::add_pse_to_path(py)?;

        let utils_module = py.import_bound("utils.evaluate")?;
        let expr_to_y = utils_module.getattr("expr_to_Y_pred")?;

        let x_py = x.to_pyarray_bound(py);
        let vars_list = PyList::new_bound(py, variables);

        let result = expr_to_y.call1((expr, x_py, vars_list))?;
        let array = result.downcast::<PyArray1<f64>>()?;

        let (vec, _offset) = array.to_owned_array().into_raw_vec_and_offset();
        Ok(vec)
    })
}

/// Calculate the complexity of a symbolic expression
pub fn expression_complexity(expr: &str) -> Result<f64> {
    Python::with_gil(|py| {
        PSRNRegressor::add_pse_to_path(py)?;

        let utils_module = py.import_bound("utils.evaluate")?;
        let get_complexity = utils_module.getattr("get_sympy_complexity")?;

        let result = get_complexity.call1((expr,))?;
        Ok(result.extract()?)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = PSRNConfig::default();
        assert_eq!(config.variables, vec!["x"]);
        assert_eq!(config.n_symbol_layers, 3);
        assert!(!config.use_const);
    }

    #[test]
    fn test_fit_config_default() {
        let config = FitConfig::default();
        assert_eq!(config.n_down_sample, 20);
        assert!((config.eta - 0.99).abs() < 1e-10);
        assert!(config.use_threshold);
    }

    #[test]
    fn test_token_generator_str() {
        assert_eq!(TokenGenerator::GP.as_str(), "GP");
        assert_eq!(TokenGenerator::MCTS.as_str(), "MCTS");
        assert_eq!(TokenGenerator::Random.as_str(), "Random");
    }
}
