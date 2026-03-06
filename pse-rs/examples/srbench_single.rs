//! Single SRBench problem runner for testing
//!
//! Run with: cargo run --example srbench_single --release

use ndarray::Array2;
use pse::{Device, FitConfig, PSRNConfig, PSRNRegressor, TokenGenerator};
use pyo3::prelude::*;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pse_path = std::env::var("PSE_PATH").expect("PSE_PATH environment variable must be set");
    let benchmark_name = std::env::var("BENCHMARK").unwrap_or_else(|_| "Nguyen-1".to_string());

    println!("Running benchmark: {}", benchmark_name);

    // Load data using Python
    let (x, y, variables, use_const, expression) = Python::with_gil(|py| -> PyResult<_> {
        let sys = py.import_bound("sys")?;
        let path = sys.getattr("path")?;
        path.call_method1("insert", (0, &pse_path))?;

        let os = py.import_bound("os")?;
        os.call_method1("chdir", (&pse_path,))?;

        let data_module = py.import_bound("utils.data")?;
        let get_data = data_module.getattr("get_benchmark_data")?;

        let result = get_data.call1(("benchmark.csv", &benchmark_name))?;
        let tuple = result.downcast::<pyo3::types::PyTuple>()?;

        let x_py = tuple.get_item(0)?;
        let x_shape: Vec<usize> = x_py.getattr("shape")?.extract()?;
        let x_flat: Vec<f64> = x_py
            .getattr("flatten")?
            .call0()?
            .getattr("tolist")?
            .call0()?
            .extract()?;
        let x = Array2::from_shape_vec((x_shape[0], x_shape[1]), x_flat).unwrap();

        let y_py = tuple.get_item(1)?;
        let y: Vec<f64> = y_py
            .getattr("flatten")?
            .call0()?
            .getattr("tolist")?
            .call0()?
            .extract()?;

        let use_const: bool = tuple.get_item(2)?.extract()?;
        let expression: String = tuple.get_item(3)?.extract()?;
        let variables: Vec<String> = tuple.get_item(4)?.extract()?;

        Ok((x, y, variables, use_const, expression))
    })?;

    println!("Expression: {}", expression);
    println!("Variables: {:?}", variables);
    println!("Data shape: {:?}", x.shape());
    println!("Use constants: {}", use_const);
    println!();

    // Configure regressor
    let pse_dir = PathBuf::from(&pse_path);
    let config = PSRNConfig {
        variables: variables.clone(),
        operators: None,
        n_symbol_layers: 3,
        use_const,
        use_dr_mask: false,
        dr_mask_dir: pse_dir.join("dr_mask"),
        stage_config: pse_dir.join("model/stages_config/benchmark.yaml"),
        token_generator_config: pse_dir.join("token_generator_config.yaml"),
        token_generator: TokenGenerator::GP,
        device: Device::Cpu,
        ..Default::default()
    };

    let mut regressor = PSRNRegressor::new(config)?;

    let fit_config = FitConfig {
        n_down_sample: 20,
        eta: 0.99,
        use_threshold: true,
        threshold: 1e-10,
        prun_const: true,
        prun_ndigit: 2,
        top_k: 10,
        real_time_display: true,
        ..Default::default()
    };

    println!("Fitting...");
    let start = Instant::now();
    let result = regressor.fit(x.view(), &y, fit_config)?;
    let elapsed = start.elapsed();

    println!("\n=== Results ===");
    println!("Converged: {}", result.converged);
    println!("Time: {:.2}s", elapsed.as_secs_f64());
    println!(
        "Pareto frontier ({} expressions):",
        result.pareto_frontier.len()
    );

    for (i, expr) in result.pareto_frontier.iter().enumerate() {
        println!(
            "  {}. {} (MSE: {:.2e}, complexity: {:.0})",
            i + 1,
            expr.expression,
            expr.mse,
            expr.complexity
        );
    }

    if let Some(best) = result.best_expression() {
        println!("\nBest expression: {best:?}");
    }

    Ok(())
}
