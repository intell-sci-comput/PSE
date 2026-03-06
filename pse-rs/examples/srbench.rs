//! SRBench evaluation runner using Rust PSE bindings
//!
//! Run with: cargo run --example srbench --release
//!
//! Options:
//!   PSE_PATH=/path/to/PSE (required)
//!   BENCHMARK_FILE=benchmark.csv (default)
//!   N_RUNS=1 (default)
//!   TOKEN_GENERATOR=GP (default, or MCTS, Random)

use ndarray::Array2;
use pse::{Device, FitConfig, PSRNConfig, PSRNRegressor, TokenGenerator};
use pyo3::prelude::*;
use std::path::PathBuf;
use std::time::Instant;

/// Benchmark problem definition
struct BenchmarkProblem {
    name: String,
    x: Array2<f64>,
    y: Vec<f64>,
    variables: Vec<String>,
    use_const: bool,
    expression: String,
}

/// Result of a single benchmark run
struct BenchmarkResult {
    name: String,
    success: bool,
    time_secs: f64,
    mse: f64,
    complexity: f64,
    best_expr: String,
}

/// Load benchmark data using Python utils
fn load_benchmark_data(
    pse_path: &str,
    benchmark_file: &str,
    benchmark_name: &str,
) -> PyResult<BenchmarkProblem> {
    Python::with_gil(|py| {
        // Add PSE to path and change working directory
        let sys = py.import_bound("sys")?;
        let path = sys.getattr("path")?;
        path.call_method1("insert", (0, pse_path))?;

        // Change to PSE directory so relative paths work
        let os = py.import_bound("os")?;
        os.call_method1("chdir", (pse_path,))?;

        // Import data utils
        let data_module = py.import_bound("utils.data")?;
        let get_data = data_module.getattr("get_benchmark_data")?;

        // Call get_benchmark_data
        let result = get_data.call1((benchmark_file, benchmark_name))?;
        let tuple = result.downcast::<pyo3::types::PyTuple>()?;

        // Extract X (numpy array)
        let x_py = tuple.get_item(0)?;
        let x_shape: Vec<usize> = x_py.getattr("shape")?.extract()?;
        let x_flat: Vec<f64> = x_py
            .getattr("flatten")?
            .call0()?
            .getattr("tolist")?
            .call0()?
            .extract()?;
        let x = Array2::from_shape_vec((x_shape[0], x_shape[1]), x_flat)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        // Extract Y (numpy array)
        let y_py = tuple.get_item(1)?;
        let y_flat: Vec<f64> = y_py
            .getattr("flatten")?
            .call0()?
            .getattr("tolist")?
            .call0()?
            .extract()?;

        // Extract use_constant
        let use_const: bool = tuple.get_item(2)?.extract()?;

        // Extract expression
        let expression: String = tuple.get_item(3)?.extract()?;

        // Extract variables
        let variables: Vec<String> = tuple.get_item(4)?.extract()?;

        Ok(BenchmarkProblem {
            name: benchmark_name.to_string(),
            x,
            y: y_flat,
            variables,
            use_const,
            expression,
        })
    })
}

/// Get list of benchmark names from CSV
fn get_benchmark_names(pse_path: &str, benchmark_file: &str) -> PyResult<Vec<String>> {
    Python::with_gil(|py| {
        let sys = py.import_bound("sys")?;
        let path = sys.getattr("path")?;
        path.call_method1("insert", (0, pse_path))?;

        let pandas = py.import_bound("pandas")?;
        let read_csv = pandas.getattr("read_csv")?;

        let csv_path = format!("{}/benchmark/{}", pse_path, benchmark_file);
        let df = read_csv.call1((csv_path,))?;
        let names = df.get_item("name")?.getattr("tolist")?.call0()?;

        names.extract()
    })
}

/// Run benchmark on a single problem
fn run_benchmark(
    pse_path: &str,
    problem: &BenchmarkProblem,
    token_generator: TokenGenerator,
) -> Result<BenchmarkResult, pse::Error> {
    let pse_dir = PathBuf::from(pse_path);

    let config = PSRNConfig {
        variables: problem.variables.clone(),
        operators: None, // Use default from stage config
        n_symbol_layers: 3,
        use_const: problem.use_const,
        use_dr_mask: false, // Disable for simplicity
        dr_mask_dir: pse_dir.join("dr_mask"),
        stage_config: pse_dir.join("model/stages_config/benchmark.yaml"),
        token_generator_config: pse_dir.join("token_generator_config.yaml"),
        token_generator,
        device: Device::Cpu,
        ..Default::default()
    };

    let mut regressor = PSRNRegressor::new(config)?;

    let fit_config = FitConfig {
        n_down_sample: 40,
        eta: 0.99,
        use_threshold: true,
        threshold: 1e-14,
        prun_const: true,
        prun_ndigit: 2,
        top_k: 10,
        real_time_display: false,
        ..Default::default()
    };

    let start = Instant::now();
    let result = regressor.fit(problem.x.view(), &problem.y, fit_config)?;
    let elapsed = start.elapsed().as_secs_f64();

    let best = result
        .pareto_frontier
        .iter()
        .min_by(|a, b| a.mse.partial_cmp(&b.mse).unwrap());

    match best {
        Some(expr) => Ok(BenchmarkResult {
            name: problem.name.clone(),
            success: result.converged,
            time_secs: elapsed,
            mse: expr.mse,
            complexity: expr.complexity,
            best_expr: expr.expression.clone(),
        }),
        None => Ok(BenchmarkResult {
            name: problem.name.clone(),
            success: false,
            time_secs: elapsed,
            mse: f64::INFINITY,
            complexity: 0.0,
            best_expr: "None".to_string(),
        }),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get configuration from environment
    let pse_path = std::env::var("PSE_PATH").expect("PSE_PATH environment variable must be set");
    let benchmark_file =
        std::env::var("BENCHMARK_FILE").unwrap_or_else(|_| "benchmark.csv".to_string());
    let n_runs: usize = std::env::var("N_RUNS")
        .unwrap_or_else(|_| "1".to_string())
        .parse()?;
    let token_gen_str = std::env::var("TOKEN_GENERATOR").unwrap_or_else(|_| "GP".to_string());

    let token_generator = match token_gen_str.as_str() {
        "MCTS" => TokenGenerator::MCTS,
        "Random" => TokenGenerator::Random,
        _ => TokenGenerator::GP,
    };

    println!("=== SRBench Evaluation (Rust) ===");
    println!("PSE_PATH: {}", pse_path);
    println!("Benchmark file: {}", benchmark_file);
    println!("N_RUNS: {}", n_runs);
    println!("Token generator: {:?}", token_generator);
    println!();

    // Get benchmark names
    let names = get_benchmark_names(&pse_path, &benchmark_file)?;
    println!("Found {} benchmark problems", names.len());
    println!();

    // Results storage
    let mut all_results: Vec<BenchmarkResult> = Vec::new();

    // Run each benchmark
    for name in &names {
        println!("=== {} ===", name);

        // Load benchmark data
        let problem = match load_benchmark_data(&pse_path, &benchmark_file, name) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("  Error loading {}: {}", name, e);
                continue;
            }
        };

        println!("  Expression: {}", problem.expression);
        println!("  Variables: {:?}", problem.variables);
        println!("  Data shape: {:?}", problem.x.shape());
        println!("  Use constants: {}", problem.use_const);

        let mut successes = 0;
        let mut total_time = 0.0;
        let mut best_mse = f64::INFINITY;
        let mut best_expr = String::new();

        for run in 0..n_runs {
            print!("  Run {}/{}: ", run + 1, n_runs);

            match run_benchmark(&pse_path, &problem, token_generator) {
                Ok(result) => {
                    println!(
                        "MSE={:.2e}, time={:.2}s, expr={}",
                        result.mse, result.time_secs, result.best_expr
                    );

                    if result.success {
                        successes += 1;
                    }
                    total_time += result.time_secs;

                    if result.mse < best_mse {
                        best_mse = result.mse;
                        best_expr = result.best_expr.clone();
                    }

                    all_results.push(result);
                }
                Err(e) => {
                    println!("Error: {}", e);
                }
            }
        }

        let recovery_rate = successes as f64 / n_runs as f64;
        let avg_time = total_time / n_runs as f64;

        println!(
            "  Summary: recovery_rate={:.1}%, avg_time={:.2}s",
            recovery_rate * 100.0,
            avg_time
        );
        println!("  Best: MSE={:.2e}, expr={}", best_mse, best_expr);
        println!();
    }

    // Print final summary
    println!("=== Final Summary ===");
    println!(
        "{:<20} {:>12} {:>12} {:>12}",
        "Name", "RecoveryRate", "AvgTime", "BestMSE"
    );
    println!("{}", "-".repeat(60));

    // Group results by name
    for name in &names {
        let name_results: Vec<&BenchmarkResult> =
            all_results.iter().filter(|r| &r.name == name).collect();

        if name_results.is_empty() {
            continue;
        }

        let successes = name_results.iter().filter(|r| r.success).count();
        let recovery_rate = successes as f64 / name_results.len() as f64;
        let avg_time: f64 =
            name_results.iter().map(|r| r.time_secs).sum::<f64>() / name_results.len() as f64;
        let best_mse = name_results
            .iter()
            .map(|r| r.mse)
            .fold(f64::INFINITY, f64::min);

        println!(
            "{:<20} {:>11.1}% {:>11.2}s {:>12.2e}",
            name,
            recovery_rate * 100.0,
            avg_time,
            best_mse
        );
    }

    Ok(())
}
