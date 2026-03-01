//! Example using MCTS token generator with constants
//!
//! Run with: cargo run --example mcts_with_constants
//!
//! Make sure PSE_PATH is set to the PSE directory:
//!   export PSE_PATH=/path/to/PSE

use ndarray::Array2;
use pse::{expression_complexity, FitConfig, PSRNConfig, PSRNRegressor, TokenGenerator};
use std::path::PathBuf;

fn main() -> pse::Result<()> {
    // Get PSE path from environment
    let pse_path = std::env::var("PSE_PATH").expect("PSE_PATH environment variable must be set");
    let pse_dir = PathBuf::from(&pse_path);

    // Generate data: y = 2.5 * sin(x) + 1.3
    let n_samples = 200;
    let mut x_data = Vec::with_capacity(n_samples);
    let mut y_data = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let x = (i as f64) * 0.1 - 10.0;
        x_data.push(x);
        y_data.push(2.5 * x.sin() + 1.3);
    }

    let x = Array2::from_shape_vec((n_samples, 1), x_data).unwrap();

    // Configure with MCTS and constants enabled
    let config = PSRNConfig {
        variables: vec!["x".to_string()],
        operators: Some(vec![
            "Add".to_string(),
            "Mul".to_string(),
            "Sin".to_string(),
            "Cos".to_string(),
        ]),
        n_symbol_layers: 3,
        use_const: true, // Enable constant optimization
        use_dr_mask: false,
        dr_mask_dir: pse_dir.join("dr_mask"),
        stage_config: pse_dir.join("model/stages_config/benchmark.yaml"),
        token_generator_config: pse_dir.join("token_generator_config.yaml"),
        token_generator: TokenGenerator::MCTS,
        device: "cpu".to_string(),
        ..Default::default()
    };

    let mut regressor = PSRNRegressor::new(config)?;

    let fit_config = FitConfig {
        n_down_sample: 50,
        eta: 0.99,
        threshold: 1e-8,
        add_bias: true, // Allow bias terms
        prun_const: true,
        prun_ndigit: 4, // Round constants to 4 digits
        ..Default::default()
    };

    println!("Fitting with MCTS and constant optimization...");
    let result = regressor.fit(x.view(), &y_data, fit_config)?;

    println!("\nResults:");
    println!("Converged: {}", result.converged);

    // Analyze discovered expressions
    for expr in result.pareto_frontier.iter().take(10) {
        let complexity = expression_complexity(&expr.expression).unwrap_or(f64::INFINITY);
        println!(
            "  {} | MSE: {:.2e} | Complexity: {:.0} | Reward: {:.4}",
            expr.expression, expr.mse, complexity, expr.reward
        );
    }

    // Get regressor params
    let params = regressor.get_params()?;
    println!("\nRegressor parameters:");
    println!("  Variables: {:?}", params.variables);
    println!("  Operators: {:?}", params.operators);
    println!("  Const range: {:?}", params.trying_const_range);

    Ok(())
}
