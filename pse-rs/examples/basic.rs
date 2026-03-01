//! Basic example of using PSE from Rust
//!
//! Run with: cargo run --example basic
//!
//! Make sure PSE_PATH is set to the PSE directory:
//!   export PSE_PATH=/path/to/PSE

use ndarray::Array2;
use pse::{FitConfig, PSRNConfig, PSRNRegressor, SortBy, TokenGenerator};
use std::path::PathBuf;

fn main() -> pse::Result<()> {
    // Get PSE path from environment
    let pse_path = std::env::var("PSE_PATH").expect("PSE_PATH environment variable must be set");
    let pse_dir = PathBuf::from(&pse_path);

    // Generate synthetic data: y = x0 * x1 + x0^2
    let n_samples = 100;
    let mut x_data = Vec::with_capacity(n_samples * 2);
    let mut y_data = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let x0 = (i as f64) * 0.1 - 5.0;
        let x1 = (i as f64) * 0.05 - 2.5;
        x_data.push(x0);
        x_data.push(x1);
        y_data.push(x0 * x1 + x0 * x0);
    }

    let x = Array2::from_shape_vec((n_samples, 2), x_data).unwrap();

    // Configure the regressor with absolute paths
    let config = PSRNConfig {
        variables: vec!["x0".to_string(), "x1".to_string()],
        operators: Some(vec![
            "Add".to_string(),
            "Mul".to_string(),
            "Sub".to_string(),
        ]),
        n_symbol_layers: 3,
        use_const: false,
        // Disable dr_mask since we don't have a pre-generated mask for these operators
        use_dr_mask: false,
        dr_mask_dir: pse_dir.join("dr_mask"),
        stage_config: pse_dir.join("model/stages_config/benchmark.yaml"),
        token_generator_config: pse_dir.join("token_generator_config.yaml"),
        token_generator: TokenGenerator::GP,
        device: "cpu".to_string(),
        ..Default::default()
    };

    // Create regressor
    let mut regressor = PSRNRegressor::new(config)?;

    // Configure fitting
    let fit_config = FitConfig {
        n_down_sample: 20,
        threshold: 1e-10,
        real_time_display: true,
        ..Default::default()
    };

    // Fit the model
    println!("Fitting regressor...");
    let result = regressor.fit(x.view(), &y_data, fit_config)?;

    println!("\nConverged: {}", result.converged);
    println!("\nPareto frontier ({} expressions):", result.pareto_frontier.len());

    // Show top expressions
    for (i, expr) in result.pareto_frontier.iter().take(5).enumerate() {
        println!(
            "  {}. {} (MSE: {:.2e}, complexity: {:.0})",
            i + 1,
            expr.expression,
            expr.mse,
            expr.complexity
        );
    }

    // Get best expression
    if let Some(best) = result.best_expression() {
        println!("\nBest expression by MSE: {}", best);
    }

    // Get Pareto frontier sorted by complexity
    let by_complexity = regressor.get_pareto_frontier(SortBy::Complexity)?;
    if let Some(simplest) = by_complexity.first() {
        println!("Simplest expression: {}", simplest.expression);
    }

    // Make predictions
    let x_test = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let predictions = regressor.predict(x_test.view())?;
    println!("\nPredictions: {:?}", predictions);

    Ok(())
}
