import numpy as np
import pandas as pd
import os

# ===== CONFIGURATION =====
N_DATASETS = 20           # Number of datasets to generate
N_SAMPLES = 500           # Number of samples per dataset
N_VARS_TOTAL = 50         # Total number of variables (dimensions)
N_VARS_USED = 12          # Number of variables used in each expression
SEED = 42                 # Random seed for reproducibility
OUTPUT_DIR = "synthetic_50d_datasets_with_gt"  # Output directory name

# Operator probabilities (unnormalized): + * - / = 4, 3, 2, 1
OPERATOR_PROBS = [4, 3, 2, 1]  # ['+', '*', '-', '/']
PARENTHESES_PROB = 0.15        # Probability of adding parentheses for complexity

MAX_ATTEMPTS = 100             # Maximum attempts to generate valid expression
# ========================

def generate_datasets_with_ground_truth():
    """
    Generates synthetic high-dimensional datasets with ground truth expressions.
    For each dataset, it saves:
    1. A .csv file with the data (input variables + 1 target variable).
    2. A .txt file with the corresponding ground truth mathematical expression.
    """
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Setup random number generator
    rng = np.random.default_rng(SEED)
    
    # Setup operators and probabilities
    operators = ['+', '*', '-', '/']
    op_probs = np.array(OPERATOR_PROBS) / np.sum(OPERATOR_PROBS)  # Normalize
    
    def generate_random_expression(var_names, rng):
        """Generate a random mathematical expression using each variable exactly once"""
        
        # Shuffle variables to randomize order
        vars_copy = var_names.copy()
        rng.shuffle(vars_copy)
        
        # We need exactly (N_VARS_USED - 1) operators to connect N_VARS_USED variables
        n_operators_needed = N_VARS_USED - 1
        
        # Ensure we have at least one of each required operator
        required_ops = ['+', '*', '-', '/']
        
        # Generate remaining operators based on probability distribution
        n_additional_ops = n_operators_needed - len(required_ops)
        if n_additional_ops > 0:
            additional_ops = rng.choice(operators, size=n_additional_ops, p=op_probs)
            all_ops = required_ops + list(additional_ops)
        else:
            all_ops = required_ops[:n_operators_needed]
        
        # Shuffle all operators
        rng.shuffle(all_ops)
        
        # Build expression: var1 op1 var2 op2 var3 ... opN varN+1
        expression_parts = [vars_copy[0]]
        
        for i in range(n_operators_needed):
            op = all_ops[i]
            next_var = vars_copy[i + 1]
            
            # Occasionally add parentheses for more complex structures
            if rng.random() < PARENTHESES_PROB and i < n_operators_needed - 1:
                # Group the next operation in parentheses
                next_op = all_ops[i + 1]
                following_var = vars_copy[i + 2]
                
                expression_parts.append(f" {op} ({next_var} {next_op} {following_var})")
                i += 1  # Skip next iteration since we used two operations
            else:
                expression_parts.append(f" {op} {next_var}")
        
        return ''.join(expression_parts)
    
    def verify_expression(expression, var_names):
        """Verify that the expression is valid"""
        # Check all operators are present
        has_all_ops = all(op in expression for op in ['+', '-', '*', '/'])
        if not has_all_ops:
            return False, "Not all operators present"
        
        # Check each variable appears exactly once
        for var in var_names:
            if expression.count(var) != 1:
                return False, f"Variable {var} appears {expression.count(var)} times"
        
        return True, "Valid"

    # Generate and save each dataset
    for dataset_idx in range(N_DATASETS):
        print(f"\n=== Generating Dataset {dataset_idx + 1}/{N_DATASETS} ===")
        
        # Keep trying until we get a valid expression
        for attempt in range(MAX_ATTEMPTS):
            # Select random variables for this dataset
            high_dim_indices = rng.choice(N_VARS_TOTAL, size=N_VARS_USED, replace=False)
            var_names = [f"x_{idx + 1}" for idx in sorted(high_dim_indices)]
            
            # Generate random expression
            expression = generate_random_expression(var_names, rng)
            
            # Verify expression validity
            is_valid, error_msg = verify_expression(expression, var_names)
            if not is_valid:
                if attempt < 5:  # Only print first few attempts
                    print(f"Attempt {attempt + 1}: {error_msg}, regenerating...")
                continue
            
            # Generate test data
            X = rng.uniform(-1, 1, size=(N_SAMPLES, N_VARS_TOTAL))
            eval_vars = {f"x_{idx + 1}": X[:, idx] for idx in range(N_VARS_TOTAL)}
            
            # Evaluate expression
            try:
                y = eval(expression, {"__builtins__": None}, eval_vars)
                
                # Check for nan or inf values
                if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                    if attempt < 5:
                        print(f"Attempt {attempt + 1}: Expression produced nan/inf, regenerating...")
                    continue
                
                # Success - expression is valid
                print(f"Selected variables: {var_names}")
                print(f"Generated expression: {expression}")
                
                # Show statistics
                var_usage = {var: expression.count(var) for var in var_names}
                op_counts = {op: expression.count(op) for op in operators}
                print(f"Variable usage: {var_usage}")
                print(f"Operator counts: {op_counts}")
                
                break
                
            except Exception as e:
                if attempt < 5:
                    print(f"Attempt {attempt + 1}: Evaluation failed ({e}), regenerating...")
                continue
        
        else:
            raise RuntimeError(f"Could not generate valid expression for dataset {dataset_idx + 1} after {MAX_ATTEMPTS} attempts")
        
        # Save dataset
        dataset = np.concatenate((X, y[:, np.newaxis]), axis=1)
        df = pd.DataFrame(dataset)
        
        base_filename = f"synthetic_50d_{dataset_idx + 1}"
        
        # Save CSV file
        csv_path = os.path.join(OUTPUT_DIR, f"{base_filename}.csv")
        df.to_csv(csv_path, header=False, index=False)
        print(f"Saved data: {csv_path}")
        
        # Save ground truth file
        txt_path = os.path.join(OUTPUT_DIR, f"{base_filename}.txt")
        with open(txt_path, 'w') as f:
            f.write(expression)
        print(f"Saved ground truth: {txt_path}")
        print("-" * 50)

if __name__ == "__main__":
    generate_datasets_with_ground_truth()
