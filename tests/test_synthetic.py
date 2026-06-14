import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tfi.synthetic import generate_ar1_series, compute_empirical_rho

def analytic_auc(rho: float) -> float:
    """Computes the expected AUC based on the analytic proof."""
    return 0.5 + (2.0 / np.pi) * np.arcsin(rho / np.sqrt(2.0))

def bootstrap_auc_se(y_true: np.ndarray, y_scores: np.ndarray, n_bootstraps: int = 500, seed: int = 42) -> float:
    """Computes the standard error of the empirical AUC via fast bootstrapping."""
    rng = np.random.RandomState(seed)
    bootstrapped_scores = []
    
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_scores), len(y_scores))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_scores[indices])
        bootstrapped_scores.append(score)
        
    return np.std(bootstrapped_scores)

def test_theorem_validation():
    """
    Validates the mathematical theorem across a sweep of phi values.
    Fails if the empirical AUC diverges from the analytic prediction by more than 3 Standard Errors.
    """
    phis = [0.5, 0.7, 0.9, 0.95, 0.99]
    
    print("\n--- Theorem Validation Sweep ---")
    for phi in phis:
        # Increase length drastically for high phi to ensure a stable transition subset
        length = 10000 if phi < 0.95 else 50000
        
        # Balanced threshold. Mean of AR(1) with mu=0 is 0. Setting tau=0 ensures breach rate ~ 0.5
        df = generate_ar1_series(phi=phi, mu=0.0, sigma=1.0, tau=0.0, length=length, seed=42)
        
        # Target: Y_t
        y_true = df['label'].values[1:]
        # Persistence Score: Lagged level X_{t-1}
        y_score = df['level'].values[:-1]
        
        # 1. Empirical AUC
        emp_auc = roc_auc_score(y_true, y_score)
        
        # 2. Analytic AUC
        rho = compute_empirical_rho(df['level'])
        ana_auc = analytic_auc(rho)
        
        # 3. Bootstrap Standard Error Tolerance
        se = bootstrap_auc_se(y_true, y_score, n_bootstraps=200, seed=42)
        tolerance = 3.0 * se
        
        diff = abs(emp_auc - ana_auc)
        
        status = "PASS" if diff <= tolerance else "FAIL"
        print(f"Phi: {phi:<4} | Rho: {rho:.4f} | Emp: {emp_auc:.4f} | Ana: {ana_auc:.4f} | Diff: {diff:.4f} | SE: {se:.4f} | Tol: {tolerance:.4f} | [{status}]")
        
        assert diff <= tolerance, f"Theorem violation at phi={phi}. Diff {diff} > Tolerance {tolerance}"
