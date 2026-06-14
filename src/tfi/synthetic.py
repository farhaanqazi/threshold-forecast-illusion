import numpy as np
import pandas as pd

def generate_ar1_series(
    phi: float, 
    mu: float = 0.0, 
    sigma: float = 1.0, 
    tau: float = 0.0, 
    length: int = 10000, 
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates a synthetic AR(1) time series: X_t = phi * X_{t-1} + mu + eps_t.
    
    Args:
        phi: AR(1) coefficient. For stationary series, |phi| < 1.
        mu: Drift term.
        sigma: Standard deviation of the innovation term (eps_t).
        tau: The threshold for the binary label (Y_t = 1 if X_t >= tau).
        length: Number of timesteps.
        seed: Random seed for reproducibility.
        
    Returns:
        A pandas DataFrame containing ['time', 'level', 'label'].
    """
    np.random.seed(seed)
    eps = np.random.normal(0, sigma, length)
    X = np.zeros(length)
    
    # Initialize from the stationary mean if strictly stationary to prevent burn-in artifacts
    if abs(phi) < 1.0:
        X[0] = (mu / (1.0 - phi)) + eps[0]
    else:
        X[0] = eps[0]
        
    for t in range(1, length):
        X[t] = phi * X[t-1] + mu + eps[t]
        
    # Generate threshold labels
    Y = (X >= tau).astype(int)
    
    return pd.DataFrame({
        'time': np.arange(length),
        'level': X,
        'label': Y
    })


def compute_empirical_rho(level_series: pd.Series) -> float:
    """
    Computes the empirical lag-1 autocorrelation (rho) of the continuous level series.
    """
    return level_series.autocorr(lag=1)
