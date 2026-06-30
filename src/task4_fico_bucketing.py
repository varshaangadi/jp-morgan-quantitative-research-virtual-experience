import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
# ================================================================
# 1. HELPER: cost for MSE-based quantization (1D)
# ================================================================

def _precompute_prefix_sums(values):
    """
    For fast bucket cost lookup:
    prefix_x[k]   = sum_{i<k} x_i
    prefix_x2[k]  = sum_{i<k} x_i^2
    """
    values = np.asarray(values, dtype=float)
    prefix_x = np.concatenate([[0.0], np.cumsum(values)])
    prefix_x2 = np.concatenate([[0.0], np.cumsum(values**2)])
    return prefix_x, prefix_x2


def _bucket_sse(prefix_x, prefix_x2, i, j):
    """
    Sum of squared errors (SSE) for bucket containing indices [i..j] inclusive.
    """
    n = j - i + 1
    if n <= 0:
        return 0.0
    s = prefix_x[j+1] - prefix_x[i]
    s2 = prefix_x2[j+1] - prefix_x2[i]
    mean = s / n
    # SSE = sum(x^2) - 2*mean*sum(x) + n*mean^2 = s2 - s^2 / n
    return s2 - (s * s) / n


def compute_mse_buckets(fico_scores, n_buckets):
    """
    Optimal 1D quantization (minimising SSE/MSE) using dynamic programming.

    Parameters
    ----------
    fico_scores : array-like
        FICO scores of borrowers.
    n_buckets : int
        Number of buckets to create.

    Returns
    -------
    boundaries : list of floats
        Bucket boundaries including min and max, length = n_buckets + 1
        Example: [300, 620, 680, 740, 800, 850]
    """
    fico = np.asarray(fico_scores, dtype=float)
    # Sort scores
    sort_idx = np.argsort(fico)
    fico_sorted = fico[sort_idx]
    n = len(fico_sorted)

    if n_buckets <= 0 or n_buckets > n:
        raise ValueError("n_buckets must be between 1 and number of points")

    prefix_x, prefix_x2 = _precompute_prefix_sums(fico_sorted)

    # dp[b][j] = minimum SSE using b buckets for first j+1 points
    dp = np.full((n_buckets+1, n), np.inf)
    parent = np.full((n_buckets+1, n), -1, dtype=int)

    # Base case: one bucket (b=1)
    for j in range(n):
        dp[1, j] = _bucket_sse(prefix_x, prefix_x2, 0, j)
        parent[1, j] = -1  # start

    # DP for b = 2..n_buckets
    for b in range(2, n_buckets+1):
        for j in range(b-1, n):  # need at least b points for b buckets
            best_cost = np.inf
            best_i = -1
            # last bucket from i..j
            for i in range(b-2, j):
                cost = dp[b-1, i] + _bucket_sse(prefix_x, prefix_x2, i+1, j)
                if cost < best_cost:
                    best_cost = cost
                    best_i = i
            dp[b, j] = best_cost
            parent[b, j] = best_i

    # Reconstruct bucket splits
    boundaries_idx = []
    b = n_buckets
    j = n - 1
    while b > 0:
        i = parent[b, j]
        boundaries_idx.append((i+1, j))  # bucket start..end indices
        j = i
        b -= 1
    boundaries_idx.reverse()

    # Convert index boundaries to FICO boundaries
    boundaries = [fico_sorted[0]]
    for (start, end) in boundaries_idx:
        # boundary is upper edge of this bucket
        boundaries.append(fico_sorted[end])
    # Ensure last boundary is max
    boundaries[-1] = fico_sorted[-1]

    return boundaries


# ================================================================
# 2. HELPER: log-likelihood-based bucketization
# ================================================================

def _precompute_prefix_counts(default_flags):
    """
    prefix_n[k]  = number of points with index < k
    prefix_k[k]  = number of defaults with index < k
    """
    flags = np.asarray(default_flags, dtype=int)
    n = len(flags)
    prefix_n = np.arange(n+1)  # since each row = 1 record
    prefix_k = np.concatenate([[0], np.cumsum(flags)])
    return prefix_n, prefix_k


def _bucket_loglik(prefix_n, prefix_k, i, j, eps=1e-6):
    """
    Log-likelihood contribution of bucket [i..j], using empirical PD.

    We use Laplace-style smoothing to avoid log(0):
      p = (k + eps) / (n + 2*eps)
    """
    n = prefix_n[j+1] - prefix_n[i]
    k = prefix_k[j+1] - prefix_k[i]
    if n == 0:
        return 0.0

    # smoothed PD
    p = (k + eps) / (n + 2*eps)
    return k * np.log(p) + (n - k) * np.log(1.0 - p)


def compute_ll_buckets(fico_scores, default_flags, n_buckets, eps=1e-6):
    """
    Optimal bucketization maximizing log-likelihood via dynamic programming.

    Parameters
    ----------
    fico_scores : array-like
        FICO scores.
    default_flags : array-like of 0/1
        Default indicator for each borrower.
    n_buckets : int
        Number of buckets desired.
    eps : float
        Smoothing to avoid log(0).

    Returns
    -------
    boundaries : list of floats
        Bucket boundaries including min and max.
    """
    fico = np.asarray(fico_scores, dtype=float)
    dflt = np.asarray(default_flags, dtype=int)
    if len(fico) != len(dflt):
        raise ValueError("fico_scores and default_flags must have same length")

    # Sort by FICO
    sort_idx = np.argsort(fico)
    fico_sorted = fico[sort_idx]
    dflt_sorted = dflt[sort_idx]
    n = len(fico_sorted)

    if n_buckets <= 0 or n_buckets > n:
        raise ValueError("n_buckets must be between 1 and number of points")

    prefix_n, prefix_k = _precompute_prefix_counts(dflt_sorted)

    # dp[b][j] = max LL using b buckets for first j+1 points
    dp = np.full((n_buckets+1, n), -np.inf)
    parent = np.full((n_buckets+1, n), -1, dtype=int)

    # Base case: one bucket
    for j in range(n):
        dp[1, j] = _bucket_loglik(prefix_n, prefix_k, 0, j, eps=eps)
        parent[1, j] = -1

    # DP for b=2..n_buckets
    for b in range(2, n_buckets+1):
        for j in range(b-1, n):
            best_ll = -np.inf
            best_i = -1
            for i in range(b-2, j):
                ll = dp[b-1, i] + _bucket_loglik(prefix_n, prefix_k, i+1, j, eps=eps)
                if ll > best_ll:
                    best_ll = ll
                    best_i = i
            dp[b, j] = best_ll
            parent[b, j] = best_i

    # Reconstruct bucket indices
    boundaries_idx = []
    b = n_buckets
    j = n - 1
    while b > 0:
        i = parent[b, j]
        boundaries_idx.append((i+1, j))
        j = i
        b -= 1
    boundaries_idx.reverse()

    # Convert to FICO boundaries
    boundaries = [fico_sorted[0]]
    for (start, end) in boundaries_idx:
        boundaries.append(fico_sorted[end])
    boundaries[-1] = fico_sorted[-1]

    return boundaries


# ================================================================
# 3. RATING MAP: from FICO score to rating
# ================================================================

def make_rating_function(boundaries):
    """
    Given a list of sorted boundaries [b0, b1, ..., bK],
    returns a function that maps FICO score -> rating.

    Requirement: lower rating = better credit score.
    So:
        highest FICO bucket -> rating 1
        ...
        lowest FICO bucket  -> rating K
    """
    boundaries = list(boundaries)
    K = len(boundaries) - 1

    def fico_to_rating(score):
        # Find which bucket score belongs to (ascending FICO)
        for i in range(K):
            low = boundaries[i]
            high = boundaries[i+1]
            # include upper bound in last bucket
            if i < K-1:
                if low <= score < high:
                    # bucket index i -> rating K-i (invert scale)
                    return K - i
            else:
                if low <= score <= high:
                    return K - i
        # If out of range, you can choose to clamp or return None
        return None

    return fico_to_rating


# ================================================================
# 4. EXAMPLE USAGE
# ================================================================
if __name__ == "__main__":
    # Load your loan/mortgage data
    # Make sure the CSV is in the SAME folder as this .py file
    df = pd.read_csv("data/Task 3 and 4_Loan_Data.csv").head(200)

    # Change these column names to match your file
    fico_col = "fico_score"       # e.g. FICO score column
    default_col = "default" # e.g. 0/1 default flag

    fico_scores = df[fico_col].values
    defaults = df[default_col].values

    n_buckets = 5  # for example, create 5 rating grades

    # --- Method 1: MSE-based boundaries ---
    mse_boundaries = compute_mse_buckets(fico_scores, n_buckets)
    print("MSE-based FICO boundaries:", mse_boundaries)

    # --- Method 2: Log-likelihood-based boundaries ---
    ll_boundaries = compute_ll_buckets(fico_scores, defaults, n_buckets)
    print("Log-likelihood-based FICO boundaries:", ll_boundaries)

    # Build rating map from LL boundaries (for example)
    fico_to_rating = make_rating_function(ll_boundaries)

    # Test on a few scores
    for s in [580, 640, 700, 760, 820]:
        print(f"FICO {s} -> Rating {fico_to_rating(s)}")
plt.hist(fico_scores, bins=20)
plt.title("Distribution of FICO Scores")
plt.xlabel("FICO Score")
plt.ylabel("Frequency")
plt.show()