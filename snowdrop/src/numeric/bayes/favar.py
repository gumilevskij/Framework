# Python Gibbs sampler code for a Bayesian FAVAR model.

# Latent factors Ft are sampled using Kalman filter and smoother.
# Factor loadings Λ are sampled with lower-triangular structure and positive diagonal constraints.
# Idiosyncratic variances Ψ and VAR parameters A, Σu are sampled from their respective conditional posteriors.
# The Gibbs sampler iterates these steps to approximate the joint posterior distribution.
# ​
import numpy as np
from numpy.linalg import inv
from scipy.stats import invgamma, multivariate_normal, truncnorm

# -----------------------------
# Simulated Data and Parameters
# -----------------------------
T = 100  # Number of time points
N = 10   # Number of observed variables
r = 3    # Number of latent factors (adjusted for identification)
p = 1    # VAR lag order

np.random.seed(42)
X = np.random.randn(N, T)

Lambda = np.random.randn(N, r)
Psi = np.eye(N) * 0.5
A = np.random.randn(r, r)
Sigma_u = np.eye(r) * 0.5
F = np.random.randn(r, T)

Lambda_prior_mean = np.zeros((N, r))
Lambda_prior_var = np.eye(r)
A_prior_mean = np.zeros((r, r))
A_prior_var = np.eye(r)
alpha_psi, beta_psi = 2, 2

# -----------------------------------
# Kalman Filter and Smoother Functions
# -----------------------------------

def kalman_filter(Y, Z, H, T_mat, Q, x0, P0):
    n_timesteps = Y.shape[1]
    n_state = x0.shape[0]

    x_pred = np.zeros((n_state, n_timesteps))
    P_pred = np.zeros((n_state, n_state, n_timesteps))
    x_filt = np.zeros((n_state, n_timesteps))
    P_filt = np.zeros((n_state, n_state, n_timesteps))

    x_pred[:, 0] = x0
    P_pred[:, :, 0] = P0

    for t in range(n_timesteps):
        if t > 0:
            x_pred[:, t] = T_mat @ x_filt[:, t-1]
            P_pred[:, :, t] = T_mat @ P_filt[:, :, t-1] @ T_mat.T + Q

        y_t = Y[:, t]
        S = Z @ P_pred[:, :, t] @ Z.T + H
        K = P_pred[:, :, t] @ Z.T @ inv(S)
        x_filt[:, t] = x_pred[:, t] + K @ (y_t - Z @ x_pred[:, t])
        P_filt[:, :, t] = P_pred[:, :, t] - K @ Z @ P_pred[:, :, t]

    return x_pred, P_pred, x_filt, P_filt

def kalman_smoother(x_pred, P_pred, x_filt, P_filt, T_mat):
    n_state, n_timesteps = x_filt.shape
    x_smooth = np.zeros_like(x_filt)
    P_smooth = np.zeros_like(P_filt)

    x_smooth[:, -1] = x_filt[:, -1]
    P_smooth[:, :, -1] = P_filt[:, :, -1]

    for t in reversed(range(n_timesteps - 1)):
        J = P_filt[:, :, t] @ T_mat.T @ inv(P_pred[:, :, t+1])
        x_smooth[:, t] = x_filt[:, t] + J @ (x_smooth[:, t+1] - x_pred[:, t+1])
        P_smooth[:, :, t] = P_filt[:, :, t] + J @ (P_smooth[:, :, t+1] - P_pred[:, :, t+1]) @ J.T

    return x_smooth, P_smooth

def sample_factors_kalman(X, Lambda, Psi, A, Sigma_u, F, p):
    T = X.shape[1]
    r = Lambda.shape[1]

    Z = Lambda
    H = Psi
    T_mat = A
    Q = Sigma_u

    x0 = np.zeros(r)
    P0 = np.eye(r) * 1e2

    x_pred, P_pred, x_filt, P_filt = kalman_filter(X, Z, H, T_mat, Q, x0, P0)
    x_smooth, P_smooth = kalman_smoother(x_pred, P_pred, x_filt, P_filt, T_mat)

    F_sampled = np.zeros_like(F)
    for t in range(T):
        F_sampled[:, t] = multivariate_normal.rvs(mean=x_smooth[:, t], cov=P_smooth[:, :, t])

    return F_sampled

# -----------------------------------
# Identification-Constrained Lambda Sampling
# -----------------------------------

def sample_lambda_identified(X, F, Psi):
    N, r = Lambda.shape
    Lambda_new = np.zeros_like(Lambda)
    Psi_inv = np.diag(1 / np.diag(Psi))

    for i in range(N):
        free_indices = [j for j in range(min(i+1, r))]  # lower-triangular including diagonal

        if len(free_indices) == 0:
            continue

        F_sub = F[free_indices, :]

        V_lambda = inv(np.eye(len(free_indices)) + (F_sub @ F_sub.T) * Psi_inv[i, i])
        M_lambda = V_lambda @ (F_sub @ (X[i, :].T) * Psi_inv[i, i])

        if i < r:
            diag_idx = free_indices.index(i)
            a, b = 0, np.inf
            mean_diag = M_lambda[diag_idx]
            std_diag = np.sqrt(V_lambda[diag_idx, diag_idx])
            a_std, b_std = (a - mean_diag) / std_diag, (b - mean_diag) / std_diag
            diag_sample = truncnorm.rvs(a_std, b_std, loc=mean_diag, scale=std_diag)

            off_diag_indices = [idx for idx in range(len(free_indices)) if idx != diag_idx]
            off_diag_samples = np.zeros(len(off_diag_indices))
            if len(off_diag_indices) > 0:
                for k, od_idx in enumerate(off_diag_indices):
                    mean_od = M_lambda[od_idx]
                    std_od = np.sqrt(V_lambda[od_idx, od_idx])
                    off_diag_samples[k] = np.random.normal(mean_od, std_od)

            free_params = np.zeros(len(free_indices))
            free_params[diag_idx] = diag_sample
            for k, od_idx in enumerate(off_diag_indices):
                free_params[od_idx] = off_diag_samples[k]
        else:
            free_params = multivariate_normal.rvs(mean=M_lambda, cov=V_lambda)

        for idx, j in enumerate(free_indices):
            Lambda_new[i, j] = free_params[idx]

        for j in range(r):
            if j > i:
                Lambda_new[i, j] = 0.0

    return Lambda_new

# -----------------------------------
# Other Gibbs Sampling Functions
# -----------------------------------

def sample_psi(X, F, Lambda):
    N, T = X.shape
    Psi_new = np.zeros((N, N))
    for i in range(N):
        resid = X[i, :] - Lambda[i, :] @ F
        alpha_post = alpha_psi + T / 2
        beta_post = beta_psi + 0.5 * np.sum(resid**2)
        Psi_new[i, i] = invgamma.rvs(a=alpha_post, scale=beta_post)
    return Psi_new

def sample_A(F, A_prior_mean, V_A_prior, Sigma_u):
    """
    Sample VAR(1) coefficient matrix A for F_t = A F_{t-1} + u_t
    
    Parameters:
    - F: (r, T) matrix of latent factors
    - A_prior_mean: (r, r) prior mean matrix for A
    - V_A_prior: (r^2, r^2) prior covariance matrix for vec(A)
    - Sigma_u: (r, r) covariance matrix of VAR residuals
    
    Returns:
    - A_new: sampled (r, r) VAR coefficient matrix
    """
    r, T = F.shape
    X = F[:, :-1].T
    Y = F[:, 1:].T
    
    Sigma_u_inv = inv(Sigma_u)
    temp = Sigma_u_inv @ X.T  # (r, T-1)
    V_A = inv(inv(V_A_prior) + X.T @ temp.T)  # (r, r)
    temp = Sigma_u_inv @ Y.T  # (r, r) @ (r, T-1) = (r, T-1)
    M_A = V_A @ (inv(V_A_prior) @ A_prior_mean.T + X @ temp)
    A_vec = multivariate_normal.rvs(mean=M_A.flatten(), cov=V_A)
    
    # Reshape to (r, r) and transpose to match original shape
    A_new = A_vec.reshape(r, r).T
    
    return A_new

def _sample_A(F, A_prior_mean, V_A_prior, Sigma_u):
    """
    Sample VAR(1) coefficient matrix A for F_t = A F_{t-1} + u_t
    
    Parameters:
    - F: (r, T) matrix of latent factors
    - A_prior_mean: (r, r) prior mean matrix for A
    - V_A_prior: (r^2, r^2) prior covariance matrix for vec(A)
    - Sigma_u: (r, r) covariance matrix of VAR residuals
    
    Returns:
    - A_new: sampled (r, r) VAR coefficient matrix
    """
    r, T = F.shape
    X = F[:, :-1]  # (r, T-1)
    Y = F[:, 1:]   # (r, T-1)
    
    Sigma_u_inv = inv(Sigma_u)
    XX_t = X @ X.T  # (r, r)
    
    # Posterior precision matrix for vec(A)
    precision_likelihood = np.kron(XX_t, Sigma_u_inv)  # (r^2, r^2)
    V_A_prior_inv = inv(V_A_prior)
    V_A_post_inv = V_A_prior_inv + precision_likelihood
    V_A_post = inv(V_A_post_inv)
    
    # Posterior mean vector for vec(A)
    term2 = (Sigma_u_inv @ Y @ X.T).T.flatten()  # vec of (r x r)
    prior_term = V_A_prior_inv @ A_prior_mean.T.flatten()
    M_A_post = V_A_post @ (prior_term + term2)
    
    # Sample vec(A) from multivariate normal
    A_vec = np.random.multivariate_normal(M_A_post, V_A_post)
    
    # Reshape to (r, r) and transpose to match original shape
    A_new = A_vec.reshape(r, r).T
    return A_new

def sample_Sigma_u(F, A):
    r, T = F.shape
    resid = F[:, 1:] - A @ F[:, :-1]
    scale = resid @ resid.T
    Sigma_new = np.diag([invgamma.rvs(a=(T/2), scale=(scale[i,i]/2)) for i in range(r)])
    return Sigma_new

# -----------------------------
# Gibbs Sampling Loop
# -----------------------------
n_iter = 1000
for it in range(n_iter):
    F = sample_factors_kalman(X, Lambda, Psi, A, Sigma_u, F, p)
    Lambda = sample_lambda_identified(X, F, Psi)
    Psi = sample_psi(X, F, Lambda)
    A = sample_A(F, A_prior_mean, A_prior_var, Sigma_u)
    Sigma_u = sample_Sigma_u(F, A)

    if it % 100 == 0:
        print(f"Iteration {it}: Lambda[0,0]={Lambda[0,0]:.3f}, Psi[0,0]={Psi[0,0]:.3f}")
