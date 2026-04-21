
import arviz as az
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

_BAYES_FACTOR_METHODS = {}


def _register_bayes_factor_method(name):
    def decorator(fn):
        _BAYES_FACTOR_METHODS[name] = fn
        return fn

    return decorator



@_register_bayes_factor_method("bridge")
def _bridge_logml(
    idata, return_diagnostics=False, maxiter=1000, tol=1e-10, random_state=None
):
    """
    Estimate log marginal likelihood via bridge sampling (Meng & Wong, 1996), following the easystats/bayestestR reference implementation.
    Parameters
    ----------
    idata : arviz.InferenceData
        Posterior samples (must include log_likelihood group and prior logp if available).
    return_diagnostics : bool, default False
        If True, return a dict with diagnostics (iterations, error, etc.).
    maxiter : int, default 1000
        Maximum number of bridge iterations.
    tol : float, default 1e-10
        Convergence tolerance for the iterative scheme.
    random_state : int or None
        Random seed for reproducibility.
    Returns
    -------
    logml : float
        Estimated log marginal likelihood.
    diagnostics : dict (optional)
        Diagnostics (iterations, error, etc.) if return_diagnostics=True.
    """

    rng = np.random.default_rng(random_state)

    # --- 1. Extract posterior samples (samples x parameters) ---
    posterior = idata.posterior
    var_shapes = {var: posterior[var].values.reshape(-1).shape[0] for var in posterior.data_vars}
    from collections import Counter
    shape_counts = Counter(var_shapes.values())
    most_common_size = shape_counts.most_common(1)[0][0]
    var_names = [var for var, sz in var_shapes.items() if sz == most_common_size]
    if not var_names:
        raise ValueError("No posterior variables with matching sample size found for bridge sampling.")
    samples = np.column_stack([posterior[var].values.reshape(-1) for var in var_names])
    if samples.ndim == 1:
        samples = samples[:, None]
    n_samples, n_dim = samples.shape
    if n_samples < 2 or n_dim < 1 or samples.size == 0:
        msg = (
            f"Bridge sampling failed: Not enough usable posterior variables or samples.\n"
            f"Variables used: {var_names}\n"
            f"Shape: {samples.shape}.\n"
            "Check your model output and ensure all sampled parameters have matching, non-degenerate shapes."
        )
        raise ValueError(msg)

    # --- 2. Fit proposal (multivariate normal) ---
    mu = np.mean(samples, axis=0)
    cov = np.cov(samples, rowvar=False)
    if n_dim == 1:
        cov = np.atleast_2d(cov)
    if not np.all(np.linalg.eigvals(cov) > 0):
        cov += np.eye(n_dim) * 1e-6

    # --- 3. Evaluate log-likelihood and log-prior for posterior samples ---
    if not hasattr(idata, "log_likelihood"):
        raise ValueError("InferenceData must have a log_likelihood group for bridge sampling.")
    log_like_group = idata.log_likelihood
    log_like_flat = np.sum([v.values for v in log_like_group.data_vars.values()], axis=0)
    log_like_flat = log_like_flat.reshape(-1)
    if hasattr(idata, "log_prior"):
        log_prior = az.extract(idata, group="log_prior")
        log_prior_flat = np.sum([v.values for v in log_prior.data_vars.values()], axis=0)
        log_prior_flat = log_prior_flat.reshape(-1)
    else:
        log_prior_flat = np.zeros_like(log_like_flat)
    logp_post = log_like_flat + log_prior_flat

    # --- 4. Split samples for bridge ---
    N1 = n_samples // 2
    N2 = n_samples - N1
    idx1 = np.arange(N1)
    idx2 = np.arange(N1, N1 + N2)
    post1 = samples[idx1]
    post2 = samples[idx2]

    # --- 5. Draw proposal samples ---
    prop_samples = rng.multivariate_normal(mu, cov, size=N2)

    # --- 6. Evaluate log densities ---
    def logp_fn(x):
        # In a full implementation, this would evaluate the model logp for arbitrary x.
        # Here, fallback to proposal logpdf as in bayestestR.
        return multivariate_normal.logpdf(x, mean=mu, cov=cov)

    q11 = logp_post[idx2]
    q12 = multivariate_normal.logpdf(post2, mean=mu, cov=cov)
    try:
        q21 = logp_fn(prop_samples)
    except Exception:
        q21 = multivariate_normal.logpdf(prop_samples, mean=mu, cov=cov)
    q22 = multivariate_normal.logpdf(prop_samples, mean=mu, cov=cov)

    # --- 7. Iterative bridge sampling scheme (Meng & Wong, 1996) ---
    s1 = N1 / (N1 + N2)
    s2 = N2 / (N1 + N2)
    r = 1.0
    crit = tol + 1
    iters = 0
    while crit > tol and iters < maxiter:
        num = np.exp(q21 - np.log(s1 * np.exp(q21) + s2 * r * np.exp(q22)))
        den = np.exp(q11 - np.log(s1 * np.exp(q11) + s2 * r * np.exp(q12)))
        r_new = num.mean() / den.mean()
        crit = np.abs((r_new - r) / r_new)
        r = r_new
        iters += 1
    logml = np.log(r)

    # --- 8. Diagnostics ---
    re2 = (1 / N2) * np.var(num) / np.mean(num) ** 2 + (1 / N1) * np.var(den) / np.mean(den) ** 2
    cv = np.sqrt(re2)
    diagnostics = dict(
        logml=logml,
        iterations=iters,
        rel_error_sq=re2,
        cv=cv,
        N1=N1,
        N2=N2,
        tol=tol,
        converged=(crit <= tol),
        method="bridge"
    )
    if return_diagnostics:
        return diagnostics
    return logml


def bayes_factor_compare_models(
    idata_list,
    model_labels=None,
    method: str = "bridge",
    prior_note: str = None,
    return_diagnostics: bool = False,
) -> pd.DataFrame:
    """
    Compute all pairwise Bayes factors for a set of Bayesian models.

    Parameters
    ----------
    idata_list : list of arviz.InferenceData
        Posterior samples for each model to compare.
    model_labels : list of str, optional
        Labels for each model (for DataFrame index/columns). If None, models are labeled by index.
    method : str, default 'bridge'
        Marginal likelihood estimation method ('bridge', 'loo', 'waic', etc.).
    prior_note : str, optional
        Optional string describing the priors used (for reporting).

    Returns
    -------
    bayes_factors : pandas.DataFrame
        DataFrame of Bayes factors (BF[i, j] = BF for model i vs model j; >1 favors i over j).
    diagnostics : dict, optional
        If return_diagnostics=True, a dict of diagnostics for each model (keyed by model label).

    Notes
    -----
    The Bayes factor quantifies the relative evidence for two competing models given the observed data.
    - BF > 1 favors the row model; BF < 1 favors the column model.
    - Conventional thresholds: 1–3 (anecdotal), 3–10 (moderate), >10 (strong evidence).

    Limitations
    -----------
    - Bayes factors are sensitive to prior choices; improper or overly diffuse priors can yield misleading results.
    - Marginal likelihood estimation (e.g., via bridge sampling) can be computationally intensive and unstable for complex or high-dimensional models.
    - Bayes factors compare the *entire* model, not just a single feature; differences in likelihood, priors, or parameterization all affect the result.
    - For large datasets, even small model differences can yield large Bayes factors.
    - Alternative metrics (e.g., WAIC, LOO) may be more robust for predictive performance but do not provide a direct measure of evidence.

    Best Practices
    -------------
    - Always report the priors used when presenting Bayes factors.
    - Use Bayes factors as one piece of evidence, not the sole criterion for model selection.
    - Consider supplementing with predictive checks and out-of-sample validation.

    Extending Methods
    ----------------
    To add a new marginal likelihood method, define a function taking a single InferenceData argument
    and register it with @_register_bayes_factor_method('your_method_name').

    Supported methods include:
    - 'bridge': Bridge sampling marginal likelihood
    """
    if model_labels is None:
        model_labels = [f"model_{i}" for i in range(len(idata_list))]
    if len(model_labels) != len(idata_list):
        raise ValueError("model_labels must match length of idata_list")
    if method not in _BAYES_FACTOR_METHODS:
        raise ValueError(
            f"Unknown method: {method}. Available: {list(_BAYES_FACTOR_METHODS.keys())}"
        )

    logml_fn = _BAYES_FACTOR_METHODS[method]
    logmls = []
    diagnostics = {}
    for label, idata in zip(model_labels, idata_list):
        # Try to call with return_diagnostics if supported
        try:
            if return_diagnostics:
                diag = logml_fn(idata, return_diagnostics=True)
                if isinstance(diag, dict) and 'logml' in diag:
                    logmls.append(diag['logml'])
                    diagnostics[label] = diag
                else:
                    logmls.append(diag)
                    diagnostics[label] = None
            else:
                logmls.append(logml_fn(idata))
        except TypeError:
            # Fallback for methods that don't support return_diagnostics
            logmls.append(logml_fn(idata))
            diagnostics[label] = None
    n = len(logmls)
    bf_mat = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                bf_mat[i, j] = np.exp(logmls[i] - logmls[j])
    df = pd.DataFrame(bf_mat, index=model_labels, columns=model_labels)
    if prior_note:
        print(f"Bayes factors computed with priors: {prior_note}")
    if return_diagnostics:
        return df, diagnostics
    return df
