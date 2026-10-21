"""Multivariate Hawkes processes with sum-of-exponential kernels.

Model
-----
For D event types and K exponential decay components the conditional intensity is

    lambda_i(t) = mu_i + sum_j sum_k alpha[i, j, k] * sum_{t_l^j < t} beta_k * exp(-beta_k (t - t_l^j)).

Each kernel component integrates to ``alpha[i, j, k]``, so ``G = alpha.sum(-1)`` is the
*branching matrix*: ``G[i, j]`` is the expected number of type-i events directly triggered
by one type-j event. The process is stationary iff the spectral radius rho(G) < 1, in
which case the mean intensity is ``(I - G)^{-1} mu``.

What is implemented
-------------------
* ``simulate``         exact simulation through the Poisson-cluster (branching) representation,
                       vectorised one generation at a time.
* ``simulate_ogata``   Ogata's (1981) thinning algorithm; slower, used to cross-validate.
* ``excitation``       exact O(N) evaluation of the kernel sums at every event, vectorised in
                       numerically safe chunks (no Python loop over events).
* ``fit``              maximum likelihood. For fixed decays the log-likelihood is *concave*
                       in (mu, alpha) and separates across target types, so each block is a
                       small convex problem (EM warm start + L-BFGS-B with analytic gradient).
                       Decays are profiled out (1-D bounded search, or a grid for K = 2).
* ``rescaled_interarrivals``  time-rescaling theorem residuals for goodness-of-fit tests.
* ``HawkesForecaster`` closed-form E[N(t, t + h] | F_t] via the linear ODE satisfied by the
                       expected excitation state (matrix exponential).
* ``HawkesState``      O(D K) online intensity tracker used inside trading strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import linalg, optimize, stats

Array = np.ndarray
Sequence3 = tuple[Array, Array, float]  # (times, types, horizon T)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HawkesParams:
    """Parameters of a D-variate Hawkes process with K exponential components.

    Attributes
    ----------
    mu : (D,) baseline intensities.
    alpha : (D, D, K) branching ratios; ``alpha[i, j, k]`` is the excitation of type i by type j
        through decay component k. A (D, D) array is accepted for K = 1.
    beta : (K,) decay rates (1 / time unit).
    """

    mu: Array
    alpha: Array
    beta: Array

    def __post_init__(self) -> None:
        mu = np.asarray(self.mu, dtype=float).reshape(-1)
        beta = np.atleast_1d(np.asarray(self.beta, dtype=float)).reshape(-1)
        alpha = np.asarray(self.alpha, dtype=float)
        if alpha.ndim == 2:
            alpha = alpha[:, :, None]
        d, k = mu.size, beta.size
        if alpha.shape != (d, d, k):
            raise ValueError(f"alpha must have shape {(d, d, k)}, got {alpha.shape}")
        if np.any(mu < 0) or np.any(alpha < 0) or np.any(beta <= 0):
            raise ValueError("require mu >= 0, alpha >= 0 and beta > 0")
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)

    @classmethod
    def exponential(cls, mu: Sequence[float], alpha: Array, beta: float) -> "HawkesParams":
        """Single-exponential kernel convenience constructor."""
        return cls(np.asarray(mu, float), np.asarray(alpha, float)[:, :, None], np.array([beta]))

    @property
    def dim(self) -> int:
        return int(self.mu.size)

    @property
    def n_components(self) -> int:
        return int(self.beta.size)

    @property
    def n_params(self) -> int:
        return self.dim + self.alpha.size + self.n_components

    @property
    def branching_matrix(self) -> Array:
        return self.alpha.sum(axis=2)

    @property
    def spectral_radius(self) -> float:
        return float(np.max(np.abs(np.linalg.eigvals(self.branching_matrix))))

    def stationary_intensity(self) -> Array:
        """Mean intensity (I - G)^{-1} mu of the stationary process."""
        if self.spectral_radius >= 1.0:
            raise ValueError("process is not stationary (spectral radius >= 1)")
        return np.linalg.solve(np.eye(self.dim) - self.branching_matrix, self.mu)

    def to_dict(self) -> dict:
        return {"mu": self.mu.tolist(), "alpha": self.alpha.tolist(), "beta": self.beta.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "HawkesParams":
        return cls(np.array(d["mu"]), np.array(d["alpha"]), np.array(d["beta"]))


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def simulate(
    params: HawkesParams,
    T: float,
    rng: np.random.Generator | int | None = None,
    burn_in: float | None = None,
) -> tuple[Array, Array]:
    """Exact simulation on [0, T) through the Poisson-cluster representation.

    Immigrants of type i arrive as a homogeneous Poisson process with rate mu_i. Every event of
    type j independently produces Poisson(G[i, j]) children of type i, each delayed by a draw
    from the normalised kernel (a mixture of exponentials with weights alpha[i, j, :] / G[i, j]).
    Whole generations are drawn with vectorised NumPy calls, so the cost is O(N) with a small
    constant and no per-event Python loop.

    The process is started at ``-burn_in`` and the prefix is discarded, so the returned path is
    (approximately) a draw from the stationary process. Use ``burn_in=0`` for an empty history.
    """
    rng = np.random.default_rng(rng)
    d, k = params.dim, params.n_components
    g = params.branching_matrix
    rho = params.spectral_radius
    if rho >= 1.0:
        raise ValueError("cannot simulate a non-stationary Hawkes process on a burn-in window")
    if burn_in is None:
        burn_in = 30.0 / (params.beta.min() * (1.0 - rho))
    t0 = -float(burn_in)

    n_imm = rng.poisson(params.mu * (T - t0))
    gen_t = rng.uniform(t0, T, size=int(n_imm.sum()))
    gen_d = np.repeat(np.arange(d), n_imm)
    out_t, out_d = [gen_t], [gen_d]

    with np.errstate(invalid="ignore", divide="ignore"):
        comp_p = np.where(g[:, :, None] > 0, params.alpha / g[:, :, None], 0.0)
    comp_cdf = np.cumsum(comp_p, axis=2)

    while gen_t.size:
        n = gen_t.size
        counts = rng.poisson(g[:, gen_d].T)  # (n, D) children of each type for each parent
        flat = counts.ravel()
        total = int(flat.sum())
        if total == 0:
            break
        parent = np.repeat(np.repeat(np.arange(n), d), flat)
        child_d = np.repeat(np.tile(np.arange(d), n), flat)
        if k == 1:
            delay = rng.exponential(1.0 / params.beta[0], size=total)
        else:
            cdf = comp_cdf[child_d, gen_d[parent]]
            comp = np.minimum((rng.random(total)[:, None] > cdf).sum(axis=1), k - 1)
            delay = rng.exponential(1.0, size=total) / params.beta[comp]
        child_t = gen_t[parent] + delay
        keep = child_t < T
        gen_t, gen_d = child_t[keep], child_d[keep]
        out_t.append(gen_t)
        out_d.append(gen_d)

    t = np.concatenate(out_t)
    typ = np.concatenate(out_d)
    keep = t >= 0.0
    t, typ = t[keep], typ[keep]
    order = np.argsort(t, kind="stable")
    return t[order], typ[order].astype(np.int64)


def simulate_ogata(
    params: HawkesParams, T: float, rng: np.random.Generator | int | None = None
) -> tuple[Array, Array]:
    """Ogata (1981) thinning, started from an empty history at t = 0.

    Between events the intensity of an exponential-kernel Hawkes process only decays, so the
    current total intensity is a valid dominating rate until the next candidate point.
    """
    rng = np.random.default_rng(rng)
    mu, beta = params.mu, params.beta
    jump = params.alpha * beta[None, None, :]
    y = np.zeros((params.dim, params.n_components))
    t, mu_sum = 0.0, float(mu.sum())
    ts: list[float] = []
    ds: list[int] = []
    while True:
        lam_bar = mu_sum + float(y.sum())
        t_cand = t + rng.exponential(1.0 / lam_bar)
        if t_cand >= T:
            break
        y *= np.exp(-beta * (t_cand - t))
        t = t_cand
        cum = np.cumsum(mu + y.sum(axis=1))
        u = rng.random() * lam_bar
        if u < cum[-1]:
            j = int(np.searchsorted(cum, u, side="right"))
            ts.append(t)
            ds.append(j)
            y += jump[:, j, :]
    return np.asarray(ts, dtype=float), np.asarray(ds, dtype=np.int64)


# ---------------------------------------------------------------------------
# Exact kernel sums
# ---------------------------------------------------------------------------
def _excitation_single(times: Array, onehot: Array, b: float, chunk: float) -> Array:
    """R[n, j] = sum_{m < n, d_m = j} b * exp(-b (t_n - t_m)) for one decay rate b.

    Events are grouped into windows of length ``chunk / b``. Inside a window the sum is a
    cumulative sum of b * exp(b * tau) terms (tau measured from the window's first event, so
    every exponent is below ``chunk`` and nothing overflows); a decayed carry links windows.
    """
    n, dim = onehot.shape
    out = np.empty((n, dim))
    win = np.floor((times - times[0]) * (b / chunk)).astype(np.int64)
    starts = np.flatnonzero(np.r_[True, win[1:] != win[:-1]])
    ends = np.r_[starts[1:], n]
    carry = np.zeros(dim)
    t_ref = times[0]
    zero_row = np.zeros((1, dim))
    for s, e in zip(starts.tolist(), ends.tolist()):
        t0 = times[s]
        carry *= np.exp(-b * (t0 - t_ref))
        t_ref = t0
        grow = np.exp(b * (times[s:e] - t0))
        w = onehot[s:e] * (b * grow)[:, None]
        cs = np.cumsum(w, axis=0)
        exclusive = np.vstack([zero_row, cs[:-1]])
        out[s:e] = (carry + exclusive) / grow[:, None]
        carry = carry + cs[-1]
    return out


def excitation(times: Array, types: Array, dim: int, beta: Array, chunk: float = 200.0) -> Array:
    """Kernel sums at every event: R[n, j, k] = sum_{m<n, d_m=j} beta_k exp(-beta_k (t_n - t_m))."""
    times = np.asarray(times, dtype=float)
    types = np.asarray(types, dtype=np.int64)
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    n = times.size
    out = np.zeros((n, dim, beta.size))
    if n == 0:
        return out
    onehot = np.zeros((n, dim))
    onehot[np.arange(n), types] = 1.0
    for k, b in enumerate(beta):
        out[:, :, k] = _excitation_single(times, onehot, float(b), chunk)
    return out


def _as_sequences(data: Sequence3 | Iterable[Sequence3]) -> list[Sequence3]:
    if isinstance(data, tuple) and len(data) == 3 and isinstance(data[2], (int, float, np.number)):
        return [data]  # a single (times, types, T)
    return list(data)  # type: ignore[arg-type]


@dataclass
class _Design:
    """Per-target design matrices: log-lik_i = sum log(X_i @ theta_i) - c @ theta_i."""

    X: list[Array]
    c: Array
    n_events: Array
    total_time: float


def _build_design(sequences: list[Sequence3], dim: int, beta: Array) -> _Design:
    beta = np.atleast_1d(beta)
    k = beta.size
    xs: list[list[Array]] = [[] for _ in range(dim)]
    comp = np.zeros((dim, k))
    total_time = 0.0
    n_events = np.zeros(dim, dtype=np.int64)
    for times, types, T in sequences:
        times = np.asarray(times, float)
        types = np.asarray(types, np.int64)
        total_time += float(T)
        r = excitation(times, types, dim, beta).reshape(times.size, dim * k)
        for i in range(dim):
            m = types == i
            n_events[i] += int(m.sum())
            xs[i].append(np.hstack([np.ones((int(m.sum()), 1)), r[m]]))
        tail = 1.0 - np.exp(-np.outer(T - times, beta))  # (N, K)
        for j in range(dim):
            comp[j] += tail[types == j].sum(axis=0)
    c = np.concatenate([[total_time], comp.ravel()])
    X = [np.vstack(x) if x else np.zeros((0, 1 + dim * k)) for x in xs]
    return _Design(X=X, c=c, n_events=n_events, total_time=total_time)


def _fit_block(X: Array, c: Array, n_em: int = 20, max_newton: int = 100, tol: float = 1e-10) -> tuple[Array, float]:
    """Maximise f(theta) = sum(log(X theta)) - c.theta over theta >= 0 (a concave problem).

    EM (minorise-maximise) warm start, then projected Newton with an Armijo line search: the
    Hessian -X' diag(1/lambda^2) X is available in closed form, so convergence is quadratic and
    a block with ~10^5 events and 9 parameters solves in a few milliseconds per iteration.
    """
    p = X.shape[1]
    n = X.shape[0]
    usable = c > 0                       # a column with zero compensator mass has no events
    theta = np.where(usable, 0.1 / max(p - 1, 1), 0.0)
    theta[0] = 0.5 * max(n, 1) / c[0]
    safe_c = np.where(usable, c, 1.0)
    for _ in range(n_em):                # theta_m <- theta_m * sum_n X_nm / lambda_n / c_m
        lam = X @ theta
        theta = np.where(usable, theta * (X.T @ (1.0 / lam)) / safe_c, 0.0)
        theta[0] = max(theta[0], 1e-12)

    def objective(th: Array) -> float:
        lam = X @ th
        if np.any(lam <= 0.0):
            return -np.inf
        return float(np.log(lam).sum() - c @ th)

    f_cur = objective(theta)
    for _ in range(max_newton):
        lam = X @ theta
        grad = X.T @ (1.0 / lam) - c
        pinned = (theta <= 0.0) & (grad <= 0.0)   # at the bound and wanting to stay there
        pinned[0] = False
        free = usable & ~pinned
        xf = X[:, free] / lam[:, None]
        info = xf.T @ xf                           # = -Hessian on the free block (PSD)
        gf = grad[free]
        try:
            direction = np.linalg.solve(info + 1e-12 * np.eye(info.shape[0]), gf)
        except np.linalg.LinAlgError:
            direction = gf
        step, accepted = 1.0, False
        for _ in range(60):
            cand = theta.copy()
            cand[free] = np.maximum(theta[free] + step * direction, 0.0)
            cand[0] = max(cand[0], 1e-12)
            f_new = objective(cand)
            if f_new >= f_cur + 1e-4 * float(gf @ (cand[free] - theta[free])):
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
        change = float(np.max(np.abs(cand - theta)))
        theta, f_cur = cand, f_new
        if change <= tol * (1.0 + float(np.max(np.abs(theta)))):
            break
    return theta, f_cur


def _block_se(X: Array, theta: Array) -> Array:
    """Standard errors from the observed information X' diag(1/lambda^2) X."""
    lam = X @ theta
    info = (X / lam[:, None]).T @ (X / lam[:, None])
    try:
        cov = np.linalg.inv(info)
        return np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        return np.full(theta.size, np.nan)


@dataclass
class HawkesFit:
    """Result of maximum-likelihood estimation."""

    params: HawkesParams
    loglik: float
    se_mu: Array
    se_alpha: Array
    se_beta: Array
    n_events: int
    total_time: float

    @property
    def aic(self) -> float:
        return 2.0 * self.params.n_params - 2.0 * self.loglik

    @property
    def bic(self) -> float:
        return self.params.n_params * np.log(self.n_events) - 2.0 * self.loglik

    def summary(self) -> dict:
        return {
            "params": self.params.to_dict(),
            "se_mu": self.se_mu.tolist(),
            "se_alpha": self.se_alpha.tolist(),
            "se_beta": self.se_beta.tolist(),
            "loglik": self.loglik,
            "aic": self.aic,
            "bic": self.bic,
            "spectral_radius": self.params.spectral_radius,
            "n_events": self.n_events,
            "total_time": self.total_time,
        }


def _fit_fixed_beta(
    sequences: list[Sequence3], dim: int, beta: Array, with_se: bool = True
) -> tuple[HawkesParams, float, Array, Array]:
    beta = np.atleast_1d(np.asarray(beta, float))
    k = beta.size
    design = _build_design(sequences, dim, beta)
    mu = np.zeros(dim)
    alpha = np.zeros((dim, dim, k))
    se_mu = np.full(dim, np.nan)
    se_alpha = np.full((dim, dim, k), np.nan)
    ll = 0.0
    for i in range(dim):
        theta, ll_i = _fit_block(design.X[i], design.c)
        mu[i], alpha[i] = theta[0], theta[1:].reshape(dim, k)
        if with_se:
            se = _block_se(design.X[i], theta)
            se_mu[i], se_alpha[i] = se[0], se[1:].reshape(dim, k)
        ll += ll_i
    return HawkesParams(mu, alpha, beta), ll, se_mu, se_alpha


def fit(
    data: Sequence3 | Iterable[Sequence3],
    dim: int,
    beta: Array | float | None = None,
    n_components: int = 1,
    beta_bounds: tuple[float, float] = (1e-2, 1e3),
    grid_size: int = 10,
) -> HawkesFit:
    """Maximum-likelihood fit of an exponential-kernel Hawkes process.

    Parameters
    ----------
    data : one ``(times, types, T)`` tuple or an iterable of them (independent episodes;
        their log-likelihoods add).
    dim : number of event types D.
    beta : decay rate(s) to hold fixed. If None they are estimated by profile likelihood.
    n_components : K, the number of exponential components (1 or 2 when ``beta`` is None).
    """
    seqs = _as_sequences(data)
    n_total = int(sum(len(s[0]) for s in seqs))
    t_total = float(sum(s[2] for s in seqs))
    lo, hi = np.log(beta_bounds[0]), np.log(beta_bounds[1])

    if beta is not None:
        b = np.atleast_1d(np.asarray(beta, float))
        params, ll, se_mu, se_alpha = _fit_fixed_beta(seqs, dim, b)
        return HawkesFit(params, ll, se_mu, se_alpha, np.zeros(b.size), n_total, t_total)

    if n_components == 1:
        def neg_profile(logb: float) -> float:
            return -_fit_fixed_beta(seqs, dim, np.array([np.exp(logb)]), with_se=False)[1]

        grid = np.linspace(lo, hi, grid_size)
        vals = np.array([neg_profile(g) for g in grid])
        i0 = int(np.argmin(vals))
        a, c = grid[max(i0 - 1, 0)], grid[min(i0 + 1, grid_size - 1)]
        res = optimize.minimize_scalar(neg_profile, bounds=(a, c), method="bounded",
                                       options={"xatol": 1e-5})
        logb = float(res.x)
        h = 1e-2
        f0, fp, fm = res.fun, neg_profile(logb + h), neg_profile(logb - h)
        curv = (fp - 2 * f0 + fm) / h**2  # second derivative of the negative profile log-lik
        se_logb = 1.0 / np.sqrt(curv) if curv > 0 else np.nan
        b = np.array([np.exp(logb)])
        params, ll, se_mu, se_alpha = _fit_fixed_beta(seqs, dim, b)
        return HawkesFit(params, ll, se_mu, se_alpha, b * se_logb, n_total, t_total)

    if n_components == 2:
        grid = np.linspace(lo, hi, grid_size)
        best = (np.inf, None)
        for i in range(grid_size):
            for j in range(i + 1, grid_size):
                v = -_fit_fixed_beta(seqs, dim, np.exp([grid[j], grid[i]]), with_se=False)[1]
                if v < best[0]:
                    best = (v, np.array([grid[j], grid[i]]))
        res = optimize.minimize(
            lambda lb: -_fit_fixed_beta(seqs, dim, np.exp(np.sort(lb)[::-1]), with_se=False)[1],
            best[1], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 200},
        )
        b = np.exp(np.sort(res.x)[::-1])
        params, ll, se_mu, se_alpha = _fit_fixed_beta(seqs, dim, b)
        return HawkesFit(params, ll, se_mu, se_alpha, np.full(2, np.nan), n_total, t_total)

    raise ValueError("profile estimation supports n_components in {1, 2}")


def loglik(data: Sequence3 | Iterable[Sequence3], params: HawkesParams) -> float:
    """Exact log-likelihood of ``params`` on one or more episodes."""
    seqs = _as_sequences(data)
    design = _build_design(seqs, params.dim, params.beta)
    total = 0.0
    for i in range(params.dim):
        theta = np.concatenate([[params.mu[i]], params.alpha[i].ravel()])
        total += float(np.log(design.X[i] @ theta).sum() - design.c @ theta)
    return total


# ---------------------------------------------------------------------------
# Goodness of fit
# ---------------------------------------------------------------------------
def compensator_at_events(times: Array, types: Array, params: HawkesParams) -> Array:
    """Lambda_i(t_n) = int_0^{t_n} lambda_i(s) ds for every event n and type i, shape (N, D)."""
    times = np.asarray(times, float)
    types = np.asarray(types, np.int64)
    d = params.dim
    r = excitation(times, types, d, params.beta)  # (N, D, K)
    onehot = np.zeros((times.size, d))
    onehot[np.arange(times.size), types] = 1.0
    counts_before = np.cumsum(onehot, axis=0) - onehot  # N_j(t_n-)
    # sum_{l<n, d_l=j} (1 - exp(-beta_k (t_n - t_l))) = N_j(t_n-) - R[n, j, k] / beta_k
    integrated = counts_before[:, :, None] - r / params.beta[None, None, :]
    return params.mu[None, :] * times[:, None] + np.einsum("ijk,njk->ni", params.alpha, integrated)


def rescaled_interarrivals(times: Array, types: Array, params: HawkesParams) -> list[Array]:
    """Time-rescaling residuals: under a correct model they are i.i.d. Exp(1) for each type."""
    lam = compensator_at_events(times, types, params)
    out = []
    for i in range(params.dim):
        idx = np.flatnonzero(np.asarray(types) == i)
        out.append(np.diff(np.r_[0.0, lam[idx, i]]))
    return out


def poisson_rescaled_interarrivals(times: Array, types: Array, T: float, dim: int) -> list[Array]:
    """Residuals of the best homogeneous-Poisson fit (the null a Hawkes model must beat)."""
    out = []
    for i in range(dim):
        ti = np.asarray(times)[np.asarray(types) == i]
        rate = ti.size / T
        out.append(rate * np.diff(np.r_[0.0, ti]))
    return out


def ks_exponential(residuals: Array) -> tuple[float, float]:
    """Kolmogorov-Smirnov statistic and p-value against Exp(1)."""
    res = stats.kstest(residuals, "expon")
    return float(res.statistic), float(res.pvalue)


# ---------------------------------------------------------------------------
# Forecasting and online state
# ---------------------------------------------------------------------------
def _state_dynamics(params: HawkesParams) -> tuple[Array, Array, Array]:
    """Linear ODE dE[y]/dt = A E[y] + b for the excitation state y (index i*K + k)."""
    d, k = params.dim, params.n_components
    beta = params.beta
    n = d * k
    a = np.zeros((n, n))
    b = np.zeros(n)
    for i in range(d):
        for c in range(k):
            r = i * k + c
            a[r, r] -= beta[c]
            for j in range(d):
                a[r, j * k: (j + 1) * k] += beta[c] * params.alpha[i, j, c]
            b[r] = beta[c] * float(params.alpha[i, :, c] @ params.mu)
    s = np.kron(np.eye(d), np.ones((1, k)))  # sums the K components of each type
    return a, b, s


class HawkesForecaster:
    """Closed-form conditional expectations for an exponential-kernel Hawkes process.

    With excitation state y (lambda = mu + S y), taking expectations of dy = -B y dt + B alpha dN
    gives a linear ODE dE[y]/dt = A E[y] + b. Hence

        E[y(t + s) | F_t] = y_inf + exp(A s) (y(t) - y_inf)
        E[N(t, t + h] | F_t] = mu h + S [ y_inf h + A^{-1} (exp(A h) - I) (y(t) - y_inf) ],

    an affine function of the current state that is precomputed once.
    """

    def __init__(self, params: HawkesParams, horizon: float):
        self.params = params
        self.horizon = float(horizon)
        a, b, s = _state_dynamics(params)
        n = a.shape[0]
        self._a, self._s = a, s
        self.y_inf = -np.linalg.solve(a, b)
        e = linalg.expm(a * self.horizon)
        self.gain = s @ np.linalg.solve(a, e - np.eye(n))  # (D, DK)
        self.offset = params.mu * self.horizon + s @ self.y_inf * self.horizon - self.gain @ self.y_inf

    def expected_counts(self, y: Array) -> Array:
        """E[N_i(t, t + horizon] | state y] for every type i."""
        return self.offset + self.gain @ y

    def expected_intensity(self, y: Array, s: float) -> Array:
        e = linalg.expm(self._a * s)
        return self.params.mu + self._s @ (self.y_inf + e @ (y - self.y_inf))


class HawkesState:
    """Online excitation state y(t) with O(D K) updates; intensity = mu + S y."""

    __slots__ = ("mu", "beta_flat", "jump", "y", "t", "_k")

    def __init__(self, params: HawkesParams, t0: float = 0.0):
        d, k = params.dim, params.n_components
        self.mu = params.mu
        self._k = k
        self.beta_flat = np.tile(params.beta, d)
        self.jump = (params.alpha * params.beta[None, None, :]).transpose(1, 0, 2).reshape(d, d * k)
        self.y = np.zeros(d * k)
        self.t = float(t0)

    def update(self, t: float, etype: int) -> None:
        """Advance to time t and register an event of type ``etype``."""
        if t > self.t:
            self.y *= np.exp(-self.beta_flat * (t - self.t))
            self.t = t
        self.y += self.jump[etype]

    def state_at(self, t: float) -> Array:
        return self.y * np.exp(-self.beta_flat * max(t - self.t, 0.0))

    def intensity_at(self, t: float) -> Array:
        return self.mu + self.state_at(t).reshape(-1, self._k).sum(axis=1)
