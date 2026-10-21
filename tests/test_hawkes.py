import numpy as np
import pytest

from quotelab import hawkes as hk


@pytest.fixture(scope="module")
def params2() -> hk.HawkesParams:
    return hk.HawkesParams.exponential(
        mu=[0.5, 0.3], alpha=np.array([[0.4, 0.2], [0.1, 0.5]]), beta=3.0
    )


@pytest.fixture(scope="module")
def long_path(params2):
    t, d = hk.simulate(params2, 4000.0, rng=1)
    return t, d, 4000.0


def brute_force_excitation(t, d, dim, beta):
    out = np.zeros((t.size, dim))
    for n in range(t.size):
        dt = t[n] - t[:n]
        for j in range(dim):
            m = d[:n] == j
            out[n, j] = (beta * np.exp(-beta * dt[m])).sum()
    return out


def test_param_validation():
    with pytest.raises(ValueError):
        hk.HawkesParams(np.ones(2), np.ones((3, 3)), np.ones(1))
    with pytest.raises(ValueError):
        hk.HawkesParams(np.ones(2), -np.ones((2, 2)), np.ones(1))


def test_excitation_matches_brute_force(params2):
    t, d = hk.simulate(params2, 300.0, rng=7)
    fast = hk.excitation(t, d, 2, np.array([3.0, 0.4]))
    for k, b in enumerate([3.0, 0.4]):
        np.testing.assert_allclose(fast[:, :, k], brute_force_excitation(t, d, 2, b), atol=1e-10)


def test_excitation_survives_long_gaps():
    # Two bursts separated by a gap long enough to underflow exp(-beta * gap).
    t = np.array([0.0, 0.01, 0.02, 1e4, 1e4 + 0.01])
    d = np.array([0, 0, 1, 0, 1])
    fast = hk.excitation(t, d, 2, np.array([50.0]))[:, :, 0]
    np.testing.assert_allclose(fast, brute_force_excitation(t, d, 2, 50.0), atol=1e-12)


def test_cluster_simulation_matches_stationary_rate(params2, long_path):
    t, d, T = long_path
    emp = np.bincount(d, minlength=2) / T
    lam = params2.stationary_intensity()
    # Hawkes counts are over-dispersed: Var N(T) ~ T * lam / (1 - rho)^2; allow 4 sigma.
    sd = np.sqrt(lam / T) / (1 - params2.spectral_radius)
    assert np.all(np.abs(emp - lam) < 4 * sd)


def test_ogata_and_cluster_agree_with_closed_form_mean(params2):
    # From an empty history, E[N(T)] = offset of the forecaster evaluated at state y = 0.
    T, reps = 20.0, 300
    expected = hk.HawkesForecaster(params2, T).expected_counts(np.zeros(2))
    ogata = np.array([np.bincount(hk.simulate_ogata(params2, T, rng=s)[1], minlength=2) for s in range(reps)])
    clust = np.array([np.bincount(hk.simulate(params2, T, rng=10_000 + s, burn_in=0.0)[1], minlength=2)
                      for s in range(reps)])
    for sample in (ogata, clust):
        se = sample.std(axis=0, ddof=1) / np.sqrt(reps)
        assert np.all(np.abs(sample.mean(axis=0) - expected) < 4 * se)


def test_mle_recovers_parameters(params2, long_path):
    fit = hk.fit(long_path, dim=2)
    true = np.concatenate([params2.mu, params2.alpha.ravel(), params2.beta])
    est = np.concatenate([fit.params.mu, fit.params.alpha.ravel(), fit.params.beta])
    se = np.concatenate([fit.se_mu, fit.se_alpha.ravel(), fit.se_beta])
    z = (est - true) / se
    assert np.all(np.abs(z) < 4.0), z
    assert fit.loglik >= hk.loglik(long_path, params2) - 1e-6  # the MLE beats the truth in-sample


def test_fixed_beta_fit_is_global_optimum_of_concave_problem(params2, long_path):
    fit = hk.fit(long_path, dim=2, beta=3.0)
    rng = np.random.default_rng(0)
    for _ in range(20):  # random feasible perturbations never improve the likelihood
        mu = np.clip(fit.params.mu * np.exp(0.05 * rng.standard_normal(2)), 1e-9, None)
        al = np.clip(fit.params.alpha * np.exp(0.05 * rng.standard_normal((2, 2, 1))), 0, None)
        assert hk.loglik(long_path, hk.HawkesParams(mu, al, fit.params.beta)) <= fit.loglik + 1e-8


def test_two_component_fit_with_known_decays():
    p = hk.HawkesParams(
        mu=np.array([0.4, 0.4]),
        alpha=np.stack([np.array([[0.3, 0.1], [0.1, 0.3]]), np.array([[0.2, 0.05], [0.05, 0.2]])], axis=2),
        beta=np.array([10.0, 0.5]),
    )
    t, d = hk.simulate(p, 6000.0, rng=3)
    fit = hk.fit((t, d, 6000.0), dim=2, beta=p.beta)
    z = (fit.params.alpha - p.alpha) / fit.se_alpha
    assert np.all(np.abs(z) < 4.5), z


def test_time_rescaling_accepts_truth_and_rejects_poisson(params2, long_path):
    t, d, T = long_path
    for res in hk.rescaled_interarrivals(t, d, params2):
        assert hk.ks_exponential(res)[1] > 1e-3
    pvals = [hk.ks_exponential(r)[1] for r in hk.poisson_rescaled_interarrivals(t, d, T, 2)]
    assert max(pvals) < 1e-3


def test_forecaster_consistent_with_stationary_intensity(params2):
    f = hk.HawkesForecaster(params2, horizon=5.0)
    lam_inf = params2.mu + f._s @ f.y_inf
    np.testing.assert_allclose(lam_inf, params2.stationary_intensity(), rtol=1e-12)
    # at the stationary mean state, expected counts are lam_inf * horizon
    np.testing.assert_allclose(f.expected_counts(f.y_inf), lam_inf * 5.0, rtol=1e-12)


def test_conditional_forecast_matches_monte_carlo(params2):
    # Condition on a burst of type-0 events at t = 0 and compare E[N(0, h]] with simulation.
    h, reps = 2.0, 400
    state = hk.HawkesState(params2)
    for _ in range(3):
        state.update(0.0, 0)
    analytic = hk.HawkesForecaster(params2, h).expected_counts(state.y)
    counts = []
    rng = np.random.default_rng(5)
    for _ in range(reps):
        # simulate continuation by Ogata thinning seeded with the burst
        y = np.zeros((2, 1)) + state.y.reshape(2, 1)
        t, n = 0.0, np.zeros(2)
        while True:
            lam_bar = params2.mu.sum() + y.sum()
            t_new = t + rng.exponential(1 / lam_bar)
            if t_new >= h:
                break
            y *= np.exp(-params2.beta * (t_new - t))
            t = t_new
            lam = params2.mu + y.sum(axis=1)
            u = rng.random() * lam_bar
            if u < lam.sum():
                j = int(u >= lam[0])
                n[j] += 1
                y += params2.alpha[:, j, :] * params2.beta
        counts.append(n)
    counts = np.array(counts)
    se = counts.std(axis=0, ddof=1) / np.sqrt(reps)
    assert np.all(np.abs(counts.mean(axis=0) - analytic) < 4 * se)


def test_online_state_matches_batch_intensity(params2):
    t, d = hk.simulate(params2, 200.0, rng=11)
    r = hk.excitation(t, d, 2, params2.beta)
    batch = params2.mu + np.einsum("ijk,njk->ni", params2.alpha, r)
    state = hk.HawkesState(params2)
    online = []
    for ti, di in zip(t, d):
        online.append(state.intensity_at(ti))  # intensity just before the event
        state.update(ti, int(di))
    np.testing.assert_allclose(np.array(online), batch, rtol=1e-9, atol=1e-12)


def test_newton_solver_matches_lbfgsb(params2, long_path):
    """The projected-Newton block solver reaches the same optimum as a generic quasi-Newton."""
    from scipy import optimize

    design = hk._build_design([long_path], 2, params2.beta)
    for i in range(2):
        X, c = design.X[i], design.c
        theta, f_newton = hk._fit_block(X, c)

        def negll(th):
            lam = X @ th
            return -(np.log(lam).sum() - c @ th), -(X.T @ (1.0 / lam) - c)

        res = optimize.minimize(negll, np.full(X.shape[1], 0.2), jac=True, method="L-BFGS-B",
                                bounds=[(1e-12, None)] + [(0, None)] * (X.shape[1] - 1),
                                options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10_000})
        assert f_newton >= -res.fun - 1e-7
        np.testing.assert_allclose(theta, res.x, atol=1e-4)
