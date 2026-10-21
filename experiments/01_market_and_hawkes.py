"""Experiment 1: does the simulated market look like a market, and can we learn its order flow?

A. Stylised facts of the simulated book (training episodes, no market maker).
B. Hawkes estimation on training episodes: Poisson vs one- vs two-exponential kernels, model
   selection by BIC, goodness of fit by the time-rescaling theorem on *held-out* episodes.
C. Estimator validation: parameter recovery and confidence-interval coverage on synthetic data
   generated from the fitted one-exponential model (where the truth is known).

Writes results/hawkes.json and docs/figures/fig1_market.png, fig2_hawkes.png.
"""

from __future__ import annotations

import numpy as np
from _common import (DURATION, FIGURES, INK, INK2, MUTED, SERIES, SURFACE, TRAIN, VALID, save_figure,
                     save_json, setup_style, step)
from scipy import stats

from quotelab import hawkes as hk
from quotelab.book import ASK, BID
from quotelab.calibration import realized_vol
from quotelab.market import EVENT_NAMES, MarketConfig, generate_stream
from quotelab.simulator import ExecutionConfig, Simulator, Strategy

LEVELS = 5


class BookRecorder(Strategy):
    """A 'strategy' that never quotes and samples the depth profile once per second."""

    name = "book_recorder"

    def __init__(self):
        self.next_t = 0.0
        self.bid = []
        self.ask = []

    def quote(self, t, view, inventory):
        if t >= self.next_t:
            book = view.book
            bb, ba = book.best_bid, book.best_ask
            self.bid.append([book.level_qty(BID, bb - k) for k in range(LEVELS)])
            self.ask.append([book.level_qty(ASK, ba + k) for k in range(LEVELS)])
            self.next_t = t + 1.0
        return None, None


def stylized_facts(cfg: MarketConfig) -> tuple[dict, dict]:
    runs, depth_b, depth_a = [], [], []
    for s in TRAIN:
        stream = generate_stream(cfg, cfg.warmup + DURATION, s)
        rec = BookRecorder()
        runs.append(Simulator(cfg, stream, rec, ExecutionConfig(sample_dt=1.0), record_trades=True).run())
        depth_b += rec.bid
        depth_a += rec.ask
    spreads = np.concatenate([r.grid_spread for r in runs])
    ret1 = np.concatenate([np.diff(r.grid_mid) for r in runs])
    # market-order sign autocorrelation, averaged over episodes
    lags = np.unique(np.round(np.logspace(0, 2, 15)).astype(int))
    acf = np.mean([[np.corrcoef(r.trades_sign[:-k], r.trades_sign[k:])[0, 1] for k in lags] for r in runs], axis=0)
    gaps = np.concatenate([np.diff(r.trades_t) for r in runs])
    sig = {h: realized_vol(runs, h) for h in (1, 5, 30, 120, 300)}
    facts = {
        "events_per_second": float(np.mean([r.n_events / (r.t_end) for r in runs])),
        "market_orders_per_second": float(np.mean([r.trades_t.size / r.t_end for r in runs])),
        "spread_share": {"1": float(np.mean(spreads == 1)), "2": float(np.mean(spreads == 2)),
                         "3+": float(np.mean(spreads >= 3)), "one_sided": float(np.mean(spreads < 0))},
        "depth_bid": np.mean(depth_b, axis=0).tolist(),
        "depth_ask": np.mean(depth_a, axis=0).tolist(),
        "sigma_ticks_per_sqrt_s": sig,
        "excess_kurtosis_1s": float(stats.kurtosis(ret1)),
        "share_of_1s_intervals_without_mid_move": float(np.mean(ret1 == 0)),
        "mo_sign_acf": dict(zip(lags.tolist(), acf.tolist())),
        "mo_interarrival_cv": float(gaps.std() / gaps.mean()),
    }
    plot_data = {"run": runs[0], "lags": lags, "acf": acf, "gaps": gaps}
    return facts, plot_data


def poisson_loglik(seqs, dim: int) -> tuple[float, int]:
    ll = 0.0
    for i in range(dim):
        n = sum(int((s[1] == i).sum()) for s in seqs)
        T = sum(s[2] for s in seqs)
        ll += n * np.log(n / T) - n
    return ll, dim


def model_row(name: str, ll: float, n_params: int, n_events: int, rho: float | None, beta) -> dict:
    return {"model": name, "loglik": ll, "n_params": n_params, "aic": 2 * n_params - 2 * ll,
            "bic": n_params * np.log(n_events) - 2 * ll, "spectral_radius": rho,
            "beta": None if beta is None else np.atleast_1d(beta).tolist()}


def recovery_study(true: hk.HawkesParams, reps: int, T: float) -> dict:
    """Simulate from known parameters, refit (decay profiled out), and check 95% CI coverage."""
    est, se = [], []
    for r in range(reps):
        t, d = hk.simulate(true, T, rng=50_000 + r)
        f = hk.fit((t, d, T), true.dim, beta_bounds=(0.5, 50.0), grid_size=8)
        est.append(np.concatenate([f.params.mu, f.params.alpha.ravel(), f.params.beta]))
        se.append(np.concatenate([f.se_mu, f.se_alpha.ravel(), f.se_beta]))
    est, se = np.array(est), np.array(se)
    truth = np.concatenate([true.mu, true.alpha.ravel(), true.beta])
    z = (est - truth) / se
    big = truth > 0.02  # boundary parameters (true value ~0) have non-normal estimators
    return {
        "reps": reps, "T": T,
        "coverage_95": float(np.mean(np.abs(z[:, big]) < 1.96)),
        "mean_z": float(np.mean(z[:, big])), "std_z": float(np.std(z[:, big])),
        "z_scores": z[:, big].ravel().tolist(),
        "relative_bias": float(np.mean((est[:, big].mean(0) - truth[big]) / truth[big])),
        "n_params_checked": int(big.sum()),
    }


def main() -> None:
    setup_style()
    import matplotlib.pyplot as plt

    cfg = MarketConfig()
    out: dict = {}

    with step("A. stylised facts"):
        facts, pdata = stylized_facts(cfg)
        out["stylized_facts"] = facts

    with step("B. Hawkes estimation (training episodes)"):
        seqs = [generate_stream(cfg, cfg.warmup + DURATION, s).hawkes_sequence() for s in TRAIN]
        n_events = sum(len(s[0]) for s in seqs)
        ll_p, k_p = poisson_loglik(seqs, 4)
        fit1 = hk.fit(seqs, 4, beta_bounds=(0.05, 50.0), grid_size=10)
        fit2 = hk.fit(seqs, 4, n_components=2, beta_bounds=(0.05, 50.0), grid_size=6)
        ll_true = hk.loglik(seqs, cfg.hawkes)
        table = [
            model_row("Poisson", ll_p, k_p, n_events, None, None),
            model_row("Hawkes, 1 exponential", fit1.loglik, fit1.params.n_params, n_events,
                      fit1.params.spectral_radius, fit1.params.beta),
            model_row("Hawkes, 2 exponentials", fit2.loglik, fit2.params.n_params, n_events,
                      fit2.params.spectral_radius, fit2.params.beta),
            model_row("True data-generating process", ll_true, cfg.hawkes.n_params, n_events,
                      cfg.hawkes.spectral_radius, cfg.hawkes.beta),
        ]
        selected = "k2" if fit2.bic < fit1.bic else "k1"
        out["n_events_train"] = n_events
        out["model_table"] = table
        out["fit_k1"] = fit1.summary()
        out["fit_k2"] = fit2.summary()
        out["selected"] = selected
        # profile log-likelihood curve (one exponential)
        grid = np.exp(np.linspace(np.log(0.5), np.log(40.0), 25))
        prof = [hk.fit(seqs, 4, beta=b).loglik for b in grid]
        out["profile_k1"] = {"beta": grid.tolist(), "loglik": prof}

    with step("B'. goodness of fit on held-out episodes"):
        held = [generate_stream(cfg, cfg.warmup + DURATION, s).hawkes_sequence() for s in VALID[:5]]
        gof = {}
        resid_pool = {}
        for name, params in [("Poisson", None), ("Hawkes, 1 exponential", fit1.params),
                             ("Hawkes, 2 exponentials", fit2.params)]:
            per_type = [[] for _ in range(4)]
            for t, d, T in held:
                res = (hk.poisson_rescaled_interarrivals(t, d, T, 4) if params is None
                       else hk.rescaled_interarrivals(t, d, params))
                for i in range(4):
                    per_type[i].append(res[i])
            gof[name] = {EVENT_NAMES[i]: dict(zip(("ks_stat", "p_value"),
                                                  hk.ks_exponential(np.concatenate(per_type[i]))))
                         for i in range(4)}
            resid_pool[name] = np.concatenate(per_type[0])  # market buys, for the QQ plot
        out["gof_heldout"] = gof

    with step("C. estimator validation by simulation"):
        out["recovery"] = recovery_study(fit1.params, reps=40 if len(TRAIN) > 4 else 4, T=DURATION)

    save_json("hawkes.json", out)

    # ------------------------------------------------------------------ figure 1: market
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.2))
    r = pdata["run"]
    m = (r.mid_t >= 300) & (r.mid_t <= 900)
    a = ax[0, 0]
    a.step(r.mid_t[m] - 300, r.mid_v[m], where="post", color=SERIES[0], lw=1.6)
    a.set_title("Mid-price, 10 minutes of one episode")
    a.set_xlabel("seconds")
    a.set_ylabel("ticks")

    a = ax[0, 1]
    lv = np.arange(1, LEVELS + 1)
    w = 0.36
    a.bar(lv - w / 2 - 0.01, facts["depth_bid"], width=w, color=SERIES[0], label="bid")
    a.bar(lv + w / 2 + 0.01, facts["depth_ask"], width=w, color=SERIES[1], label="ask")
    a.set_xticks(lv, ["best", "+1", "+2", "+3", "+4"])
    a.set_title("Average depth by distance from the best quote")
    a.set_xlabel("ticks behind the touch")
    a.set_ylabel("resting quantity (lots)")
    a.legend(loc="upper right")

    a = ax[1, 0]
    a.axhline(0, color=MUTED, lw=0.8)
    a.plot(pdata["lags"], pdata["acf"], color=SERIES[0], marker="o", ms=5, mec=SURFACE, mew=1.5)
    a.set_xscale("log")
    a.set_title("Market-order signs are autocorrelated")
    a.set_xlabel("lag (market orders)")
    a.set_ylabel("autocorrelation of buy/sell sign")

    a = ax[1, 1]
    g = np.sort(pdata["gaps"])
    surv = 1.0 - np.arange(g.size) / g.size
    xs = np.linspace(0, np.quantile(g, 0.999), 400)
    a.plot(g, surv, color=SERIES[0], label="simulated market orders")
    a.plot(xs, np.exp(-xs / g.mean()), color=SERIES[1], label="Poisson, same rate")
    a.set_yscale("log")
    a.set_xlim(0, np.quantile(g, 0.999))
    a.set_ylim(1e-3, 1.05)
    a.set_title(f"Clustered arrivals (CV of gaps = {facts['mo_interarrival_cv']:.2f})")
    a.set_xlabel("time between market orders (s)")
    a.set_ylabel("P(gap > x)")
    a.legend(loc="upper right")
    fig.tight_layout()
    save_figure(fig, "fig1_market")

    # ------------------------------------------------------------------ figure 2: Hawkes
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.2))
    a = ax[0, 0]
    q = np.linspace(0.005, 0.995, 120)
    theo = -np.log(1 - q)
    a.plot([0, theo[-1]], [0, theo[-1]], color=MUTED, lw=1.0)
    for (name, res), col in zip(resid_pool.items(), SERIES[:3]):
        a.plot(theo, np.quantile(res, q), color=col, label=name)
    a.set_title("Time-rescaled residuals vs Exp(1), held-out data")
    a.set_xlabel("Exp(1) quantile")
    a.set_ylabel("residual quantile (market buys)")
    a.legend(loc="upper left")

    a = ax[0, 1]
    pr = out["profile_k1"]
    a.plot(pr["beta"], np.array(pr["loglik"]) - max(pr["loglik"]), color=SERIES[0])
    a.axvline(fit1.params.beta[0], color=MUTED, lw=1.0)
    a.text(0.5, 0.06, f"maximum at beta = {fit1.params.beta[0]:.2f}/s (vertical line)", transform=a.transAxes,
           ha="center", color=INK2, fontsize=8.5)
    a.set_xscale("log")
    a.set_title("Profile log-likelihood of the decay (1-exponential model)")
    a.set_xlabel("beta (1/s)")
    a.set_ylabel("log-likelihood minus maximum")

    a = ax[1, 0]
    z = np.array(out["recovery"]["z_scores"])
    a.hist(z, bins=30, density=True, color=SERIES[0], alpha=0.85, edgecolor=SURFACE, linewidth=1.0)
    xs = np.linspace(-4, 4, 200)
    a.plot(xs, stats.norm.pdf(xs), color=SERIES[1], label="N(0, 1)")
    a.set_title(f"Recovery z-scores: 95% CI coverage {out['recovery']['coverage_95']:.1%}")
    a.set_xlabel("(estimate - truth) / standard error")
    a.set_ylabel("density")
    a.legend(loc="upper right")

    a = ax[1, 1]
    tt = np.logspace(-3, 1.3, 200)

    def kernel(p, i=0, j=0):
        return (p.alpha[i, j, :][None, :] * p.beta[None, :] * np.exp(-np.outer(tt, p.beta))).sum(1)

    a.plot(tt, kernel(cfg.hawkes), color=INK, lw=1.2, label="true (2 time scales)")
    a.plot(tt, kernel(fit1.params), color=SERIES[0], label="fitted, 1 exponential")
    a.plot(tt, kernel(fit2.params), color=SERIES[1], label="fitted, 2 exponentials")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_ylim(1e-4, 10.0)
    a.set_title("Self-excitation kernel of market buys")
    a.set_xlabel("time since a market buy (s)")
    a.set_ylabel("excitation (events/s)")
    a.legend(loc="lower left")
    fig.tight_layout()
    save_figure(fig, "fig2_hawkes")
    print("selected model:", selected, "| figures in", FIGURES)


if __name__ == "__main__":
    main()
