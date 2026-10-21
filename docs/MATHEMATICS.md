# Mathematical background

The models behind QuoteLab, with derivations, and how each piece is implemented and tested.
Numbers quoted here come from `results/`; see the [research report](RESEARCH_REPORT.md) for the full
experimental results.

## 1. Hawkes processes

**Definition.** For D event types the intensity of type i is

$$\lambda_i(t) = \mu_i + \sum_{j}\sum_{t^j_l < t} \phi_{ij}(t - t^j_l), \qquad \phi_{ij}(s) = \sum_k \alpha_{ijk}\,\beta_k e^{-\beta_k s}.$$

Each kernel integrates to $\alpha_{ij} = \sum_k \alpha_{ijk}$, so $G = [\alpha_{ij}]$ is the
**branching matrix**: $G_{ij}$ is the expected number of type-i events triggered directly by one
type-j event.

**Branching (cluster) representation.** Immigrants arrive as a Poisson process with rate $\mu$.
Each event of type j has Poisson($G_{ij}$) children of type i, each delayed by a draw from
$\phi_{ij}/G_{ij}$. The expected total progeny of one event is $\sum_{n\ge1} G^n$, finite iff the
spectral radius $\rho(G) < 1$. Then the stationary mean intensity solves
$\bar\lambda = \mu + G\bar\lambda$, so $\bar\lambda = (I - G)^{-1}\mu$.
*Interpretation:* with one event type the branching ratio is exactly the fraction of events triggered
by earlier events rather than arriving exogenously (the market's "endogeneity"); with several types
that share is $1 - \mu_i/\bar\lambda_i$ for type i, and $\rho(G)$ governs stability. Estimates on
real markets are often high, close to 1 (Filimonov & Sornette 2012; Hardiman, Bercot & Bouchaud 2013).

**Likelihood.**

$$\ell = \sum_i \Big[\sum_{n: d_n = i} \log \lambda_i(t_n) - \int_0^T \lambda_i(s)\,ds\Big],\qquad
\int_0^T \lambda_i = \mu_i T + \sum_{j,k}\alpha_{ijk}\sum_{l: d_l=j}\big(1 - e^{-\beta_k (T - t_l)}\big).$$

**O(N) evaluation.** With $R_{jk}(t_n) = \sum_{l<n,\,d_l=j}\beta_k e^{-\beta_k(t_n - t_l)}$,

$$R_{jk}(t_n) = e^{-\beta_k (t_n - t_{n-1})}\big(R_{jk}(t_{n-1}) + \beta_k \mathbf 1\{d_{n-1} = j\}\big).$$

`hawkes.excitation` evaluates this without a Python loop over events: inside a window of length
$C/\beta$ it is a cumulative sum of $\beta e^{\beta\tau}$ terms (exponents stay below $C$, so nothing
overflows), and a decayed carry links windows. Tested against brute force to 1e-10.

**Why the problem is easy for fixed decays.** $\lambda_i(t_n) = \theta_i^\top x_n$ with
$\theta_i = (\mu_i, \alpha_{i\cdot\cdot})$ and $x_n = (1, R(t_n)) \ge 0$. The log of an affine
function is concave and the compensator is linear in $\theta_i$, so $\ell$ is **concave** in
$(\mu, \alpha)$ and **separates** over target types. Each block is a small convex problem:

* gradient $g = \sum_n x_n/\lambda_n - c$, Hessian $H = -\sum_n x_n x_n^\top/\lambda_n^2$;
* warm start with EM (an MM algorithm): $\theta_m \leftarrow \theta_m \sum_n (x_{nm}/\lambda_n)/c_m$,
  which comes from the branching structure (each event is attributed to the immigrant or to a
  parent component with probability $\theta_m x_{nm}/\lambda_n$);
* then projected Newton with an Armijo line search (`_fit_block`). Tested to reach the same optimum
  as L-BFGS-B.

The decays are profiled out: $\ell_p(\beta) = \max_\theta \ell(\theta, \beta)$, maximised by a grid
plus Brent (one exponential) or a grid plus Nelder-Mead (two). Standard errors of $(\mu,\alpha)$
come from the observed information $\sum_n x_n x_n^\top/\lambda_n^2$; for $\beta$ from the
curvature of the profile likelihood (delta method on $\log\beta$).

**Goodness of fit: time-rescaling theorem.** If $\Lambda_i(t) = \int_0^t \lambda_i$ is the true
compensator, the transformed event times $\Lambda_i(t_n)$ form a unit-rate Poisson process, so the
increments are i.i.d. Exp(1). Test with Kolmogorov-Smirnov and a QQ plot. A Poisson model fails this
test spectacularly on clustered data; that is the cleanest argument for Hawkes.

**Model selection.** BIC $= p\log N - 2\ell$. The data are generated with two time scales (about 0.1 s
and 3 s); BIC prefers the two-exponential model. A single exponential cannot represent the slow
component, so it attributes part of it to the baseline: the fitted $\mu$ goes up and the fitted
branching ratio goes down. That is a good example of what misspecification does to parameters.

**Closed-form forecasts.** Write $\lambda = \mu + S y$ where $y$ stacks the kernel sums. Between
events $dy = -By\,dt$; at a type-j event $y$ jumps by $B\alpha_{\cdot j\cdot}$. Taking expectations,

$$\frac{d}{dt}E[y] = A\,E[y] + b, \qquad A = -B + B\,\alpha\,S, \quad b = B\,\alpha\,\mu,$$

so $E[y(t+s)\mid\mathcal F_t] = y_\infty + e^{As}(y(t) - y_\infty)$ with $y_\infty = -A^{-1}b$, and

$$E[N(t, t+h]\mid\mathcal F_t] = \mu h + S\big[y_\infty h + A^{-1}(e^{Ah} - I)(y(t) - y_\infty)\big],$$

an affine function of the current state. `HawkesForecaster` precomputes the matrices once, so a
forecast inside the trading loop costs one small dot product. Tested against Monte Carlo.

## 2. Optimal market making

**Setup (Avellaneda-Stoikov 2008).** Mid $dS = \sigma dW$. The market maker posts $S - \delta^b$
and $S + \delta^a$; market orders hit them as Poisson processes with intensity $\Lambda(\delta) =
Ae^{-k\delta}$. Maximise $E[-\exp(-\gamma(X_T + q_T S_T))]$.

**HJB.**

$$0 = \partial_t u + \tfrac12\sigma^2\partial_{ss}u
+ \max_{\delta^b} Ae^{-k\delta^b}\big[u(x - s + \delta^b, q+1) - u\big]
+ \max_{\delta^a} Ae^{-k\delta^a}\big[u(x + s + \delta^a, q-1) - u\big].$$

**Ansatz** $u = -e^{-\gamma x}e^{-\gamma\theta(t,s,q)}$. Define the indifference (reservation) prices
$r^b = \theta(q+1) - \theta(q)$ and $r^a = \theta(q) - \theta(q-1)$ and let $\eta^b = s - r^b$.
The bid term becomes $(-u)\,Ae^{-k\delta}\big(1 - e^{-\gamma(\delta - \eta^b)}\big)$, so the
first-order condition gives

$$\delta^{b*} = \eta^b + \frac1\gamma\ln\Big(1 + \frac\gamma k\Big),$$

an inventory term plus a **monopoly markup**. Same for the ask.

**AS approximation.** Use the value of holding $q$ without trading,
$\theta = qs - \tfrac12\gamma\sigma^2 q^2 (T-t)$. Then $\eta^b = (q + \tfrac12)\gamma\sigma^2(T - t)$,
the reservation price is $r = s - q\gamma\sigma^2(T-t)$, and the total spread is
$\gamma\sigma^2(T-t) + \tfrac2\gamma\ln(1 + \tfrac\gamma k)$.

*Known weaknesses:* the inventory penalty vanishes at $T$, exactly when a desk most
wants to be flat; it grows without bound for long horizons; inventory is unconstrained.

**GLFT (Guéant, Lehalle & Fernandez-Tapia 2013).** Add the constraint $|q| \le Q$. With
$u = -e^{-\gamma(x + qs)}v_q(t)^{-\gamma/k}$ the HJB becomes a **linear** ODE system

$$\dot v_q = \alpha q^2 v_q - \eta\,(v_{q-1} + v_{q+1}),\qquad \alpha = \tfrac{k\gamma\sigma^2}{2},\quad \eta = A\Big(1 + \frac\gamma k\Big)^{-(1 + k/\gamma)},$$

i.e. $v(t) = e^{-M(T-t)}\mathbf 1$ for a tridiagonal $M$. Far from the horizon $v$ is proportional to
the ground-state eigenvector $f$ of $M$, giving stationary quotes
$\delta^b(q) = \tfrac1k\ln\tfrac{f(q)}{f(q+1)} + \tfrac1\gamma\ln(1 + \tfrac\gamma k)$.
A Gaussian approximation $f(q) \approx e^{-cq^2}$ gives the closed form used in practice:

$$\delta^b(q) \approx c_1 + \big(q + \tfrac12\big)c_2,\quad
\delta^a(q) \approx c_1 + \big(\tfrac12 - q\big)c_2,\qquad
c_1 = \tfrac1\gamma\ln\big(1 + \tfrac\gamma k\big),\quad
c_2 = \sqrt{\frac{\gamma\sigma^2}{2kA}\Big(1 + \frac\gamma k\Big)^{1 + k/\gamma}}.$$

*Intuition for $c_2$:* the skew grows with volatility and risk aversion and shrinks with liquidity
$A$, because a liquid market lets you work inventory off quickly. `glft_exact_eta` solves the exact
problem (eigenvalue from `eigvalsh`, eigenvector ratios by a backward continued fraction because the
tails underflow). The closed form is within about 5% at low risk aversion (checked in
`tests/test_strategies.py`) but off by 20-30% at $\gamma = 1$, where the Gaussian approximation of
$f$ breaks down.

## 3. From continuous quotes to a tick grid

On a large-tick book the optimal distance usually falls between two feasible prices. With half-spread
0.57 ticks, rounding bids down and asks up puts both quotes a full tick behind the touch and kills the
fill rate. The right move is to maximise the same Hamiltonian over the feasible ticks,

$$H(\delta) = Ae^{-k\delta}\big(1 - e^{-\xi(\delta - \eta)}\big)/\xi,$$

which is unimodal, so comparing the two neighbouring ticks is exact (`tick_quotes`; brute-force test
over the whole grid). Ablation: on 100 paired test episodes, tick-optimal quoting earns 73 ticks more per episode than floor/ceil rounding (95% CI 62 to 83).

## 4. Pricing a fill, not the mid (the main idea)

GLFT assumes the mid is a martingale independent of our fills. In reality a fill is not a random
draw: it happens exactly when a market order arrives at our price, and market orders cluster.
Glosten-Milgrom logic says to quote around the value *conditional on being filled*. So each
candidate price $p$ is valued with the expected markout $M(p, x)$ of a fill there:

$$H(p) = Ae^{-k\delta}\Big(1 - e^{-\xi(\delta + M(p,x) - \eta(q))}\Big)/\xi,\qquad \delta = |{\rm mid} - p|,$$

$$M = c_{\rm queue} + b_{\rm same}\,\widehat{N}_{\rm same} + b_{\rm opp}\,\widehat N_{\rm opp} + b_{\rm obi}\,{\rm OBI}_{\rm own}.$$

* $c_{\rm queue}$ depends on whether the order sits at the touch or behind it. A fill behind the
  touch means a market order swept the whole best level: strong momentum. In the training data a touch fill has a 1 s markout of about -0.25 ticks and a fill behind the touch about -1.06 ticks, so a behind-the-touch fill loses most of its extra tick of edge immediately.
* $\widehat N$ are Hawkes forecasts of market orders over the markout horizon, in excess of their
  stationary mean, hitting our side ("same") or the other side.
* OBI is the top-of-book imbalance oriented towards our side: a thin queue on our side is about to
  be run over.

The model is fitted by OLS on fills of probe quotes in training episodes, with errors clustered by
episode (fills within an episode are not independent). The markout horizon is the inventory holding
period by Little's law (mean |inventory| divided by fill rate, about one second).
When no candidate has $H > 0$ the quote is withdrawn. GLFT is the special case $M \equiv 0$
(tested). *Approximation to admit:* the markout is applied to the filled unit only; a fuller
treatment would also revalue existing inventory conditional on the fill.

## 5. Execution details that matter

* **Own orders are excluded from the strategy's view** of the book. Otherwise a quote that improves
  the mid changes the strategy's own fair value and the strategy chases itself (an early version of this
  simulator livelocked exactly this way).
* **Queue premium.** A resting order's queue position is an asset; moving it sends it to the back.
  The strategy keeps a resting order unless the new price's $H$ is better by a factor $(1+\kappa)$.
  This cut messages and raised PnL (+77 ticks per episode for GLFT and +20 for MK-full on paired test episodes, with 18% and 36% fewer orders sent). A principled alternative is to model queue
  position explicitly (Moallemi & Yuan 2017).
* **News and the latency race.** On public news the fair value jumps by $J$; an arbitrageur reacts
  after an Exponential(mean 5 ms) delay and takes every order on the wrong side of the new value.
  The market maker sees the news and cancels exposed quotes, but the cancel only counts if it
  arrives first: with latency $L$, $P(\text{win}) = e^{-L/5\,\rm ms}$ (Budish, Cramton & Shim 2015).

## 6. Evaluation and statistics

* **Three seed sets.** Training estimates everything, validation picks hyper-parameters, the test
  set is run once. 46 configurations were tried on validation.
* **Common random numbers.** All randomness is drawn before an episode starts, and every strategy
  sees the same draws. $\mathrm{Var}(X_A - X_B) = \mathrm{Var}X_A + \mathrm{Var}X_B - 2\mathrm{Cov}$,
  and the covariance is large: pairing removed 69-82% of the variance of the PnL differences between the model-based strategies in the test set.
* **Paired bootstrap** CIs and paired t-tests on per-episode differences, **Holm-Bonferroni** across
  comparisons (sort p-values, multiply the i-th smallest by $m - i + 1$, enforce monotonicity).
* **Newey-West** errors for regressions with overlapping targets: sampling an h-second change every
  0.25 s creates an MA(h/0.25 - 1) error structure; naive t-statistics were about twice too large at
  a 1-second horizon and up to five times at 30 seconds.
* **Sharpe** is reported per 30-minute episode and not annualised; episodes are i.i.d. draws from a
  stationary simulator, which real trading days are not, so annualising would overstate it.

## 7. Accounting identities

With wealth $W = \text{cash} + q\,m$:

$$W_T - W_0 = \underbrace{\sum_{\rm fills} x_i\,\text{edge}_i}_{\text{spread capture}} + \underbrace{\sum q\,\Delta m}_{\text{inventory PnL}} - \text{fees},\qquad \text{edge} = \text{side}\cdot(m - p).$$

Proof: at a buy of $x$ at $p$, cash falls by $px$ and inventory rises by $x$, so $\Delta W = x(m - p)$;
between fills $\Delta W = q\,\Delta m$. The simulator checks it to 1e-6 every episode in the tests.
**Markout** at horizon h: side $\cdot(m_{t+h} - m_t)$. **Realised spread** = edge + markout, the PnL
per unit if the position were closed at the mid h seconds later.
