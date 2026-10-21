### stylized

| Statistic | Value |
| :--- | ---: |
| Exogenous events per second | 23.4 |
| Market orders per second | 3.11 |
| Share of time the spread is 1 tick | 94.4% |
| Share of time the spread is 3+ ticks | 1.1% |
| Mean depth at best bid / ask (lots) | 12.8 / 12.8 |
| Volatility at 1 s / 300 s sampling (ticks per sqrt s) | 0.778 / 0.704 |
| Excess kurtosis of 1 s mid changes | 41.5 |
| Market-order sign autocorrelation, lags 1 / 10 | 0.222 / 0.025 |
| Coefficient of variation of market-order gaps (Poisson = 1) | 1.34 |

### hawkes_models

| Model | Parameters | Log-likelihood | BIC | Branching ratio | Decay rates (1/s) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Poisson | 4 | 961,763 | -1,923,470 | - | - |
| Hawkes, 1 exponential | 21 | 987,333 | -1,974,379 | 0.390 | 6.71 |
| Hawkes, 2 exponentials | 38 | 988,016 | -1,975,511 | 0.553 | 7.99, 0.354 |
| True data-generating process | 38 | 987,995 | -1,975,469 | 0.561 | 8, 0.35 |

899,444 training events. Lower BIC is better.

### gof

| Model | Market buys | Market sells | Limit buys | Limit sells |
| :--- | ---: | ---: | ---: | ---: |
| Poisson | 0.1577 (p <0.001) | 0.1651 (p <0.001) | 0.0315 (p <0.001) | 0.0326 (p <0.001) |
| Hawkes, 1 exponential | 0.0045 (p 0.930) | 0.0128 (p 0.012) | 0.0046 (p 0.033) | 0.0026 (p 0.509) |
| Hawkes, 2 exponentials | 0.0058 (p 0.707) | 0.0114 (p 0.037) | 0.0037 (p 0.137) | 0.0026 (p 0.545) |

Kolmogorov-Smirnov statistic of time-rescaled inter-arrival times against Exp(1), on five held-out episodes. Estimator check: on 40 data sets simulated from known parameters, nominal 95% intervals covered the truth 93.9% of the time (z-scores: mean -0.03, sd 1.02).

### calibration

| Quantity | Value |
| :--- | ---: |
| Volatility sigma (ticks per sqrt s) | 0.778 |
| Fill intensity A (fills per s per side) | 1.375 |
| Fill decay k (per tick) | 1.845 |
| Measured fill rate at the touch / 1 / 2 ticks behind (per s) | 0.5515 / 0.0765 / 0.0186 |
| Inventory holding period by Little's law | 1.22 s |
| Markout horizon used by the strategies | 1 s |

### markout_models

| Model | At the touch | Behind the touch | Hawkes: flow at our side | Hawkes: flow at other side | Imbalance (own side) | R-squared |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Queue position | -0.254 (-36.7) | -1.060 (-40.7) | - | - | - | 5.0% |
| + imbalance | -0.197 (-29.5) | -0.854 (-34.8) | - | - | +0.455 (+22.1) | 8.3% |
| + Hawkes flow | -0.163 (-24.7) | -0.793 (-34.2) | -0.257 (-5.8) | -0.007 (-0.2) | +0.417 (+18.5) | 9.2% |

OLS on 43,297 fills of probe quotes in training episodes; 1-second markouts in ticks; t-statistics clustered by episode in parentheses. Hawkes features are expected market orders over the horizon in excess of their stationary mean.

### validation

| Strategy | Selected gamma | Mean PnL | Std PnL | Sharpe |
| :--- | ---: | ---: | ---: | ---: |
| GLFT + markouts (queue, OBI, Hawkes) | 0.1 | 248.5 | 50.4 | 4.93 |
| GLFT + markouts (queue, OBI) | 0.01 | 316.2 | 66.6 | 4.75 |
| GLFT + markouts (queue) | 0.1 | 233.9 | 55.0 | 4.25 |
| GLFT | 0.1 | 178.7 | 77.3 | 2.31 |
| Avellaneda-Stoikov | 0.0003 | 173.4 | 88.3 | 1.96 |
| Join BBO | - | 253.8 | 186.1 | 1.36 |

60 validation episodes; queue premium kappa = 0.5 chosen first on GLFT; 46 configurations tried in total.

### test_compact

| Strategy | Sharpe per episode [95% CI] | Mean PnL [95% CI] | Std PnL | 5th pct PnL | Mean max drawdown | Mean abs inventory |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| GLFT + markouts (queue, OBI, Hawkes) | **4.64** [4.13, 5.26] | 246.0 [239.5, 252.6] | 53.1 | 150 | 40 | 0.60 |
| GLFT + markouts (queue, OBI) | 3.55 [3.19, 4.02] | 304.9 [294.3, 315.7] | 85.9 | 129 | 68 | 1.35 |
| GLFT + markouts (queue) | 3.91 [3.49, 4.47] | 233.0 [225.6, 240.5] | 59.6 | 131 | 49 | 0.70 |
| GLFT | 2.25 [1.99, 2.58] | 182.9 [172.8, 193.0] | 81.4 | 27 | 74 | 1.04 |
| Avellaneda-Stoikov | 1.52 [1.30, 1.79] | 170.5 [156.7, 184.5] | 112.3 | -10 | 95 | 1.66 |
| Join BBO | 1.34 [1.14, 1.57] | 276.8 [251.0, 302.6] | 206.9 | -57 | 164 | 5.27 |

### test_detail

| Strategy | Fills | Edge per lot | Markout 1 s | Realised spread 5 s | PnL per lot | Messages per fill | Lots sniped | Losing episodes |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GLFT + markouts (queue, OBI, Hawkes) | 1,030 | 0.632 | -0.333 | 0.253 | 0.239 | 8.81 | 42.9 | 0.0% |
| GLFT + markouts (queue, OBI) | 1,340 | 0.606 | -0.283 | 0.265 | 0.228 | 4.39 | 47.1 | 0.0% |
| GLFT + markouts (queue) | 1,280 | 0.588 | -0.330 | 0.200 | 0.183 | 4.35 | 39.6 | 0.0% |
| GLFT | 1,515 | 0.579 | -0.349 | 0.153 | 0.121 | 2.83 | 45.7 | 2.4% |
| Avellaneda-Stoikov | 1,589 | 0.565 | -0.335 | 0.157 | 0.108 | 2.79 | 47.3 | 6.4% |
| Join BBO | 1,766 | 0.534 | -0.264 | 0.211 | 0.156 | 2.07 | 52.4 | 7.6% |

Per-lot quantities in ticks. PnL per lot is also the break-even maker fee.

### paired_glft

| Strategy | PnL difference [95% CI] | t | p (Holm) | Episodes won | Variance removed by pairing |
| :--- | ---: | ---: | ---: | ---: | ---: |
| GLFT + markouts (queue, OBI, Hawkes) | +63.1 [+56.5, +69.8] | +18.5 | <0.001 | 90% | 69% |
| GLFT + markouts (queue, OBI) | +122.0 [+115.1, +128.9] | +34.6 | <0.001 | 98% | 78% |
| GLFT + markouts (queue) | +50.1 [+44.8, +55.4] | +18.7 | <0.001 | 90% | 82% |
| Avellaneda-Stoikov | -12.4 [-21.4, -3.0] | -2.6 | 0.009 | 45% | 71% |
| Join BBO | +93.8 [+69.7, +118.2] | +7.6 | <0.001 | 70% | 23% |

### paired_join

| Strategy | PnL difference [95% CI] | t | p (Holm) | Episodes won | Variance removed by pairing |
| :--- | ---: | ---: | ---: | ---: | ---: |
| GLFT + markouts (queue, OBI, Hawkes) | -30.7 [-55.2, -6.2] | -2.5 | 0.028 | 40% | 16% |
| GLFT + markouts (queue, OBI) | +28.1 [+5.4, +51.0] | +2.4 | 0.028 | 55% | 34% |
| GLFT + markouts (queue) | -43.7 [-68.1, -19.2] | -3.5 | 0.002 | 40% | 17% |
| GLFT | -93.8 [-118.2, -69.7] | -7.6 | <0.001 | 30% | 23% |
| Avellaneda-Stoikov | -106.2 [-127.8, -84.7] | -9.7 | <0.001 | 24% | 46% |

### attribution

| Strategy | Spread captured | Inventory PnL | Closing cost | Net PnL |
| :--- | ---: | ---: | ---: | ---: |
| GLFT + markouts (queue, OBI, Hawkes) | 651.5 | -405.1 | -0.3 | 246.0 |
| GLFT + markouts (queue, OBI) | 812.2 | -506.5 | -0.8 | 304.9 |
| GLFT + markouts (queue) | 752.6 | -519.2 | -0.4 | 233.0 |
| GLFT | 878.1 | -694.5 | -0.6 | 182.9 |
| Avellaneda-Stoikov | 897.7 | -723.6 | -3.6 | 170.5 |
| Join BBO | 942.7 | -662.4 | -3.5 | 276.8 |

### ladder

| Model (gamma = 0.1) | Sharpe [95% CI] | Sharpe gain | Mean PnL [95% CI] | PnL gain | Fills | Markout 1 s | Mean abs inventory |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GLFT | 2.25 [1.99, 2.58] | - | 182.9 [172.8, 193.0] | - | 1,515 | -0.349 | 1.04 |
| + queue position | 3.91 [3.49, 4.47] | +1.66 [+1.37, +2.02] | 233.0 [225.6, 240.5] | +50.1 [+44.8, +55.4] | 1,280 | -0.330 | 0.70 |
| + book imbalance | 4.01 [3.60, 4.53] | +0.10 [-0.22, +0.40] | 244.5 [236.9, 252.1] | +11.5 [+7.3, +15.7] | 1,115 | -0.336 | 0.69 |
| + Hawkes flow | 4.64 [4.13, 5.26] | +0.63 [+0.32, +0.98] | 246.0 [239.5, 252.6] | +1.5 [-2.0, +5.2] | 1,030 | -0.333 | 0.60 |

All rungs on the same 250 test episodes, same risk aversion and queue premium; gains are paired differences with bootstrap CIs.

### fill_quality

| Strategy | Fill type | Share of volume | Edge per lot | Markout 5 s | Realised spread 5 s | Realised spread per episode |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| GLFT + markouts (queue, OBI, Hawkes) | improving the spread | 5.4% | 0.937 | -0.408 | 0.530 | 29.3 |
|  | at the touch | 85.6% | 0.555 | -0.265 | 0.290 | 253.8 |
|  | behind the touch | 4.8% | 1.694 | -1.904 | -0.210 | -10.4 |
|  | taken by the news arbitrageur | 4.2% | 0.591 | -0.949 | -0.357 | -15.1 |
| GLFT | improving the spread | 6.2% | 0.688 | -0.534 | 0.153 | 14.3 |
|  | at the touch | 87.2% | 0.534 | -0.332 | 0.201 | 264.6 |
|  | behind the touch | 3.5% | 1.517 | -2.172 | -0.655 | -35.0 |
|  | taken by the news arbitrageur | 3.0% | 0.597 | -0.973 | -0.375 | -17.2 |
| Join BBO | improving the spread | 0.2% | 0.702 | -0.889 | -0.187 | -0.7 |
|  | at the touch | 96.6% | 0.531 | -0.304 | 0.227 | 384.3 |
|  | behind the touch | 0.3% | 1.892 | -1.535 | 0.357 | 1.6 |
|  | taken by the news arbitrageur | 3.0% | 0.513 | -0.795 | -0.282 | -14.8 |

First 100 test episodes. Position is measured against the market's best price on our side just before the trade, excluding our own orders. Per-lot quantities in ticks.

### ablations

| Comparison | PnL gain per episode [95% CI] | t | p | Episodes won |
| :--- | ---: | ---: | ---: | ---: |
| Markout model (MK-full) vs plain GLFT | +68.7 [+58.5, +79.1] | +12.9 | <0.001 | 89% |
| Imbalance priced per fill vs used as an unconditional alpha | +65.0 [+53.1, +76.6] | +10.7 | <0.001 | 88% |
| Tick-optimal quoting vs floor/ceil rounding (GLFT) | +72.9 [+62.5, +83.2] | +13.6 | <0.001 | 91% |
| Queue premium vs none (GLFT) | +77.4 [+68.1, +86.9] | +16.2 | <0.001 | 95% |
| Queue premium vs none (MK-full) | +20.2 [+15.3, +25.2] | +7.9 | <0.001 | 76% |
| Pulling quotes on news vs ignoring news (MK-full) | +7.8 [+4.2, +11.4] | +4.3 | <0.001 | 68% |
| Exact GLFT solution vs closed form | +2.2 [+0.7, +4.0] | +2.7 | 0.008 | 29% |
| Markout horizon 2 s vs 1 s (MK-full) | -9.4 [-14.5, -4.3] | -3.6 | <0.001 | 35% |

Paired on the first 100 test episodes.

### latency

| Order-entry latency | GLFT + markouts (queue, OBI, Hawkes) | GLFT | Join BBO |
| :--- | ---: | ---: | ---: |
| 0.5 ms | 255.0 (6) | 182.2 (6) | 267.8 (7) |
| 1 ms | 253.5 (12) | 180.9 (13) | 267.9 (15) |
| 2 ms | 248.8 (22) | 178.3 (24) | 261.2 (27) |
| 5 ms | 241.9 (43) | 173.2 (46) | 253.5 (52) |
| 10 ms | 235.9 (58) | 173.4 (62) | 238.9 (71) |
| 20 ms | 229.5 (66) | 184.8 (70) | 238.1 (80) |
| 50 ms | 223.1 (66) | 209.3 (70) | 236.8 (80) |
| 100 ms | 219.5 (65) | 221.7 (69) | 239.6 (79) |

Mean PnL per episode; in parentheses, lots per episode taken by the news arbitrageur, whose reaction time is Exponential with mean 5 ms. No recalibration across latencies.

### stress

| Market | GLFT + markouts (queue, OBI, Hawkes) | GLFT | Join BBO |
| :--- | ---: | ---: | ---: |
| Base market (branching ratio 0.56) | 4.64 (241.9) | 2.15 (173.2) | 1.29 (253.5) |
| Toxic flow (branching ratio 0.68) | 0.22 (39.9) | -1.80 (-508.2) | -0.79 (-326.9) |
| Thin book (branching ratio 0.56) | 0.32 (38.5) | -1.07 (-176.7) | 0.05 (14.3) |

Sharpe per episode (mean PnL). Strategies keep their base-market calibration. Toxic flow: market-order self-excitation up by a third. Thin book: 25% fewer limit orders.
