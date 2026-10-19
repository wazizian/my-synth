# Two-speed volatility diagnostic

Run from the repository root:

```sh
MPLCONFIGDIR=/tmp/mpl-volatility python -m volatility_dynamics.demo
```

Requires the project's JAX, Chex and NumPy dependencies plus Matplotlib. Outputs are in `results/`: RV/IV PNG and PDF, an example matched underlying-price plot, all simulated arrays and `summary.json`. This is local CPU work, with one seed (20261017), 128 physical paths and 256 conditional pricing paths.

## Dynamics

Two independent log-variance factors follow OU dynamics between shocks. Under either measure M:

    dX_i = -kappa_i^M X_i dt + eta dB_i^M
    v = 0.20^2 exp(X_1 + X_2)
    dS/S = sqrt(v) dW^M

Stock and factor Brownian motions are independent; interest rates are zero. `eta=0.35` per square-root year. Factors use exact daily OU transitions. Daily stock log returns use the variance at the start of the interval, defining a discrete-time stochastic-volatility model. Quotes refer to that same start-of-day state. A single imposed event at day 35 adds log(2) to each factor in base/hard, doubling instantaneous volatility from roughly 20% to 40%.

Half-lives in trading days:

| Regime | Physical X1 / X2 | Pricing X1 / X2 | Shock |
|---|---|---|---|
| Easy | 3 / 3 | 3 / 3 | None |
| Base | 3 / 3 | 40 / 40 | Same positive shock |
| Hard | 40 / 3 | 1 / 40 | Same positive shock |

The different P/Q drifts encode volatility risk premia. In base, the pricing distribution expects the currently elevated factors to persist, so IV responds strongly; both physical factors actually normalize quickly. In hard, the physically persistent factor has a short pricing half-life, while the temporary factor contributes most of the IV spike. Its physical decay brings IV down despite remaining high physical variance. These are deliberately constructed parameters, not an empirical calibration. P and Q differ only in factor drifts between events, retaining the same diffusion coefficients.

## What is plotted

RV at date t is the annualized square root of the mean squared underlying log returns over the five preceding days, without subtracting the estimated mean. Each path's RV is therefore observable from its own price history. The orange line pools the pathwise RVs by RMS; shading is the cross-path 10–90% range, not a confidence interval. Gray is latent instantaneous volatility, shown solely for diagnosis.

IV is actual Black–Scholes implied volatility for a rolling 21-trading-day ATM call. Conditional on a Q factor trajectory, the normalized call is `2 Phi(sqrt(integrated_variance)/2)-1`; averaging over Q trajectories prices the call. Bisection recovers its BS IV. The blue line is the mean across physical states. Neither IV nor latent factors would be hedger inputs. Conditional Q draws are shared across dates and scenarios for numerical stability. Finite-Q-sample pricing error is not quantified by the orange shading.

## Checks and scope

The script checks finite positive IV, BS price reconstruction to 1e-12, base RV/IV decay, and persistent hard RV above both hard IV and base RV at predetermined dates. These demonstrate the intended dynamics, not specialist superiority or difficulty under an optimized hedge.

The imposed shock is an external event-study intervention: pricing conditions on current state and no further interventions. Its fixed date is a plotting alignment device, not a calendar event anticipated by the pricer. Consequently this is NOT yet a complete arbitrage-consistent pricing model across shock arrivals. A benchmark must specify stochastic arrivals and their P/Q laws, along with a hedging payoff/horizon and common observation interface. No training or cloud jobs have been launched.
