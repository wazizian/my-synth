# Next experiment: hidden IV / observable price history

User-selected design:
- Easy: no volatility shocks.
- Base: fast shock, IV and RV normalize quickly together (allow the mechanical lag in rolling RV).
- Hard: a persistent physical-volatility shock; IV normalizes while RV remains elevated.
- Hedger inputs: underlying price history, time and its own positions; no option quotes, IV, latent variance/factors, regime labels or pricing-measure parameters.
- Same recurrent hedger architecture and information across methods. One shared training seed, no multi-seed. Entropic training; entropic risk, CVaR95 and SD of daily hedging errors as metrics.

The uploaded demo reproduces the requested RV/IV picture using two OU log-variance factors with different physical/pricing mean-reversion speeds. README.md explains the exact equations, parameters, timing and limitations. results/ contains diagnostic figures/summary, not training evidence.

Suggested execution:
1. Audit the source and reproduce the diagnostic on an approved worker, following the existing Hetzner control-plane, budget and shutdown rules. Do not install/run numerical jobs on the control plane. This handoff does not establish a new spending cap.
2. Replace the imposed event-study intervention by a coherent stochastic shock-arrival model under P and Q. Avoid a publicly scheduled event ignored by the option pricer. Freeze arrivals, payoff, observation timing, maturity, cost/position constraints and train/evaluation distributions before looking at specialist results. Account for future Q shocks in prices; check numerical convergence and martingale/ledger consistency.
3. Preserve the intended easy/base/hard pattern in ensemble plots of observable rolling RV and actual option IV. Expose only causal price observations to the agent. Quantify quote Monte Carlo error and check multiple option maturities. Plotting-time labels and latent states must not enter features.
4. Check the economic premise before large training. A Q-versus-P volatility mismatch alone does NOT guarantee specialist gains. With only stock trading, a terminal payoff and an entropic terminal objective, intermediate option marks cancel from self-financing terminal P&L, and a fixed initial premium only shifts the entropic loss: changing Q alone need not change the optimal hedge. Different P persistence can matter; show this rather than attributing an effect automatically to hidden IV. Decide explicitly whether the task is terminal replication or management of marked-to-market risk. Do not add tradable options or change the objective silently.
5. Run the bounded baseline/easy-specialist/hard-specialist screen first, one training seed, identical architecture and matched evaluation paths. Treat baseline underperformance on both targets and hard residual-risk ordering as hypotheses, not guaranteed properties. Freeze economically meaningful margins and adequate evaluation precision. Report failure honestly; no large adaptive campaign if the screen fails.
6. If the gate passes, follow the existing comparison plan for later methods within authorized budget. Retain source/config hashes, checkpoints, training loss/evaluation curves, raw results and reports; verify exports and shut down workers.

Keep prior credit/equity campaign evidence unchanged. This upload launches no jobs and sends no instruction directly to another agent.
