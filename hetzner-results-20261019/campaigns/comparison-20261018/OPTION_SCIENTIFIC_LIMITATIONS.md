# Scope of the option entropic scores

The option experiment reports finite-sample empirical log-mean-exp scores. Its bootstrap intervals are conditional comparisons of those empirical scores for the fitted policies. They are not guarantees of finite population entropic risk.

The stock has an unbounded lognormal final-interval return. Given the information before that interval, the causal stock holding h lies in [0,1]. On the large-stock-price tail, the short call payoff, stock hedge and proportional final liquidation cost leave the terminal-loss coefficient

`1 - h + c*h >= c > 0`,

where c is the transaction-cost rate. All earlier cashflows are fixed when conditioning on that history. A positive coefficient of a lognormal stock price has an infinite positive exponential moment. Thus this frozen daily model with positive liquidation costs does not provide a finite untruncated population entropic objective. A finite test set produces a finite empirical score despite that limitation.

The reported CVaR, terminal dispersion, costs and marked-error diagnostics remain separate finite-sample quantities. The option comparison must be described as an empirical benchmark of the training methods, not a population entropic-risk guarantee. There is no outcome-driven tail cap or model redesign. The credit benchmark has bounded losses and does not inherit this particular issue.
