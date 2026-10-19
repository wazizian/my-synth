# Specialist versus adaptive teachers

Authorized 2026-10-18 21:05 UTC. Deadline 2026-10-19 06:05 UTC.

User requests the best credit campaign and, if specialists help, the price-history-only option campaign, comparing baseline and learned specialists against DPG-hard, SOAR-hard, self-play, DPG-mixed and SOAR-mixed. One shared initialization, no seed repeats. GPUs authorized as needed; numerical work and tests exclusively on workers.

Credit candidate selection uses disjoint development screen and confirmation, retaining the frozen .001 face useful-gain threshold and residual-risk ordering. Option retains its frozen .0005 spot threshold. Do not select using final test outcomes. Preserve negative results. The latest user goal requests the best credit comparison unconditionally and explicitly conditions the option comparison on specialists helping. Therefore a failed credit gate must be labeled exploratory and does not cancel the requested credit teacher comparison; a failed option gate stops the option comparison. An optional clarification was presented; absent steering, follow this reading of the latest goal.

Select among the original two credit campaigns and three costs using development only: passing screen plus confirmation first; rank otherwise by the smaller of the two confirmation specialist gain lower bounds, with mean confirmation point gain as a tie breaker. Freeze campaign, cost and positive baseline development scales before teacher training. This ranking does not make a failed gate pass.

The baseline remains pure ordinary/base, with no easy/hard exposure. Adaptive teachers may sample a declared equal-prior easy/base/hard menu; report that coverage changes as well as allocation. All policy observations, costs, model architecture, initialization, loss and contract rules remain shared within each benchmark.

Adaptive methods retain the original persistent learner/optimizer algorithms. DPG differentiates the actual nonlinear minibatch entropic objective through the learner block. SOAR rewards independent target improvement. Self-play receives no target feedback and uses its historical learner with actual Adam preconditioning. Mixed targets are balanced entropic risks with positive frozen development scales. Development selection for all adaptive arms is the same balanced criterion; report base retention separately.

GPU execution must be detached with closed stdin, redirected output, durable progress/exit files, complete atomic resume checkpoints and bounded runtime. Verify identity and remote state before replacing a worker. Incremental bounded export, artifact hashes and API-confirmed cleanup required. Hetzner is a one-vCPU control plane with limited disk: no local numerical work, tests or project-import checks; only Ruff/py_compile on changed files.

Agents: credit owns credit completion/selection; option owns volatility completion/gate; algorithms owns shared teacher implementation; root owns integration, execution freeze, validation, final comparison/report and cleanup audit.
