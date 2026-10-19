"""Assemble the negative-outcome record from frozen worker evidence."""
import datetime,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
RUN=Path('/home/jump/vast-tools/runs/20261018T090952Z-5439b23c');O=RUN/'outputs'
read=lambda p:json.loads(p.read_text())
design=read(P/'design.json');summary=read(O/'gate_summary.json');art=read(P/'artifact_audit.json');num=read(P/'numeric_audit_final.json');billing=read(P/'provider_billing.json')
assert summary['accepted'] is None and len(summary['results'])==2
assert art['passed'] and art['sync_status']['final_sync_complete']
assert not billing['active_campaign_instances'] and not billing['unreported_instance_ids']
assert num['dataset_count']==90 and num['check_count']==1350
rows=[];texrows=[]
for c in summary['results']:
    g=read(O/c['candidate']/'screen/gate.json')
    for d in ('easy','hard'):
        v=g['paired_intervals'][d]
        rows.append(f"| {c['candidate']} | {d} | {v['improvement']:.7f} | [{v['low']:.7f}, {v['high']:.7f}] | {v['minimum']:.4f} | {'Pass' if g['checks'][d+'_specialization'] else 'Fail'} |")
        texrows.append(f"{c['candidate']} & {d} & {v['improvement']:.7f} & [{v['low']:.7f}, {v['high']:.7f}] & {v['minimum']:.4f} "+chr(92)*2)
report=f'''# Entropic benchmark gate: no accepted benchmark

Neither of the two predeclared candidates passed the required baseline-to-specialist improvement on both target domains. The bounded development gate is complete with a negative outcome. Following PLAN.md §3, the campaign stopped before adaptive training. No benchmark was frozen as accepted, no confirmation set was opened, and no final-test set was opened. This is not a completed eight-arm adaptive comparison or evidence of adaptive superiority.

## Results

A positive improvement is baseline entropic risk minus specialist entropic risk on paired independent screening episodes. Acceptance required the lower endpoint of the paired 95% episode-bootstrap interval to exceed the fixed minimum. Each domain used 8,192 screening episodes and 400 resamples. These intervals are conditional on policies trained with the single shared seed 20261017; they do not measure variation across training seeds.

| Candidate | Target | Improvement | Conditional 95% interval | Required minimum | Result |
|---|---|---:|---:|---:|---|
'''+ '\n'.join(rows)+f'''

Both candidates failed on easy. The first also failed the hard confidence-bound requirement. The second cleared the hard requirement alone. Thresholds and the two-candidate menu were recorded before running; neither was changed in response to outcomes.

## Difficulty interpretation

Both raw screens showed higher hard-domain pre-premium entropic risk, CVaR and daily-error SD. Those comparisons are insufficient to establish the intended residual difficulty ordering. Independent supplementary diagnostics found lower terminal-loss SD, mean-centered entropic risk and mean-centered CVaR for the hard specialist than for the easy specialist in BOTH candidates. For example, first-candidate mean-centered CVaR was 0.030787 on easy versus 0.021871 on hard. The higher raw hard loss partly reflects a more expensive liability. These diagnostics reinforce rejection; they did not change the predeclared screening decision or trigger extra candidates. Daily-error dispersion alone does not resolve the conflicting terminal dispersion evidence.

See [independent numerical and residual audit](numeric_audit_final.json). The finite-menu family is a martingale trinomial model with low-intensity stable easy paths and hidden persistent increases in move intensity on hard paths. It is a discrete research model, not a calibrated Heston market. No accepted benchmark or general claim about market difficulty follows.

## Protocol and retained evidence

- Two candidates; baseline, easy_reference and hard_reference for each. Identical GRU(6,32), scalar bounded stock position, initial weights, Adam learning rate 0.001, batch 256, 8 updates/block, and gradient clipping 1. One training seed; independent namespaces for training, selection and screening.
- Stock-only hedging; 32 observation-context dates and 20 trading intervals; proportional cost 0.001. Hidden switches occur at trading intervals 6–12, leaving 8–14 intervals after the change. Baseline has strictly positive mass on all nine scenarios. Policy inputs exclude hidden labels and future switch times.
- Exact common-P0 Bayesian trinomial liability marks use only realized observations. Target populations are conditional scenario subsets of that common model. Gamma=10, pre-premium normalized terminal loss; CVaR alpha=0.95 with fractional tail mass. Initial costs enter interval zero, liquidation costs the last interval.
- Each diagnostic arm received 180 active simulation/update seconds and checkpoint opportunities at 0, 60, 120 and 180 seconds. Actual block overruns were below one second. Domain-specific selection used base/easy/hard selection risk respectively. All initialization hashes and paired dataset hashes agree where required.
- Retained: 24 learner/optimizer/PRNG checkpoints plus two validation-resume snapshots, 90 raw evaluation array sets, selection and screening metrics on all three domains, training logs, 36 figures, environment locks, source and configuration hashes. No teacher histories exist because teacher training was never entered.
- Finite-minibatch log-mean-exp is an estimator, not exact population entropic risk. Exponential weight concentration diagnostics accompany every evaluation. CPU orchestration performed no numerical experiment; simulation, training, metrics, figures and independent numerical reconstruction ran on the GPU worker host.

## Validation and limitations

Pretraining GPU checks passed for martingale construction, exact pricing recursion, cash translation, fractional CVaR, causal actions, finite gradients and exact checkpoint resume. Independent NumPy reconstruction on the GPU worker later covered all 90 evaluation datasets (1,350 checks). Independent Bayesian marks agree within 4.45e-16; metric implementations agree within 2.10e-16 for entropic risk and 5.56e-17 for CVaR.

The full numerical audit is **not an all-pass result**: seven dataset-level accounting checks exceeded the 1e-7 tolerance. Maximum loss reconstruction error was 1.240e-7 and telescoping error 1.302e-7. All failures are retained in the audit; the tolerance was not loosened after this audit. These discrepancies are much smaller than the prespecified 0.0002/0.0005 useful-gain scales, but numerical certification for a future accepted experiment remains incomplete. The preliminary 256-path check had missed the larger discrepancies. An earlier worker attempt failed its stricter 1e-10 check before any diagnostic training and was exported and destroyed.

Compilation, evaluation and checkpoint I/O were excluded from active training. Per-arm post-warmup wall time is recorded, but their individual compile/evaluation/I/O durations were not separately instrumented; this is a telemetry limitation, not evidence those costs were zero. There were no adaptive teacher feedback costs. The initial and later failed numerical checks are preserved, not relabeled as successful.

Plots with easy-domain event alignment have no switch events by construction; their empty traces do not represent zero event risk. Selection curves are development curves, not final-test curves. No confirmation was warranted after either failed screen.

## Artifacts

- [Predeclared design](design.json), [campaign ledger](ledger.json), [submitted worker source](gate_packaged.py).
- [Worker report]({O}/report.md), [worker LaTeX tables]({O}/report.tex), [all figures]({O}/figures).
- [Authoritative raw output directory]({O}), [worker artifact hashes]({O}/artifact_hashes.json), [launcher inventory]({RUN}/artifacts.json).
- [Artifact/protocol audit](artifact_audit.json): 1,165 checks passed, final export complete, zero transfer failures. This metadata/hash pass is distinct from the numerical audit failures above.
- [Numerical audit](numeric_audit_final.json), [provider billing and shutdown evidence](provider_billing.json), [requirement audit](completion_audit.json).
- [LaTeX final assessment](report.tex).

## Closure

Both paid instances, 56528797 and 56529076, are confirmed absent from active provider state. All known instances have billing records. Provider-reported total is **${billing['reported_usd']:.3f}**, rounded and subject to later settlement, within the separate $5 diagnostic and $25 overall ceilings. No further rentals were launched. The prior pilot's existing files were preserved; its earlier user-authorized cleanup had already deleted datasets/checkpoints, so they could not be preserved retroactively.

The required next step is a new benchmark design, with a new bounded development plan, if further research is requested. Automatically expanding the candidate list or lowering the thresholds would violate this campaign's stopping rule. Entropic teacher implementation, accepted-benchmark freeze, eight-arm adaptive training and final-test evaluation were not pursued after the mandatory negative gate.
'''
(P/'report.md').write_text(report)
tex=[r'\documentclass{article}',r'\usepackage[margin=1.5cm]{geometry}',r'\begin{document}',r'\section*{Entropic benchmark gate: negative result}',
'Neither predeclared candidate passed both target specialization requirements. As required by the bounded development plan, adaptive training stopped before an eight-arm comparison. One shared training seed; intervals below are conditional paired episode-bootstrap intervals.',r'\begin{center}',r'\begin{tabular}{lllll}', r'Candidate & Domain & Gain & 95\% interval & Minimum '+chr(92)*2]+texrows+[r'\end{tabular}',r'\end{center}',r'\section*{Scientific limitations}',
'Although raw pre-premium risk and daily-error dispersion were higher on hard, both candidates had lower hard-specialist terminal standard deviation and mean-centered tail risk. Higher liability value is not sufficient evidence of greater residual hedging difficulty.',
'Independent worker-side reconstruction covered 90 datasets and 1,350 checks. Seven accounting checks exceeded the fixed $10^{-7}$ tolerance; maximum discrepancy was $1.302\times10^{-7}$. Those failures remain recorded. Independent Bayesian prices agree within $4.45\times10^{-16}$. Metadata and artifact integrity passed 1,165 checks. Compilation, evaluation and I/O durations were not separately instrumented.',r'\section*{Artifacts and closure}',
'Raw episode arrays, daily errors, positions, costs, checkpoint states, dataset/source hashes, all-run metrics and 36 figures are retained. No confirmation or final-test data were opened. All GPU instances were exported and destroyed. Provider-reported cost: USD '+f"{billing['reported_usd']:.3f}"+', subject to settlement. See the companion Markdown report and JSON audits for exact artifact paths, checks and limitations.',r'\end{document}']
(P/'report.tex').write_text('\n'.join(tex)+'\n')
requirements=[
 ('Separate campaign and preserve existing pilot','verified','New campaign namespace; old results untouched. Prior cleanup losses explicitly documented.'),
 ('Bounded candidates and thresholds before outcomes','verified','design.json hash matches ledger and worker source; exactly two candidates.'),
 ('Common causal model, positive baseline support, stock-only pricing','verified','Submitted source, distribution validation, independent Bayesian reconstruction.'),
 ('Entropic learner and all three metrics','verified','Worker source and 90 independent array reconstructions; numerical accounting exceptions separately retained.'),
 ('Pretraining validation and resume','verified with later limitation','Both preliminary validations passed; later full-array numerical audit found seven accounting tolerance failures.'),
 ('One seed, identical architecture/initialization/settings/runtime/checkpoints','verified','Initialization/config/source hashes and artifact_audit.json.'),
 ('Baseline and both specialists evaluated on three development domains','verified','Six diagnostic lineages, four checkpoints each, 90 evaluation datasets.'),
 ('Meaningful gains on BOTH targets and credible residual difficulty','not established','Both screens rejected; supplemental residual-risk ordering also contradicts simple easy/hard labels.'),
 ('Fresh confirmation for screened candidate','not applicable','Neither candidate passed screening; confirmation not opened.'),
 ('Stop and report if bounded gate fails','verified','No accepted benchmark; report.md/report.tex; no adaptive jobs launched.'),
 ('Freeze accepted benchmark, gamma/scales/datasets and run eight arms','not entered under mandatory stop','Requires accepted gate; not falsely claimed complete.'),
 ('Entropic DPG/SOAR/selfplay teacher adaptation and symmetric tuning','not entered under mandatory stop','No accepted benchmark; no teacher implementation/training claimed.'),
 ('Final selection freeze and final-test evaluation','not entered under mandatory stop','No final data opened, generated or used for development.'),
 ('Plots/tables/LaTeX, raw outcomes and resumable checkpoint provenance','verified for diagnostic scope','36 figures, six diagnostic arm lineages, 24 checkpoints, 90 raw arrays; source/environment hashes retained.'),
 ('Conditional paired uncertainty and no across-seed superiority claims','verified','400 paired episode resamples; explicitly conditional single-seed evidence.'),
 ('Cost, timing, export and shutdown','verified with telemetry limitation','Provider $0.272, no active/unreported instances; final sync complete; individual compile/eval/I/O durations unavailable.'),
]
completion=dict(outcome='completed mandatory negative-gate stopping branch',scientific_success=False,accepted_benchmark=None,adaptive_campaign_executed=False,requirements=[dict(requirement=r,status=s,evidence=e) for r,s,e in requirements],numerical_audit_passed=False,artifact_audit_passed=True,mandatory_stop_satisfied=True,remaining_work_in_this_bounded_campaign=False,future_work='A newly authorized bounded benchmark design, not an expansion of the exhausted candidate menu.')
(P/'completion_audit.json').write_text(json.dumps(completion,indent=2))
l=read(P/'ledger.json');l['current_job'].update(status='finished; exported; destroyed; no candidate accepted',finished_verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat());l['reservations'][0]['status']='closed';l['provider_reported_usd']=billing['reported_usd'];l['provider_billing_scope']=billing['scope'];l['closure_outcome']=completion['outcome'];(P/'ledger.json').write_text(json.dumps(l,indent=2))
(P/'STATE.md').write_text('# Closed: mandatory negative benchmark gate\n\nBoth predeclared candidates failed screening. No adaptive training or final testing was run. All workers exported and destroyed, verified through provider API. Provider-reported cost $0.272, subject to settlement.\n\nRead report.md, completion_audit.json, artifact_audit.json, numeric_audit_final.json and provider_billing.json. Artifact/protocol audit passed; full numerical audit retained seven small accounting tolerance failures. Do not describe numerical validation as an all-pass result. No benchmark is accepted.\n\nPlan §3 requires stopping after the bounded candidates fail. Do not restart or expand this campaign automatically. Existing prior-pilot files remain unchanged. Raw diagnostic artifacts are under /home/jump/vast-tools/runs/20261018T090952Z-5439b23c/outputs.\n')
print(json.dumps(dict(outcome=completion['outcome'],report=str(P/'report.md'),reported_usd=billing['reported_usd'])))
