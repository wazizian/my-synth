"""Assemble authoritative closure from already-computed results; no local numerics."""
from pathlib import Path
import json,datetime,hashlib,sys
ROOT=Path(__file__).resolve().parent;RUN=Path(sys.argv[1]);OUT=RUN/'outputs'
def read(p):return json.loads(p.read_text())
def write(p,data):p.write_text(json.dumps(data,indent=2)+'\n')
summary=read(OUT/'gate_summary.json');audit=read(ROOT/'artifact_audit.json');numeric=read(OUT/'independent_audit.json');billing=read(ROOT/'provider_billing.json')
assert summary['complete'] and audit['passed'] and numeric['passed'] and numeric['complete']
assert not billing['active_campaign_instances'] and not billing['unreported_instance_ids']
accepted=summary['accepted'];final=read(OUT/'final_summary.json') if accepted else None
now=datetime.datetime.now(datetime.timezone.utc);ledger=read(ROOT/'ledger.json');elapsed=(now-datetime.datetime.fromisoformat(ledger['started_at'])).total_seconds()
lines=['# BS/SLV campaign final assessment','']
if accepted:
    conclusion=f'Family {accepted} passed development screening and fresh confirmation; all eight arms completed the frozen comparison. This does not by itself establish adaptive superiority.'
else:
    conclusion='Neither of the two frozen benchmark families passed the full development gate. The plan requires stopping before adaptive training and final-test evaluation. This is an uninformative benchmark for the proposed adaptive comparison.'
lines += [conclusion,'','One shared training seed: 20261017. Evaluation intervals condition on those trained policies; they do not measure training-seed uncertainty.','',f'Provider-reported total cost: ${billing["reported_usd"]:.3f}; campaign elapsed time: {elapsed/60:.1f} minutes. All campaign instances are confirmed shut down. Provider charges are rounded and may settle later.','','## Development results','','| Family | Target | Baseline minus specialist entropic risk | 95% interval | Required lower bound |','|---|---|---:|---|---:|']
texrows=[]
for entry in summary['results']:
    family=entry['family'];gate=read(OUT/family/'screen'/'gate.json')
    for domain in ['easy','hard']:
        x=gate['intervals'][domain];lines.append(f'| {family} | {domain} | {x["improvement"]:.8f} | [{x["low"]:.8f}, {x["high"]:.8f}] | {x["minimum"]:.5f} |')
        texrows.append(f'{family} & {domain} & {x["improvement"]:.8f} & [{x["low"]:.8f}, {x["high"]:.8f}] & {x["minimum"]:.5f} '+chr(92)*2)
lines += ['','Primary gates and the supplementary marked-residual review are retained separately. The latter checks X minus V0 so differing initial option values cannot establish difficulty by themselves. Its conservative numerical margins were declared before scientific training.','']
for entry in summary['results']:
    family=entry['family'];lines += [f'### Family {family}','',f'Overall screen passed: {entry["screen_passed"]}. Fresh confirmation passed: {entry.get("confirmation_passed","not opened")}.','']
    for phase in ['screen','confirmation']:
        p=OUT/family/phase/'gate.json'
        if not p.exists():continue
        gate=read(p);review=read(p.with_name('residual_review.json'))
        failed=[name for name,passed in gate['checks'].items() if not passed]
        lines += [f'{phase}: primary failures: {", ".join(failed) or "none"}; material marked-residual contradiction: {review["material_contradiction"]}.','',f'Initial-value SD: easy {review["easy_v0_sd"]:.8f}, hard {review["hard_v0_sd"]:.8f}. Marked terminal SD: easy {review["easy_marked"][0]:.8f}, hard {review["hard_marked"][0]:.8f}.','']
lines += ['## Numerical evidence and limits','','All final-run numerical, independent pricing/filter, and objective-gradient checks passed before the first diagnostic arm trained. Independent reconstruction of exported accounting and metrics passed at the fixed1e-10 accounting tolerance. See independent_audit.json for exact coverage and maxima.','',
'The initial coarse price grid failed its frozen tolerance. Production resolution was increased to192 variance nodes,1601 log-spot nodes and64 Gauss-Hermite nodes; reference resolution doubles each dimension. Market parameters and usefulness thresholds were not changed.','',
'A NumPy reference-checker shape error was corrected. A matched-weight/path probe then established float32 finite-difference instability: float64 finite differences converged to autodiff at about5e-12. Every scientific learner and optimizer was promoted to float64 before training, retaining the original initialization values and identical architecture/settings. This shared numerical feasibility change and the original design are preserved in precision_change.json and design.initial.json. Failed attempts remain available.','',
'The simulator uses a declared daily log-Euler stock step and exact CIR variance transition with coefficients fixed during each interval. Claims concern this discrete simulator. Two-substep Monte Carlo diagnostics are reported with sampling error; they do not certify continuous-time SLV discretization error at the useful-gain scale. Price/filter refinement checks cover the declared validation paths and independent reference states, not all possible latent states.','',
'Family A changes training model-family coverage as well as allocation: its baseline is pure base SLV, while the adaptive generator has positive off-base support. It is not a pure matched-support reweighting experiment. Family B has shared baseline/adaptive support. Neither synthetic leverage function is claimed to be market-calibrated.','',
'Raw pre-premium risk, mean loss, initial liability value, centered terminal risk, daily dispersion, costs, turnover and positions are retained separately. Specialists are learned comparators, not proven optimal hedges. Empty event-aligned plots in domains without switches indicate no qualifying events, not zero risk.','',
'Compilation, evaluation/dataset generation, checkpoint I/O and active training are recorded per arm. Provider cost includes setup, validation, failed attempts and exports. The original MSE and rejected trinomial campaigns remain closed; prior authorized pilot cleanup had already deleted some datasets/checkpoints.','',
'## Evidence','',f'- Canonical GPU outputs: `{OUT}`.',f'- Original worker report and LaTeX: `{OUT}/report.md`, `{OUT}/report.tex`.',f'- Raw datasets, per-checkpoint episode arrays, all checkpoints and figures: `{OUT}`.',f'- Metadata/hash/protocol audit: `{ROOT}/artifact_audit.json`.',f'- Independent numeric audit: `{OUT}/independent_audit.json`.',f'- Provider accounting and shutdown: `{ROOT}/provider_billing.json`.','']
if final:lines += ['Final eight-arm scores and paired entropic comparisons are in final_summary.json; selections and all final dataset hashes were committed before test evaluation. No across-seed or general adaptation-superiority claim follows from a single training seed.','']
(ROOT/'report.md').write_text('\n'.join(lines))
tex=(OUT/'report.tex').read_text();preface='\n'.join([r'\section*{Final assessment}',conclusion.replace('_',' '),f'Provider-reported cost: USD {billing["reported_usd"]:.3f}. All instances shut down.',r'\subsection*{Development specialization gates}',r'\begin{tabular}{lllll}',r'Family & Target & Improvement & 95\% interval & Minimum '+chr(92)*2]+texrows+[r'\end{tabular}',r'\paragraph{Scope.} Single training seed; episode intervals are conditional on the policies. All scientific learners used float64 after a pretraining precision repair. Results concern the declared daily discrete simulator; continuous-time discretization error is not certified at the improvement threshold. Family A changes model-family coverage as well as sampling allocation. Detailed numerical limitations and provenance are in the accompanying Markdown report.'])
tex=tex.replace(r'\begin{document}',r'\usepackage{amsmath,amssymb}'+chr(10)+r'\begin{document}'+'\n'+preface+'\n',1)
tex=tex.replace(r'\end{document}',(ROOT/'methods.tex').read_text()+chr(10)+r'\end{document}');(ROOT/'report.tex').write_text(tex)
requirements=[
 ('Separate namespace, preserve retained prior results','proven','preservation_audit.json and separate campaign/run paths'),
 ('Two bounded configurations, support and observation/pricing specifications','proven','design.json; design.initial.json; scientific fields unchanged under precision_change.json'),
 ('Entropic objectives and required metric conventions','proven','actual_source.py, both objective_validation.json, independent_audit.json'),
 ('GPU numerical validation before scientific training','proven','Both families validation/independent_pricing/objective_validation records before first diagnostic checkpoint'),
 ('One shared seed, same architecture/settings/initial weights, six diagnostic arms','proven','Initialization hashes, checkpoint schedules and artifact_audit.json'),
 ('Paired specialization gates and residual-difficulty review','proven','Both screen/gate.json and residual_review.json'),
 ('Fresh confirmation only after successful screening','proven','gate_summary.json, confirmation artifacts when required'),
 ('Freeze accepted benchmark, then eight equal-budget arms','proven' if accepted else 'not entered under mandatory stop','accepted_freeze.json/final_training' if accepted else 'Neither family passed full gate'),
 ('Freeze selection and all final hashes before final evaluation','proven' if accepted else 'not entered under mandatory stop','final_selections_frozen.json/final_dataset_hashes_frozen.json' if accepted else 'No final data opened'),
 ('Raw outcomes, resumable states, metrics, plots and LaTeX report','proven','Canonical outputs plus final report.tex/report.md'),
 ('Artifact/numeric audit, spend accounting and confirmed shutdown','proven','artifact_audit.json, independent_audit.json, provider_billing.json'),
 ('Honest negative/inconclusive conclusions with single-seed limits','proven','This final assessment; no optimality or across-seed claim')]
completion=dict(objective='Execute revised BS/SLV plan',conditional_branch='eight-arm comparison' if accepted else 'mandatory negative benchmark stop',accepted_benchmark=accepted,complete=True,requirements=[dict(requirement=a,status=b,evidence=c) for a,b,c in requirements],numerical_scope_limits_explicit=True,remaining_required_work=False,closed_at=now.isoformat())
write(ROOT/'completion_audit.json',completion)
ledger.update(status='closed',closed_at=now.isoformat(),spent_reported_usd=billing['reported_usd'],reserved_usd=0,final_run=str(RUN),conditional_branch=completion['conditional_branch']);write(ROOT/'ledger.json',ledger)
(ROOT/'STATE.md').write_text('# CLOSED\n\n'+conclusion+'\n\nAll canonical outputs and prior failed attempts preserved. Final report/report.tex, completion_audit.json, metadata and numeric audits, provider billing and zero-instance shutdown verified. Do not automatically expand candidate menu, lower thresholds or reopen this budget.\n')
write(ROOT/'closure_hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.iterdir() if p.is_file() and p.name!='closure_hashes.json'})
print(json.dumps(completion,indent=2))
