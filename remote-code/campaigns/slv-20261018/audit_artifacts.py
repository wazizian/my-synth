"""Control-plane metadata/hash audit only; scientific calculations stay on worker."""
from pathlib import Path
import json,hashlib,zipfile,sys
campaign=Path(__file__).resolve().parent
run=Path(sys.argv[1]);out=run/'outputs';checks=[]
def check(name,condition,detail=None):
    checks.append(dict(name=name,passed=bool(condition),detail=detail))
def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
check('final_sync',read(run/'sync-status.json')['final_sync_complete'])
for item in read(run/'artifacts.json'):
    p=out/item['path'];check('launcher_hash:'+item['path'],p.exists() and digest(p)==item['sha256'])
if (out/'artifact_hashes.json').exists():
    for name,expected in read(out/'artifact_hashes.json').items():check('worker_hash:'+name,digest(out/name)==expected)
check('frozen_scientific_design',read(campaign/'design.json')==read(out/'design.json'))
initial=read(campaign/'design.initial.json');current=read(campaign/'design.json')
for name in ['families','gamma','alpha','minimum_improvement','evaluation','numerical_tolerances']:
    check('market_or_threshold_unchanged:'+name,initial[name]==current[name])
for name,value in initial['common'].items():check('original_common_setting_unchanged:'+name,current['common'][name]==value)
check('shared_float64_feasibility',current['common']['policy_dtype']=='float64' and current['common']['optimizer_dtype']=='float64')
original_sources=read(campaign/'source_hashes.json')
check('root_plan_unchanged',digest(Path('/home/jump/plan.md'))==original_sources['plan.md'])
check('repo_plan_unchanged',digest(Path('/home/jump/deep-hedging/PLAN.md'))==original_sources['PLAN.md'])
if (out/'actual_source_sha256.json').exists():
    actual=read(out/'actual_source_sha256.json')['sha256'];check('actual_worker_source',digest(out/'actual_source.py')==actual)
    expected=digest(run/('actual-job.py' if (run/'actual-job.py').exists() else 'submitted-job.py'))
    check('source_execution_provenance',actual==expected)
if (out/'gate_summary.json').exists():
    summary=read(out/'gate_summary.json');check('bounded_families',[x['family'] for x in summary['results']]==['A','B']);check('gate_complete',summary['complete'])
    initial=[];raw_files=[]
    for entry in summary['results']:
        root=out/entry['family']
        for name in ['validation.json','independent_pricing.json','objective_validation.json']:check(entry['family']+':'+name,read(root/name)['passed'])
        primary=read(root/'screen'/'gate.json');residual=read(root/'screen'/'residual_review.json')
        check('overall_screen:'+entry['family'],entry['screen_passed']==(primary['passed'] and not residual['material_contradiction']))
        for arm in ['baseline','easy_reference','hard_reference']:
            initial.append(read(root/arm/'initialization.json')['parameter_sha256'])
            for second in [0,60,120,180]:check('checkpoint:'+entry['family']+arm+str(second),(root/arm/f'checkpoint-{second}.pkl').is_file())
            selected=read(root/arm/'selection.json');check('equal_active_cap:'+entry['family']+arm,180<=selected['active_seconds']<190)
            domain=selected['criterion_domain'];scores={t:read(root/arm/f'selection-{t}'/'metrics.json')[domain]['entropic'] for t in [0,60,120,180]}
            best=min(scores,key=scores.get);check('selection_argmin:'+entry['family']+arm,Path(selected['checkpoint']).name==f'checkpoint-{best}.pkl')
        for phase in ['screen','confirmation']:
            if not (root/phase/'gate.json').exists():continue
            gate=read(root/phase/'gate.json');check('gate_boolean:'+entry['family']+phase,gate['passed']==all(gate['checks'].values()))
            for domain in ['easy','hard']:
                interval=gate['intervals'][domain];threshold=read(campaign/'design.json')['minimum_improvement'][domain]
                check('frozen_gain_threshold:'+entry['family']+phase+domain,interval['minimum']==threshold)
                check('gain_gate:'+entry['family']+phase+domain,gate['checks'][domain+'_specialization']==(interval['low']>threshold))
            for domain in ['base','easy','hard']:
                paired=[read(root/phase/arm/'metrics.json')[domain]['dataset_sha256'] for arm in ['baseline','easy_reference','hard_reference']]
                check('paired:'+entry['family']+phase+domain,len(set(paired))==1)
        check('confirmation_only_after_pass:'+entry['family'],not (root/'confirmation').exists() or entry['screen_passed'])
    check('identical_initialization',len(set(initial))==1)
    eligible=[e['family'] for e in summary['results'] if e.get('confirmation_passed',False)]
    expected=next((name for name in ['A','B'] if name in eligible),None)
    check('predeclared_family_choice',summary['accepted']==expected)
    if summary['accepted'] is None:
        check('mandatory_negative_stop',not (out/'final_training').exists() and not (out/'final_test').exists())
        check('no_final_data_on_failure',not any((out/f/'datasets'/f'8-{d}-16384.npz').exists() for f in ['A','B'] for d in ['base','easy','hard']))
    else:
        arms=['baseline','easy_reference','hard_reference','dpg_hard','soar_hard','selfplay','dpg_mixed','soar_mixed']
        check('eight_final_arms',sorted(read(out/'final_summary.json')['scores'])==sorted(arms))
        check('final_selection_freeze',(out/'final_selections_frozen.json').exists())
        check('all_final_dataset_hashes_frozen',set(read(out/'final_dataset_hashes_frozen.json'))=={'base','easy','hard'})
        for arm in arms:
            selection=read(out/'final_training'/arm/'selection.json');check('final_equal_cap:'+arm,600<=selection['active_seconds']<620)
            for t in [0,200,400,600]:check('final_checkpoint:'+arm+str(t),(out/'final_training'/arm/f'checkpoint-{t}.pkl').exists())
    numeric=read(out/'independent_audit.json');check('independent_numeric_pass',numeric['passed'] and numeric['complete'])
    for p in out.rglob('*.npz'):
        with zipfile.ZipFile(p) as z:
            if 'loss.npy' not in z.namelist():continue
        name=str(p.relative_to(out));raw_files.append(name)
        check('numeric_audit_covers:'+name,numeric['file_sha256'].get(name)==digest(p))
    check('nonempty_scientific_evidence',len(raw_files)>=90)
    check('markdown_report',(out/'report.md').is_file());check('latex_report',(out/'report.tex').is_file());check('figures',len(list((out/'figures').glob('*.png')))>=36)
result=dict(passed=all(x['passed'] for x in checks),checks=len(checks),failures=[x for x in checks if not x['passed']],scope='Metadata, hashes, declared protocol and numeric-audit coverage; visual/manual scientific review remains separate')
(campaign/'artifact_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));raise SystemExit(not result['passed'])
