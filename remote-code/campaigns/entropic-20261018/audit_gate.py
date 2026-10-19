"""Read-only control-plane audit: metadata and file hashes, no numerical workloads."""
import argparse,hashlib,json,math
from pathlib import Path

ARMS=('baseline','easy_reference','hard_reference')
DOMAINS=('base','easy','hard')

def audit(run):
    out=run/'outputs'; failures=[]; checks=[]
    def check(condition,description):
        (checks if condition else failures).append(description)
    def read(p):
        return json.loads(p.read_text())
    summary=read(out/'gate_summary.json'); design=read(out/'design.json')
    local=Path(__file__).resolve().parent
    check(design==read(local/'design.json'),'worker design matches predeclared local design')
    ledger=read(local/'ledger.json')
    check(hashlib.sha256((run/'submitted-job.py').read_bytes()).hexdigest()==ledger['source_sha256'],'submitted source hash matches ledger')
    check(hashlib.sha256((local/'design.json').read_bytes()).hexdigest()==ledger['design_sha256'],'predeclared design hash remains unchanged')
    manifest=read(out/'artifact_hashes.json')
    for name,expected in manifest.items():
        p=out/name
        check(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==expected,'artifact hash: '+name)
    for candidate in summary['results']:
        name=candidate['candidate']; root=out/name
        check(name in {x['name'] for x in design['candidates']},name+': predeclared candidate')
        validation=read(root/'validation.json'); distribution=read(root/'distribution_validation.json')
        check(validation['passed'] and validation['causality'] and validation['resume_exact'] and validation['finite_gradients'],name+': numerical invariants')
        check(validation['telescoping_max_error']<=validation['telescoping_tolerance']<min(design['minimum_improvement'].values())/100,name+': accounting error below useful gap')
        check(distribution['passed'] and distribution['base_full_support'],name+': distribution construction')
        initial=[read(root/a/'initialization.json')['sha256'] for a in ARMS]
        check(len(set(initial))==1,name+': identical initial learner parameters')
        for arm,domain in zip(ARMS,DOMAINS):
            armroot=root/arm; selection=read(armroot/'selection.json')
            check(selection['criterion_domain']==domain,name+'/'+arm+': correct selection purpose')
            check(180<=selection['active_seconds']<181,name+'/'+arm+': common active-time cap within one-second block tolerance')
            check(selection['simulated_training_paths']==selection['steps']*design['common']['batch_size'],name+'/'+arm+': path accounting')
            chosen=Path(selection['checkpoint']).name
            checkpoints=list(armroot.glob('checkpoint-*.pkl'))
            check({x.name for x in checkpoints}=={f'checkpoint-{t}.pkl' for t in design['common']['checkpoint_seconds']},name+'/'+arm+': common checkpoint opportunities')
            points={f'checkpoint-{t}.pkl':read(armroot/f'selection-{t}'/'metrics.json')[domain]['entropic'] for t in design['common']['checkpoint_seconds']}
            check(chosen in points and points[chosen]==min(points.values())==selection['score'],name+'/'+arm+': selection matches declared criterion')
            check((armroot/chosen).is_file(),name+'/'+arm+': selected checkpoint exported')
        for phase in [f'selection-{t}' for t in design['common']['checkpoint_seconds']]+['screen','confirmation']:
            if phase=='confirmation' and not (root/phase).exists():continue
            metrics=[]
            for arm in ARMS:
                path=root/arm/phase if phase.startswith('selection') else root/phase/arm
                m=read(path/'metrics.json'); metrics.append(m)
                for domain in DOMAINS:
                    check((path/(domain+'.npz')).is_file(),name+'/'+phase+'/'+arm+'/'+domain+': raw episode arrays')
                    check(m[domain]['finite_count']==m[domain]['sample_count'],name+'/'+phase+'/'+arm+'/'+domain+': all terminal losses finite')
                    for field in ('entropic','cvar95','daily_error_std','exponential_ess','max_exponential_weight'):
                        check(math.isfinite(m[domain][field]),name+'/'+phase+'/'+arm+'/'+domain+': finite '+field)
            for domain in DOMAINS:
                check(len({m[domain]['dataset_sha256'] for m in metrics})==1,name+'/'+phase+'/'+domain+': paired evaluation data')
        for phase in ('screen','confirmation'):
            p=root/phase/'gate.json'
            if not p.exists():continue
            gate=read(p); actual={}
            for d in ('easy','hard'):
                v=gate['paired_intervals'][d]
                check(v['minimum']==design['minimum_improvement'][d],name+'/'+phase+'/'+d+': frozen absolute threshold')
                actual[d+'_specialization']=v['low']>design['minimum_improvement'][d]
            for field in ('entropic','cvar95','daily_error_std'):
                actual['difficulty_'+field]=gate['scores']['easy_reference']['easy'][field]<gate['scores']['hard_reference']['hard'][field]
            check(actual==gate['checks'] and gate['passed']==all(actual.values()),name+'/'+phase+': gate logic')
            check(gate['passed']==candidate[phase+'_passed'],name+'/'+phase+': summary agrees')
        check(len(list((out/'figures').glob(name+'-*.png')))==18,name+': all requested gate plots')
    accepted=summary['accepted']
    check(accepted is None or any(c['candidate']==accepted and c['screen_passed'] and c.get('confirmation_passed') for c in summary['results']),'accepted benchmark requires both independent stages')
    if accepted is None:
        check(len(summary['results'])==len(design['candidates']),'all bounded candidates attempted before negative result')
    for name in ('report.md','report.tex'):
        check((out/name).is_file() and (out/name).stat().st_size>0,'report: '+name)
    sync=read(run/'sync-status.json')
    check(sync.get('final_sync_complete') is True,'final artifact transfer completed')
    inventory=read(run/'artifacts.json')
    indexed={x['path']:x for x in inventory}
    for name,expected in manifest.items():
        check(name in indexed and indexed[name]['sha256']==expected,'launcher/worker manifest agreement: '+name)
    scientific={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and (p.suffix in ('.npz','.pkl','.png','.tex') or p.name in ('metrics.json','selection.json','training.jsonl','gate.json','gate_summary.json','design.json','report.md'))}
    check(scientific<=set(manifest),'manifest covers all scientific artifacts')
    return dict(passed=not failures,check_count=len(checks)+len(failures),failures=failures,checks=checks,accepted=accepted,sync_status=sync,scope='File integrity, metadata consistency and recorded gate decisions. Numerical validation runs on GPU. Shutdown and provider billing require separate verification.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    result=audit(a.run)
    if a.output:a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
    raise SystemExit(0 if result['passed'] else 1)
