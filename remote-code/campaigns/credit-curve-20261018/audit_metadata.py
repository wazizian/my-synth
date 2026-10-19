"""Control-plane integrity and protocol audit; no numerical computations."""
from pathlib import Path
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET

parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);args=parser.parse_args()
campaign=Path(__file__).resolve().parent
run=args.run.resolve();out=run/'outputs';checks=[]

def check(name,condition,evidence):
    checks.append(dict(requirement=name,passed=bool(condition),evidence=evidence))

def read(path):return json.loads(path.read_text())

sync_path=run/'recovered-sync-status.json' if (run/'recovered-sync-status.json').exists() else run/'sync-status.json'
inventory_path=run/'recovery_artifacts.json' if (run/'recovery_artifacts.json').exists() else run/'artifacts.json'
check('Final output sync',read(sync_path)['final_sync_complete'],str(sync_path))
manifest=read(inventory_path)
cleanup_path=Path('/home/jump/campaign-control/checkpoint-cleanup-20261018-followup.json')
cleanup=read(cleanup_path)
removed={row['path']:row for row in cleanup['files']} if cleanup['status']=='complete' else {}
for row in manifest:
    path=out/row['path'];digest=hashlib.sha256()
    deletion=removed.get(str(path))
    if deletion is not None and not path.exists():
        check('User-authorized post-export checkpoint deletion',
              deletion['sha256']==row['sha256'] and deletion['bytes']==row['bytes']
              and path.suffix=='.eqx' and 'checkpoints' in path.parts,
              dict(path=row['path'],authorization=str(cleanup_path)))
        continue
    if path.is_file():
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    check('Final artifact integrity',path.is_file() and path.stat().st_size==row['bytes'] and digest.hexdigest()==row['sha256'],row['path'])
for name,digest in read(out/'source_hashes.json').items():
    path=out/'source'/name
    check('Executed source/config provenance',path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==digest,name)
check('Frozen design unchanged',hashlib.sha256((out/'source/design.json').read_bytes()).hexdigest()==hashlib.sha256((campaign/'design.json').read_bytes()).hexdigest(),'source/design.json')
validation=read(out/'validation.json')
check('GPU validation passed',validation['exit_code']==0 and validation['backend']=='gpu','validation.json')
xml=ET.parse(out/'tests.xml');cases=xml.findall('.//testcase')
check('Test outcomes',len(cases)==39 and not xml.findall('.//failure') and not xml.findall('.//error'),'tests.xml')
names={x.attrib.get('name','') for x in cases}
for name in ('test_two_factor_mixed_coupon_book_is_self_financing_in_expectation',
             'test_independent_full_calendar_reference_matches_production_tables',
             'test_observable_rules_obey_actions_and_cash_accounting',
             'test_finite_gradient_and_exact_checkpoint_resume'):
    check('Required numerical/ledger validation coverage',name in names,name)
design=read(campaign/'design.json');numerical=read(out/'numerical/numerical_summary.json')
for domain in ('easy','ordinary','hard'):
    evidence=numerical[domain];cfg=evidence['production_config']
    check('Pricing refinement and boundary gate',evidence['passed'] and evidence['boundary_max_cash_error']<design['thresholds']['pricing_max_error'],domain)
    frozen=design['common']|design['components'][domain]|design['calendar']
    check('Economic parameters remain frozen',all(cfg[k]==v for k,v in frozen.items() if k not in ('grid_size','fast_grid_size')),domain)
policies=('unhedged','replenish_0.5','replenish_1.0','replace_daily_0.5','replace_daily_1.0','observable_sensitivity')
metrics=read(out/'economic/all_metrics.json')
for domain in ('easy','ordinary','hard','base'):
    check('Required domains and feasible policies',set(metrics[domain])==set(policies),domain)
    for policy in policies:
        row=metrics[domain][policy]
        check('Metric coverage',all(k in row for k in ('entropic','cvar95','daily_error_std','mean','terminal_std','centered_entropic','centered_cvar95','mean_cost','mean_turnover','initial_bond_price','entropic_ess')),domain+'/'+policy)
        if domain!='base':
            check('Frozen episode count',row['n']==8192 and row['default_count']+row['survival_count']==8192,domain+'/'+policy)
            paths=list((out/'economic'/domain).glob(policy+'-[0-9][0-9][0-9][0-9][0-9].npz'))
            check('All raw daily/terminal batches retained',len(paths)==64,domain+'/'+policy)
    check('Risk and daily figures retained',all((out/'economic/figures'/f'{domain}-{kind}.png').is_file() for kind in ('risk','daily')),domain)
audit=read(out/'economic/independent_audit.json')
check('Independent raw-record audit',audit['passed'] and len(audit['checks'])==18 and all(x['episodes']==8192 and x['max_accounting_error']<=1e-10 and all(v<=1e-10 for v in x['metric_errors'].values()) for x in audit['checks']),'economic/independent_audit.json')
gate=read(out/'economic/gate.json')
check('Completion marker agrees with gate',read(out/'economic_complete.json')==gate,'economic_complete.json')
check('Report artifacts retained',all((out/'economic'/f'report.{ext}').is_file() for ext in ('md','tex')),'economic/report.md and report.tex')
for old in ('slv-20261018','entropic-20261018'):
    root=campaign.parent/old
    for name,expected in read(root/'closure_hashes.json').items():
        p=root/name;check('Prior closure artifacts unchanged',p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==expected,old+'/'+name)
result=dict(passed=all(x['passed'] for x in checks),checks=checks,
            economic_stop=gate['stop_uninformative'],
            goal_completion_not_asserted=True,
            remaining_completion_requirements=['Review gate intervals and economic mechanism','Inspect representative figures and LaTeX report','Verify provider billing and zero active instances','Audit conditional plan scope and write final campaign assessment'])
(campaign/'artifact_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(passed=result['passed'],check_count=len(checks),failures=[x for x in checks if not x['passed']],economic_stop=result['economic_stop'])))
if not result['passed']:raise SystemExit(1)
