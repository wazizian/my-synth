"""GPU-host stdlib fixture executing the actual exploratory controller with rental stubs."""
import ast
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import types

source=Path(sys.argv[1]);draft=Path(sys.argv[2]);output=Path(sys.argv[3]);output.mkdir(exist_ok=True)
tree=ast.parse(source.read_text());compile(tree,str(source),'exec')
functions=[node for node in tree.body if isinstance(node,ast.FunctionDef)]
arms=('baseline','easy_reference','hard_reference','dpg_hard','soar_hard','selfplay','dpg_mixed','soar_mixed')
base=Path(tempfile.mkdtemp(prefix='option-exploratory-control-'));root=base/'comparison';root.mkdir();campaign=base/'volatility-shocks-20261018';campaign.mkdir()
namespace=root/'option-exploratory-20261019';deadline=datetime.datetime(2026,10,19,18,tzinfo=datetime.timezone.utc).timestamp()
def write_json(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
freeze={'selected':None,'status':'benchmark_skipped','comparison_scope':'original failed gates',
        'candidates':[{'bp':bp,'passed':False,'scales':[.01,.02]} for bp in (1,3,10)]}
write_json(campaign/'option_freeze.json',freeze);freeze_hash=hashlib.sha256((campaign/'option_freeze.json').read_bytes()).hexdigest()
for bp in (1,3,10):write_json(campaign/f'gate-{bp}bp-results.json',{'passed':False})
write_json(root/'option-pipeline-state.json',{'stage':'skipped_no_cost_passed_specialist_gate'})
for name in ('option_training_approval.json','controller_handoff_approval.json','history_archive_approval.json'):
    write_json(root/name,{'status':'GPU_validated','source_hashes':{},'adam_eps_root':1e-16})
shutil.copy2(draft,root/'design.option.draft.json')
(root/'option-source').mkdir();write_json(root/'option-source/initial_premiums.json',{})
write_json(root/'option-exploratory-budget-approval.json',{'status':'approved','selected_bp':3,'namespace':str(namespace),'reserve_usd':10,'valid_until_utc':'2026-10-19T18:00:00+00:00'})
for name in ('final_evaluation.py','package_evaluation.py'):(root/name).write_text('# fixture payload\n')
write_json(root/'evaluation_approval.json',{'status':'GPU_validated','source_hashes':{'final_evaluation.py':hashlib.sha256((root/'final_evaluation.py').read_bytes()).hexdigest()},'packager_sha256':hashlib.sha256((root/'package_evaluation.py').read_bytes()).hexdigest()})
template=base/'credit-ordinary-long-20261018/launchers/baseline';template.mkdir(parents=True)
for name in ('job.py','smoke.py','requirements-jax.txt','example_jax.py','USAGE.txt','AGENTS.md'):(template/name).write_text('fixture')
runs={}
for arm in arms:
    run=namespace/arm/'launcher/runs/fixture';out=run/'outputs';out.mkdir(parents=True);runs[arm]=run
    (out/'selected.pkl').write_bytes(b'fixture model; never unpickled')
    write_json(out/'selection.json',{'trials':[{'lr':lr} for lr in (.0001,.0003,.001)]})
    for lr in (.0001,.0003,.001):
        trial=out/'training'/f'lr-{lr}';trial.mkdir(parents=True)
        for category in ('selection','monitor'):
            for boundary in (0,60,180,360,600,1200,1800):write_json(trial/f'{category}-{boundary}.json',{})
commands=[];budget_calls=[];design_checks=[]
def subprocess_run(command,**kwargs):
    commands.append(command)
    if any(str(item).endswith('launch_arms.py') for item in command):
        assert '--arm' not in command and command[command.index('--bp')+1]=='3'
        design=json.loads((root/'design.option-exploratory-20261019.frozen.json').read_text())
        assert design['learning_rates']==[.0001,.0003,.001] and design['checkpoint_active_seconds'][-1]==1800
        assert design['discovery_gate_passed'] is False and design['deadline_utc']=='2026-10-19T18:00:00+00:00'
    if any(str(item).endswith('package_evaluation.py') for item in command):
        manifest=json.loads((root/'option-exploratory-final-manifest.json').read_text())
        assert manifest['curves']=={} and len(manifest['trial_curves'])==24
        assert set(manifest['models'])==set(arms) and manifest['user_authorized_gate_override'] is True
        assert manifest['discovery_freeze']['selected'] is None and manifest['seed']==520261017
        assert manifest['episodes']==8192 and manifest['bootstrap']==1000
        assert 'exploratory' in manifest['comparison_scope']
    return types.SimpleNamespace(returncode=0)
def preallocation(command,launch,log,passed_deadline):
    assert passed_deadline==deadline and command[command.index('--max-hourly')+1]=='0.60'
    assert 0<int(command[command.index('--timeout-seconds')+1])<=7200
    run=launch/'runs/evaluation-fixture';write_json(run/'outputs/evaluation/complete.json',{})
    write_json(run/'outputs/evaluation/results.json',{'fixture':True});write_json(run/'artifacts.json',{})
scope={'ROOT':root,'NAMESPACE':namespace,'DEADLINE':deadline,'ARMS':arms,'Path':Path,'json':json,
       'datetime':datetime.datetime,'timezone':datetime.timezone,'time':time,'fcntl':fcntl,'hashlib':hashlib,
       'shutil':shutil,'subprocess':types.SimpleNamespace(run=subprocess_run,STDOUT=-2),
       'wait_for_admission':lambda *args:budget_calls.append(args),
       'deadline_utc_for':lambda _: '2026-10-19T18:00:00+00:00',
       'assert_design_deadline':lambda *args:design_checks.append(args),
       'run_preallocation':preallocation,'resumable_preallocation':lambda *args:False}
exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),scope)
scope['complete_runs']=lambda _:runs
scope['main']()
assert len(budget_calls)==1 and len(design_checks)==1
assert hashlib.sha256((campaign/'option_freeze.json').read_bytes()).hexdigest()==freeze_hash
assert json.loads((root/'option-pipeline-state.json').read_text())['stage']=='skipped_no_cost_passed_specialist_gate'
assert json.loads((root/'option-exploratory-pipeline-state.json').read_text())['stage']=='eight_arm_training_and_final_evaluation_complete'
proof={'status':'passed','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
       'actual_controller_main_executed':True,'all_eight_full_three_lr_1800_budgets':True,
       'original_failed_gate_and_skip_preserved':True,'exploratory_scope_explicit':True,
       'fresh_namespace_deadline_budget_checks':True,'bounded_curves_empty_all24_trial_curves_retained':True,
       'frozen_all8_manifest_before_mock_final_evaluation':True,'no_actual_rentals_or_numerical_imports':True}
(output/'validation.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
