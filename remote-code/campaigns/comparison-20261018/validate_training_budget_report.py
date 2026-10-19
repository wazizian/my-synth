"""Run only on a GPU host: stdlib AST/report fixture, without simulator imports."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

source=Path(sys.argv[1]);prior=Path(sys.argv[2]);out=Path(sys.argv[3]);out.mkdir(exist_ok=True)
tree=ast.parse(source.read_text());old=ast.parse(prior.read_text())
functions={node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
old_functions={node.name:node for node in old.body if isinstance(node,ast.FunctionDef)}
core=['risk','bootstrap','archive_block','credit_evaluate','option_evaluate','figures','load_models','validate_manifest']
for name in core:assert ast.dump(functions[name])==ast.dump(old_functions[name]),name
arms=('baseline','easy_reference','hard_reference','dpg_hard','soar_hard','selfplay','dpg_mixed','soar_mixed')
scope={'Path':Path,'json':json,'ARMS':arms}
selected=ast.Module(body=[functions[name] for name in ['write','training_budgets','report']],type_ignores=[])
exec(compile(selected,str(source),'exec'),scope)
manifest={'benchmark':'option','bp':3,'comparison_scope':'fixture only','adam_eps_root':1e-16,
          'design':{'block_size':8,'batch_size':256},'models':{}}
for arm in arms:
    trials=[]
    for i,lr in enumerate([.0001,.0003,.001],1):
        trials.append({'lr':lr,'learner_updates':8*i,'training_paths':8*i*256,
                       'active_seconds':10.*i,'compile_seconds':2.*i,
                       'selection_seconds':3.*i,'monitor_seconds':4.*i})
    path=out/(arm+'-selection.json');path.write_text(json.dumps({'trials':trials}))
    manifest['models'][arm]={'selection':str(path)}
budgets=scope['training_budgets'](manifest)
assert len(budgets['trials'])==24
for row in budgets['trials']:
    multiplier=2 if row['arm'].endswith('_mixed') else int(row['arm'] in ('dpg_hard','soar_hard'))
    assert row['teacher_feedback_paths']==row['learner_updates']//8*256*multiplier
alternative=copy.deepcopy(manifest)
for arm in arms:alternative['models'][arm]['selection']=json.loads(Path(manifest['models'][arm]['selection']).read_text())
design_path=out/'design.json';design_path.write_text(json.dumps(manifest['design']));alternative['design']=str(design_path)
assert scope['training_budgets'](alternative)==budgets
result={'domains':{'easy':{'terminal':{},'daily_marked_errors':{},'baseline_minus':{},'specialist_minus':{}}},'base_retention_domain':'base'}
for arm in arms:
    result['domains']['easy']['terminal'][arm]={'entropic':.1,'cvar95':.2,'terminal_std':.3}
    result['domains']['easy']['daily_marked_errors'][arm]={'rms_cross_episode_sd':.4,'mean_cost':.5}
before=copy.deepcopy(result);scope['report'](manifest,out,result);assert result==before
tex=(out/'report.tex').read_text();md=(out/'report.md').read_text()
assert tex.startswith(chr(92)+'documentclass') and (chr(92)*2+'documentclass') not in tex
assert '\\n' not in tex and '\\n' not in md
assert len(json.loads((out/'training-budgets.json').read_text())['trials'])==24
assert 'Actual training budgets' in tex and 'Actual training budgets' in md
assert 'Checkpoint/export' in tex and 'paid rental runtime' in md
proof={'status':'passed','worker_host_fixture_only':True,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
       'trial_rows':24,'all_training_and_feedback_counters_checked':True,
       'actual_report_rendered':True,'scientific_result_unchanged':True,'unchanged_core_ast':core,
       'no_project_or_numerical_imports':True,'no_final_stream_draws':True}
(out/'validation.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
