"""GPU-host-only followup; existing diagnostic records, no simulations or final draws."""
from pathlib import Path
import ast,hashlib,json,os,pickle,py_compile,subprocess,sys,types,re,time
HERE=Path(__file__).resolve().parent
OUT=Path('/workspace/job/outputs/option-control-followup');OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(HERE))
import jax,numpy as np
assert jax.default_backend()=='gpu'
import final_evaluation as f

def nodes(path):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
old=nodes(HERE/'old-final_evaluation.py');new=nodes(HERE/'final_evaluation.py')
for name in ['archive_block','credit_evaluate','option_evaluate','bootstrap','risk','figures','Daily','load_models','validate_manifest']:assert old[name]==new[name],name
for benchmark in ['credit','option']:
    folder=HERE/'records'/benchmark;results=json.loads((folder/'results.json').read_text())
    raw=json.loads((folder/'raw-record-manifest.json').read_text());assert len(raw)==3 and all(r['lossless_roundtrip_verified'] and r['max_archived_accounting_error']<1e-10 for r in raw)
    for domain,row in results.items():
        with np.load(folder/f'terminal-{domain}.npz') as archive:
            for arm in f.ARMS:
                now=f.terminal_metrics(archive[arm]);before=row['terminal'][arm]
                for key in now:
                    if key in before:assert abs(now[key]-before[key])<1e-12,(benchmark,domain,arm,key)
                assert now['finite_count']==now['n'] and abs(now['centered_cvar95']-(now['cvar95']-now['mean']))<1e-12
    report=OUT/benchmark;report.mkdir(exist_ok=True)
    f.report({'benchmark':benchmark,'bp':1,'adam_eps_root':1e-16,'comparison_scope':'GPU renderer proof using existing diagnostic records; no final test opened'},report,{'domains':results,'base_retention_domain':'base' if benchmark=='option' else 'ordinary'})
    tex=(report/'report.tex').read_text();md=(report/'report.md').read_text();b=chr(92)
    assert tex.startswith(b+'documentclass') and not tex.startswith(b*2)
    assert chr(10) in tex and chr(10) in md
    assert b*2+'documentclass' not in tex and b+'n' not in tex
    assert not re.search(re.escape(b*2)+r'[A-Za-z]',tex)
# Preserve the shared historical source admission while proving only the additions.
import package_evaluation as packaging
fixture=OUT/'package-fixture';fixture.mkdir(exist_ok=True);cp=fixture/'model.pkl';cp.write_bytes(pickle.dumps({'student':('fixture',{})}));selection=fixture/'selection.json';selection.write_text(json.dumps({'final_test_opened':False}))
manifest={'benchmark':'option','source':str(HERE/'option-source'),'bp':1,'frozen':True,'initial_premiums':json.loads((HERE/'option-source/initial_premiums.json').read_text()),'models':{a:{'checkpoint':str(cp),'selection':str(selection),'sha256':hashlib.sha256(cp.read_bytes()).hexdigest()} for a in f.ARMS},'static_addenda':{name:str(HERE/name) for name in ['OPTION_SCIENTIFIC_LIMITATIONS.md','OPTION_SCIENTIFIC_LIMITATIONS.tex']}}
fixture_manifest=fixture/'manifest.json';fixture_manifest.write_text(json.dumps(manifest));packaging.package(fixture_manifest,fixture/'worker.py');py_compile.compile(str(fixture/'worker.py'),doraise=True)
# Exercise actual generated evaluator wrapper with a fake final-main, never real520.
mock=types.ModuleType('final_evaluation')
def mock_main(manifest_path,out):
    out.mkdir(exist_ok=True);(out/'report.md').write_text('# Mock report'+chr(10));(out/'report.tex').write_text(chr(92)+'documentclass{article}'+chr(10)+chr(92)+'begin{document}'+chr(10)+chr(92)+'end{document}'+chr(10))
mock.main=mock_main;saved=sys.modules['final_evaluation'];sys.modules['final_evaluation']=mock;original_env=dict(os.environ);eval_root=OUT/'mock-evaluation';eval_root.mkdir(exist_ok=True)
try:
    os.environ['JOB_OUTPUT_DIR']=str(eval_root);exec(compile((fixture/'worker.py').read_text(),str(fixture/'worker.py'),'exec'),{'__name__':'__main__'})
finally:
    sys.modules['final_evaluation']=saved;os.environ.clear();os.environ.update(original_env)
assert '[Scientific limitations]' in (eval_root/'evaluation/report.md').read_text()
assert chr(92)+'input{OPTION_SCIENTIFIC_LIMITATIONS.tex}' in (eval_root/'evaluation/report.tex').read_text()
assert (eval_root/'evaluation/OPTION_SCIENTIFIC_LIMITATIONS.tex').read_text().startswith(chr(92)+'section')
# Compile and execute the actual ALL_LR wrapper with fake learner/JAX/adapter only.
subprocess.run([sys.executable,str(HERE/'package_trial.py'),'--benchmark','option','--source',str(HERE/'option-source'),'--design',str(HERE/'design.json'),'--arm','selfplay','--all-learning-rates','--bp','1','--output',str(OUT/'trial-worker.py')],check=True)
py_compile.compile(str(OUT/'trial-worker.py'),doraise=True)
mock_out=OUT/'mock-trial/outputs';mock_out.mkdir(parents=True,exist_ok=True);events=[]
fakejax=types.ModuleType('jax');fakejax.config=types.SimpleNamespace(update=lambda *a,**k:None);fakejax.default_backend=lambda:'gpu';fakejax.clear_caches=lambda:None
fakelearner=types.ModuleType('train_adaptive')
def fake_run(adapter,make_model,folder,design,arm,lr):
    folder.mkdir(parents=True,exist_ok=True);history=Path(os.environ['JOB_SELFPLAY_HISTORY_DIR']);assert not history.is_relative_to(mock_out);history.mkdir(parents=True,exist_ok=True)
    for i in range(3):(history/f'{i:08d}.pkl').write_bytes(pickle.dumps({'round':i,'lr':lr}))
    (folder/'best.pkl').write_bytes(b'known-checkpoint');(folder/'selection.json').write_text(json.dumps({'checkpoint':'best.pkl','selection_score':lr}));events.append({'lr':lr,'history':str(history)})
fakelearner.run=fake_run;fakeadapter=types.ModuleType('option_adapter');fakeadapter.OptionAdapter=lambda *a:types.SimpleNamespace(make_model=lambda seed:None)
import archive_history
real_archive=archive_history.archive_history
marker_true=[]
def checked_archive(history,destination):
    marker=json.loads((mock_out/'recovery-requirements.json').read_text());assert marker['external_history_required_until_complete'];marker_true.append(str(history));return real_archive(history,destination)
archive_history.archive_history=checked_archive
saved_modules={name:sys.modules.get(name) for name in ['jax','train_adaptive','option_adapter']}
sys.modules.update(jax=fakejax,train_adaptive=fakelearner,option_adapter=fakeadapter);original_env=dict(os.environ)
try:
    os.environ['JOB_OUTPUT_DIR']=str(mock_out);exec(compile((OUT/'trial-worker.py').read_text(),str(OUT/'trial-worker.py'),'exec'),{'__name__':'__main__'})
finally:
    for name,value in saved_modules.items():
        if value is None:sys.modules.pop(name,None)
        else:sys.modules[name]=value
    os.environ.clear();os.environ.update(original_env);archive_history.archive_history=real_archive
assert len(events)==3 and len(marker_true)==3
assert len(list((mock_out/'histories').glob('*.tar.gz')))==3
assert not json.loads((mock_out/'recovery-requirements.json').read_text())['external_history_required_until_complete']
assert (mock_out/'complete.json').exists()
# The pending gate report template must also compile for both first and final cost.
tree=ast.parse((HERE/'early_gate.py').read_text());template=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TEMPLATE' for t in n.targets)))
for bp in [1,10]:
    text=template.replace('SOURCE',repr('')).replace('PAYLOAD','{}').replace('INITIAL',"{'easy':0.,'hard':0.}").replace('SELECTED','{}').replace('PREVIOUS','[]').replace('BP',str(bp));compile(text,'gate-fixture.py','exec')
proof={'passed':True,'unchanged_archival_and_simulation_core':True,'all_existing_metric_values_preserved':True,'centered_cvar95_and_finite_count':True,'report_single_command_slashes_and_real_newlines':True,'package_evaluation_generated_worker_compiles':True,'static_addendum_mock_runtime':True,'package_trial_ALL_LR_mock_runtime':True,'cache_outside_outputs':True,'all_LR_archives_and_marker_transitions':True,'gate_report_template_compiles':True,'no_final520_or920_draws':True,'source_hashes':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['final_evaluation.py','package_evaluation.py','package_trial.py','archive_history.py','OPTION_SCIENTIFIC_LIMITATIONS.tex']}}
(OUT/'validation.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof),flush=True)
