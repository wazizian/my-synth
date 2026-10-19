"""Development-only all-cost option screen, confirmation and GPU benchmark freeze."""
from pathlib import Path
import base64,fcntl,json,subprocess,time,shutil,re,hashlib,sys
C=Path(__file__).resolve().parent
sys.path.insert(0,str(C.parent/'comparison-20261018'))
from wallet_guard import assert_funding_available
lock=(C/'early-gate.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
TEMPLATE='import os,sys,pathlib,json,base64,hashlib\nroot=pathlib.Path(os.environ[\'JOB_OUTPUT_DIR\']);root.mkdir(exist_ok=True)\nsource=SOURCE\n(root/\'model.py\').write_text(source);sys.path.insert(0,str(root))\nfrom model import *\nassert jax.default_backend()==\'gpu\'\nMODELS=PAYLOAD\nmodel0=Hedger(jax.random.key(20261017));models={}\nfor a,b in MODELS.items():\n p=root/(a+\'.eqx\');p.write_bytes(base64.b64decode(b));models[a]=eqx.tree_deserialise_leaves(p,model0)\ninitial=INITIAL\nresult={\'bp\':BP,\'streams\':{},\'threshold\':.0005,\'final_test_opened\':False,\'selected_sources\':SELECTED}\nfor name,seed in [(\'screen\',821261017),(\'confirmation\',822261017)]:\n stream={}\n for di,domain in enumerate([\'easy\',\'hard\']):\n  rows={a:[] for a in models}\n  for start in range(0,8192,256):\n   logs=simulate(jax.random.fold_in(jax.random.key(seed),di*100000+start),domain,256)[\'logs\']\n   for a,m in models.items():rows[a].append(np.asarray(rollout(m,logs,BP/10000)[\'loss\'])-initial[domain])\n  values={a:np.concatenate(v) for a,v in rows.items()};np.savez_compressed(root/f\'{name}-{domain}.npz\',**values)\n  rng=np.random.default_rng(seed+di);draws=[]\n  for _ in range(1000):\n   ix=rng.integers(8192,size=8192);draws.append(risk(values[\'baseline\'][ix])-risk(values[domain][ix]))\n  stream[domain]={\'gain\':risk(values[\'baseline\'])-risk(values[domain]),\'ci95\':np.quantile(draws,[.025,.975]).tolist(),\'metrics\':{a:metrics(v) for a,v in values.items()}}\n checks={\'easy_useful_gain\':stream[\'easy\'][\'ci95\'][0]>.0005,\'hard_useful_gain\':stream[\'hard\'][\'ci95\'][0]>.0005,\'hard_residual_sd_above_easy\':stream[\'hard\'][\'metrics\'][\'hard\'][\'sd\']>stream[\'easy\'][\'metrics\'][\'easy\'][\'sd\']}\n stream[\'checks\']=checks;stream[\'passed\']=all(checks.values());result[\'streams\'][name]=stream\nresult[\'passed\']=all(s[\'passed\'] for s in result[\'streams\'].values());result[\'status\']=\'complete\'\nwrite(root/\'results.json\',result);write(root/\'complete.json\',{\'status\':\'complete\'})\n\nif BP==10:\n all_results=PREVIOUS+[result];assert sorted(row[\'bp\'] for row in all_results)==[1,3,10];candidates=[]\n for row in all_results:\n  confirmation=row[\'streams\'][\'confirmation\'];low=[confirmation[d][\'ci95\'][0] for d in [\'easy\',\'hard\']];points=[confirmation[d][\'gain\'] for d in [\'easy\',\'hard\']]\n  candidates.append({\'bp\':row[\'bp\'],\'passed\':row[\'passed\'],\'minimum_confirmation_lower_bound\':min(low),\'mean_confirmation_gain\':sum(points)/2,\'scales\':[max(.001,row[\'streams\'][\'screen\'][d][\'metrics\'][\'baseline\'][\'sd\']) for d in [\'easy\',\'hard\']]})\n passing=[r for r in candidates if r[\'passed\']];chosen=max(passing,key=lambda c:(c[\'minimum_confirmation_lower_bound\'],c[\'mean_confirmation_gain\'])) if passing else None\n write(root/\'option_freeze.json\',{\'selected\':chosen,\'candidates\':candidates,\'status\':\'benchmark_frozen\' if chosen else \'benchmark_skipped\',\'comparison_scope\':\'gated\' if chosen else \'no option adaptive comparison; no cost passed the specialist gate\',\'rule\':\'passing costs only; maximum minimum confirmation gain lower bound, then mean point gain\',\'scale_rule\':\'baseline screen terminal SD, floored at.001\',\'final_test_opened\':False,\'final_test_seed\':520261017})\n\n b=chr(92);nl=chr(10)\n lines=[\'# Option specialist discovery\',\'\',\'All three declared costs were screened and independently confirmed on GPU. Useful gain threshold 0.0005; no final test stream was opened.\',\'\',\'| Cost(bp) | Stream | Easy gain | Easy lower95% | Hard gain | Hard lower95% | Gate passed |\',\'|---|---|---:|---:|---:|---:|---|\']\n tex=[b+\'documentclass{article}\',b+\'usepackage[margin=1in]{geometry}\',b+\'begin{document}\',b+\'section*{Option specialist discovery}\',\'No final test stream was opened. Useful gain threshold 0.0005.\',b+\'begin{tabular}{llrrrr}\',\'Cost & Stream & Easy gain & Easy lower95\'+b+\'% & Hard gain & Hard lower95\'+b+\'% \'+b*2]\n for row in all_results:\n  for stream_name in [\'screen\',\'confirmation\']:\n   stream=row[\'streams\'][stream_name];e=stream[\'easy\'];h=stream[\'hard\'];lines.append(f"| {row[\'bp\']} | {stream_name} | {e[\'gain\']:.6g} | {e[\'ci95\'][0]:.6g} | {h[\'gain\']:.6g} | {h[\'ci95\'][0]:.6g} | {stream[\'passed\']} |")\n   tex.append(f"{row[\'bp\']} & {stream_name} & {e[\'gain\']:.6g} & {e[\'ci95\'][0]:.6g} & {h[\'gain\']:.6g} & {h[\'ci95\'][0]:.6g}"+\' \'+b*2)\n tex.extend([b+\'end{tabular}\',b+\'end{document}\']);(root/\'option-discovery-report.md\').write_text(nl.join(lines)+nl);(root/\'option-discovery-report.tex\').write_text(nl.join(tex)+nl)\n'

def selected(arm,bp):
 for r in sorted((C/'launchers'/arm/'runs').glob('*'),reverse=True):
  out=r/f'outputs/training/cost-{bp}bp'
  if (out/'selection.json').exists() and (out/'selected.eqx').exists():return out

def run(bp):
 destination=C/f'gate-{bp}bp-results.json'
 if destination.exists():return
 while True:
  paths={a:selected(a,bp) for a in ['baseline','easy','hard']}
  if all(paths.values()):break
  time.sleep(30)
 payload={a:base64.b64encode((p/'selected.eqx').read_bytes()).decode() for a,p in paths.items()}
 source=(C/'model.py').read_text();diagnostic=json.loads((C/'diagnostic.json').read_text())
 previous=[json.loads((C/f'gate-{cost}bp-results.json').read_text()) for cost in [1,3] if (C/f'gate-{cost}bp-results.json').exists()]
 replacements={'SOURCE':repr(source),'PAYLOAD':repr(payload),'INITIAL':repr({d:diagnostic['quotes'][d]['126']['price'] for d in ['easy','hard']}),'SELECTED':repr({a:json.loads((p/'selection.json').read_text()) for a,p in paths.items()}),'PREVIOUS':repr(previous),'BP':str(bp)}
 # Validate transferred bytes on the GPU before restoring any model or drawing paths.
 hashes={a:hashlib.sha256((p/'selected.eqx').read_bytes()).hexdigest() for a,p in paths.items()}
 marker="model0=Hedger(jax.random.key(20261017));models={}"
 guard="expected_hashes="+repr(hashes)+"\nfor arm,encoded in MODELS.items():\n assert hashlib.sha256(base64.b64decode(encoded)).hexdigest()==expected_hashes[arm],arm+' checkpoint transfer hash mismatch'\n"
 template=TEMPLATE.replace('\n'+marker,'\n'+guard+marker)
 template=template.replace("\ninitial=", "\nwrite(root/'checkpoint-restoration.json',{'status':'all_three_restored','sha256':expected_hashes})\ninitial=",1)
 worker=re.sub(r'\b(SOURCE|PAYLOAD|INITIAL|SELECTED|PREVIOUS|BP)\b',lambda match:replacements[match.group()],template)
 if bp==3:
  # Artifact reproduction only: the existing1bp streams/models are unchanged.
  first=C/'launchers/early-gate-1bp/runs/20261018T230620Z-d1b37803/outputs'
  reference={name:base64.b64encode((first/name).read_bytes()).decode() for name in ['screen-easy.npz','screen-hard.npz','confirmation-easy.npz','confirmation-hard.npz','results.json']}
  replay=(C/'gate-1bp-worker.py').read_text()
  worker=worker.replace("\nwrite(root/'results.json',result);write(root/'complete.json',{'status':'complete'})", "\nwrite(root/'results.json',result)")
  worker+='\nimport subprocess\nreplay_root=root/"verified-1bp-replay";replay_root.mkdir(exist_ok=True)\nreplay_job=root/"replay-1bp.py"\nreplay_job.write_text('+repr(replay)+')\nreference='+repr(reference)+'\n'
  worker+='env=dict(os.environ);env["JOB_OUTPUT_DIR"]=str(replay_root);env["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"\nsubprocess.run([sys.executable,str(replay_job)],env=env,check=True)\nchecks={}\n'
  worker+='for name,encoded in reference.items():\n original=root/("original-1bp-"+name);original.write_bytes(base64.b64decode(encoded))\n if name.endswith(".npz"):\n  before=np.load(original);after=np.load(replay_root/name);assert set(before.files)==set(after.files)\n  checks[name]={}\n  for arm in before.files:\n   error=float(np.max(np.abs(before[arm]-after[arm])));assert np.allclose(before[arm],after[arm],rtol=1e-12,atol=1e-12);checks[name][arm]={"max_error":error,"exact":bool(np.array_equal(before[arm],after[arm]))}\n'
  worker+='old=json.loads((root/"original-1bp-results.json").read_text());new=json.loads((replay_root/"results.json").read_text())\nassert old["passed"]==new["passed"]\nfor stream in ["screen","confirmation"]:\n for domain in ["easy","hard"]:\n  for field in ["gain","ci95"]:assert np.allclose(old["streams"][stream][domain][field],new["streams"][stream][domain][field],rtol=1e-12,atol=1e-12)\nwrite(root/"verified-1bp-replay.json",{"status":"passed","role":"artifact reproduction, not fresh confirmation","terminal_arrays":checks,"metrics_tolerance":1e-12,"same_seeds":[821261017,822261017]})\nwrite(root/"complete.json",{"status":"complete"})\n'
 job=C/f'gate-{bp}bp-worker.py';job.write_text(worker)
 l=C/'launchers'/f'early-gate-{bp}bp';l.mkdir(exist_ok=True)
 for f in ['job.py','smoke.py','example_jax.py','requirements-jax.txt','AGENTS.md','USAGE.txt']:
  if not (l/f).exists():shutil.copy2(C/'launchers/baseline'/f,l/f)
 if (l/'active.json').exists():raise RuntimeError('Existing gate instance; reconnect rather than rent duplicate')
 subprocess.run(['python',str(C.parent/'comparison-20261018/harden_export.py'),str(l)],check=True)
 runs=l/'runs';existing=[r for r in runs.glob('*') if (r/'artifacts.json').exists() and (r/'outputs/results.json').exists()] if runs.exists() else []
 if not existing:
  attempted=l/'attempted-worker.sha256'
  digest=hashlib.sha256(job.read_bytes()).hexdigest()
  if attempted.exists() and attempted.read_text().strip()==digest:
   raise RuntimeError('Identical failed gate payload requires explicit recovery; no automatic rental loop')
  attempted.write_text(digest+'\n')
  args=['/home/jump/.local/bin/uv','run','--no-project','python',str(l/'job.py'),'--exclude-offer','48448052','--exclude-offer','55701183','--exclude-offer','51500671','--exclude-offer','56219704','--exclude-offer','55917102','--job',str(job),'--requirements',str(C/'requirements.txt'),'--timeout-seconds','3600','--max-hourly','.60','--sync-timeout-seconds','600','--require-artifact','results.json','--require-artifact','complete.json']
  if bp==10:args+=['--require-artifact','option_freeze.json','--require-artifact','option-discovery-report.md','--require-artifact','option-discovery-report.tex']
  if bp==3:args+=['--require-artifact','verified-1bp-replay.json','--require-artifact','verified-1bp-replay/results.json']
  stale=Path('/home/jump/campaign-control/stale_offer_ids.json')
  excluded={46384002,28780319}
  if stale.exists():excluded.update(json.loads(stale.read_text()))
  for offer in sorted(excluded):args+=['--exclude-offer',str(offer)]
  assert_funding_available()
  subprocess.run(args,check=True)
  existing=[r for r in runs.glob('*') if (r/'artifacts.json').exists() and (r/'outputs/results.json').exists()]
 r=max(existing);shutil.copy2(r/'outputs/results.json',destination)
 if bp==1:shutil.copy2(destination,C/'early_gate_results.json')
 if bp==10:
  for name in ['option_freeze.json','option-discovery-report.md','option-discovery-report.tex']:shutil.copy2(r/'outputs'/name,C/name)

for bp in [1,3,10]:run(bp)
