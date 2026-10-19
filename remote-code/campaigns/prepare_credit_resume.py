from pathlib import Path
import json,base64,zlib,hashlib,shutil
ROOT=Path(__file__).resolve().parent
for name in ('credit-ordinary-long-20261018','credit-high-default-20261018'):
 c=ROOT/name;p=c/'source/credit_hedging/learned_run.py';s=p.read_text()
 if 'resume_replay_error' not in s:
  s=s.replace('    trials=[]\n','''    trials=json.loads((out/'trials.json').read_text()) if (out/'trials.json').exists() else []
    for row in trials:
        row['checkpoint']=str(out/f"lr-{row['lr']:g}"/'best.eqx')
''')
  s=s.replace("    for lr in design['learning_rates']:\n", "    for lr in design['learning_rates']:\n        if any(row['lr']==lr for row in trials):continue\n")
  start=s.index("        save('initial');save('best');save('latest')")
  end=s.index("            with (trial/'train.jsonl').open('a') as log:",start)
  s=s[:start]+'''        progress_path=trial/'progress.json'
        resumed=progress_path.exists()
        if resumed:
            progress=json.loads(progress_path.read_text())
            history=progress['history'];active=history[-1]['active_seconds'];steps=history[-1]['steps']
            best=progress['best'];best_step=progress['best_step'];best_active=progress['best_active_seconds']
            boundary=design['checkpoint_active_seconds'][len(history)]
            template=(model,state,jax.random.key_data(key))
            model,state,keydata=eqx.tree_deserialise_leaves(trial/f'active-{boundary}.eqx',template)
            key=jax.random.wrap_key_data(keydata)
            best_model,_,_=eqx.tree_deserialise_leaves(trial/'best.eqx',template)
            replay=selection(best_model,tables,cfgs,domains,weights,design['selection_seed'])
            resume_replay_error=abs(replay['entropic']-best)
            if resume_replay_error>1e-10:raise RuntimeError('Resume best-checkpoint replay mismatch')
            oldlog=trial/'train.jsonl'
            if oldlog.exists():os.replace(oldlog,trial/'train-pre-restart.jsonl')
            # Compile using restored state without consuming an extra optimizer step or PRNG draw.
            t=time.monotonic();warm=update(model,state,key,tables,cfgs,domains,counts,weights,optimizer);jax.block_until_ready(warm[3]);compile_time=progress['compile_seconds']+time.monotonic()-t
            write(trial/'resume.json',dict(restored_boundary=boundary,restored_steps=steps,restored_active_seconds=active,resume_replay_error=resume_replay_error))
            print(json.dumps(dict(event='resumed',arm=arm,lr=lr,active_seconds=active,steps=steps,resume_replay_error=resume_replay_error)),flush=True)
        else:
            save('initial');save('best');save('latest')
            t=time.monotonic();model,state,key,value,gn=update(model,state,key,tables,cfgs,domains,counts,weights,optimizer);jax.block_until_ready(value)
            compile_time=time.monotonic()-t;steps=1
            print(json.dumps(dict(event='compiled',arm=arm,lr=lr,seconds=compile_time,loss=float(value),gradient=float(gn))),flush=True)
        for boundary in design['checkpoint_active_seconds'][1+len(history):]:
'''+s[end:]
  compile(s,str(p),'exec');p.write_text(s)
 # Rebuild scientific source package with resume support.
 import subprocess
 subprocess.run(['python3',str(c/'package.py')],check=True)
 for arm in ('baseline','easy','hard'):
  runs=sorted((c/'launchers'/arm/'runs').glob('*'));old=runs[-1];out=old/'outputs';files={}
  keep=[]
  for cost in sorted(out.glob('cost-*')):
   learned=cost/'learned'
   if (learned/'complete.json').exists():
    keep += [learned/f for f in ('complete.json','selection.json','selected.eqx')]
   else:
    if (learned/'trials.json').exists():keep.append(learned/'trials.json')
    for trial in learned.glob('lr-*'):
     keep += list(trial.glob('best.eqx'))
     progress=trial/'progress.json'
     if progress.exists():
      d=json.loads(progress.read_text());boundary=(0,60,180,360,600,1200,1800)[len(d['history'])]
      keep += [progress,trial/f'active-{boundary}.eqx']
  for p in keep:
   assert p.exists(),p
   files[str(p.relative_to(out))]=base64.b64encode(zlib.compress(p.read_bytes())).decode()
  files['restart_provenance.json']=base64.b64encode(zlib.compress(json.dumps(dict(original_run=str(old),reason='SSH disconnect; resume complete boundary with optimizer and PRNG state',preserve_completed_costs=True)).encode())).decode()
  worker=(c/'worker.py').read_text()
  insertion="import base64,zlib\nRESUME_PAYLOAD="+repr(files)+"\nfor n,v in RESUME_PAYLOAD.items():\n p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(zlib.decompress(base64.b64decode(v)))\n"
  marker="root=pathlib.Path(os.environ['JOB_OUTPUT_DIR']);src=root/'source';src.mkdir(exist_ok=True)"
  assert marker in worker
  worker=worker.replace(marker,marker+'\n'+''.join('    '+line+'\n' for line in insertion.splitlines()).rstrip())
  before="     main(src,destination/'learned',sys.argv[1])"
  assert before in worker
  worker=worker.replace(before,"     if (destination/'learned/complete.json').exists():\n      print('Preserved completed cost',cost,flush=True)\n     else:\n      main(src,destination/'learned',sys.argv[1])")
  target=c/f'resume_{arm}_worker.py';compile(worker,str(target),'exec');target.write_text(worker)
  (c/f'restart-{arm}-manifest.json').write_text(json.dumps(dict(worker_sha256=hashlib.sha256(worker.encode()).hexdigest(),original_run=str(old),restored_files=sorted(files)),indent=2))
  print('Prepared resume package',name,arm,len(worker),flush=True)
