"""Control-only package and run independent credit development gates."""
from pathlib import Path
import json,hashlib,base64,subprocess,shutil,sys,argparse
from offer_exclusions import exclusion_arguments
root=Path(__file__).resolve().parent;base=root.parent
parser=argparse.ArgumentParser();parser.add_argument('--bp',type=int,choices=(1,3,10),default=1);parser.add_argument('--campaign',choices=('credit-ordinary-long-20261018','credit-high-default-20261018'));args=parser.parse_args();bp=args.bp
processes=[]
for campaign in ([args.campaign] if args.campaign else ('credit-ordinary-long-20261018','credit-high-default-20261018')):
 c=base/campaign;gate=root/(campaign+f'-gate-{bp}bp');gate.mkdir(exist_ok=True)
 launch=gate/'launcher';launch.mkdir(exist_ok=True)
 if (launch/'active.json').exists():raise RuntimeError('Existing gate worker: '+str(launch))
 if (gate/'launcher.log').exists():raise RuntimeError('Existing attempt; inspect before restart')
 for name in ('job.py','smoke.py','requirements-jax.txt','example_jax.py','USAGE.txt','AGENTS.md'):
  source=c/'launchers/baseline'/name
  if name in ('smoke.py','example_jax.py'):source=base/'credit-ordinary-long-20261018/launchers/baseline'/name
  shutil.copy2(source,launch/name)
 subprocess.run(['python',str(root/'harden_export.py'),str(launch)],check=True)
 files={str(p.relative_to(c/'source')):p.read_text() for p in (c/'source').rglob('*') if p.is_file() and p.suffix in ('.py','.json')}
 files['credit_gate.py']=(root/'credit_gate.py').read_text();payload={};provenance={}
 for arm in ('baseline','easy','hard'):
  run=next(r for r in sorted((c/'launchers'/arm/'runs').glob('*'),reverse=True) if (r/f'outputs/cost-{bp}bp/learned/selected.eqx').exists())
  provenance[arm]={'run':str(run)}
  for name in ('selection.json','selected.eqx'):
   data=(run/f'outputs/cost-{bp}bp/learned'/name).read_bytes();payload[arm+'/'+name]=base64.b64encode(data).decode();provenance[arm][name]=hashlib.sha256(data).hexdigest()
 worker="""import pathlib,os,sys,base64
root=pathlib.Path(os.environ['JOB_OUTPUT_DIR']);src=root/'source';src.mkdir(exist_ok=True)
for n,v in FILES.items():
 p=src/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(v)
for n,v in PAYLOAD.items():
 p=root/'selected'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(v))
sys.path.insert(0,str(src))
from credit_gate import main
main(src,root/'selected',root/'gate',BP)
""".replace('BP',str(bp),1).replace('FILES',repr(files),1).replace('PAYLOAD',repr(payload),1)
 (gate/'worker.py').write_text(worker)
 (gate/'provenance.json').write_text(json.dumps({'checkpoints':provenance,'source_hashes':{n:hashlib.sha256(v.encode()).hexdigest() for n,v in files.items()},'streams':{'screen':620261017,'confirmation':720261017},'final_unopened':520261017,'minimum_gain':.001},indent=2))
 cmd=['uv','run','--no-project','python',str(launch/'job.py'),'--job',str(gate/'worker.py'),'--requirements',str(c/'requirements.txt'),'--timeout-seconds','7200','--max-hourly','0.60','--require-artifact','gate/complete.json','--require-artifact','gate/results.json']
 cmd += exclusion_arguments()
 log=(gate/'launcher.log').open('w');p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);log.close();processes.append((campaign,p))
for campaign,p in processes:
 code=p.wait();print(campaign,code,flush=True)
