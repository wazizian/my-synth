"""Central supervisor launcher: isolated worker lifecycle, bounded reservation."""
import argparse,json,shutil,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from src.budget import Budget,COMPUTE_DEADLINE
from scripts.run_campaign import build_link_base,LINK_BASE,asset_transfer_plan,transfer_plan
p=argparse.ArgumentParser();p.add_argument('--worker',required=True);p.add_argument('--job',required=True);p.add_argument('--seconds',type=int,required=True);p.add_argument('--assets');p.add_argument('--dry-run',action='store_true');a,rest=p.parse_known_args()
for forbidden in ('--max-hourly','--disk','--boot-timeout-seconds','--timeout-seconds','--sync-timeout-seconds'):
    if forbidden in rest:raise SystemExit('Reserved launcher settings cannot be overridden: '+forbidden)
# Provision + run + bounded export; all paid time is reserved before launch.
transfers=asset_transfer_plan(a.assets) if a.assets else transfer_plan()
seconds=a.seconds+transfers['launcher_overhead_seconds']
if a.dry_run:print(json.dumps({'worker':a.worker,'max_paid_seconds':seconds,'max_hourly':.60,'reserved_usd':seconds*.60/3600+.5,'transfers':transfers}));raise SystemExit()
# Immutable-only link base shares canonical archive inodes with future rsync
# handoffs. The source builder never rewrites existing scientific files.
build_link_base(root,LINK_BASE)
budget=Budget(root);token=budget.reserve(a.worker,seconds,.60)
w=root/'workers'/a.worker;w.mkdir(parents=True,exist_ok=True)
(w/'transfer_plan.json').write_text(json.dumps(transfers,indent=2))
for name in ('job.py','smoke.py','requirements-jax.txt','example_jax.py'):
    shutil.copyfile(Path('/home/jump/vast-tools')/name,w/name)
# Reusing reviewed provider machinery in isolated directories permits centrally
# owned independent jobs, as explicitly authorized by the seven-arm plan.
if a.assets:
    target=w/'job.py';text=target.read_text()
    insert="""with open(ASSET_PATH, 'rb') as asset_stream:
                    remote('mkdir -p /workspace/job/outputs/project; tar -xzf - -C /workspace/job/outputs/project', ASSET_TIMEOUT, stdin=asset_stream)
                upload(args.job, 'job.py')""".replace('ASSET_PATH',repr(str(Path(a.assets).resolve()))).replace('ASSET_TIMEOUT',str(transfers['upload_timeout_seconds']))
    text=text.replace("upload(args.job, 'job.py')",insert)
    target.write_text(text)
# Runtime provenance is per worker, never part of an immutable trial config.
target=w/'job.py';text=target.read_text()
sync_needle="command = ['rsync', '-rlt', '--safe-links'"
if text.count(sync_needle)!=1:raise RuntimeError('Reviewed rsync injection anchor changed')
text=text.replace(sync_needle,"command = ['rsync', '-rlt', '--link-dest="+str(LINK_BASE)+"', '--safe-links'")
text=text.replace("'--timeout=60'","'--timeout=120'")
lifetime_needle='args.lifetime = args.boot_timeout_seconds + 900 + args.timeout_seconds + 4 * args.sync_timeout_seconds + 120'
if text.count(lifetime_needle)!=1:raise RuntimeError('Reviewed worker lifetime anchor changed')
text=text.replace(lifetime_needle,lifetime_needle+' + '+str(transfers['upload_timeout_seconds']))
# Keep rsync's normal temporary-file/rename behavior. Never use --inplace:
# mutable remote updates must not modify shared immutable archive inodes.
needle="                job_args = args.job_args[1:]"
insertion="""                rental = json.loads(smoke.STATE.read_text())
                history = json.loads((ROOT/'rental_history.json').read_text())
                user_env += ' INSTANCE_ID=' + str(rental['instance_id']) + ' INSTANCE_RATE_PER_HOUR=' + str(rental['listed_hourly']) + ' RENTAL_STARTED_AT=' + str(history['started']) + ' CAMPAIGN_COST_AT_LAUNCH=LAUNCH_COST'
                job_args = args.job_args[1:]""".replace('LAUNCH_COST',str(budget.estimated_spend()))
text=text.replace(needle,insertion)
# A worker timeout gives the runner a bounded SIGTERM checkpoint interval before
# the outer SSH deadline and final export. Preflight shares this overall budget.
text=text.replace("' /opt/jax/bin/python -u job.py '","' timeout --signal=TERM --kill-after=90 " + str(max(1,a.seconds-120)) + " /opt/jax/bin/python -u job.py '")
import ast
ast.parse(text);target.write_text(text)
# Keep a durable rental history even if boot fails before workload metadata exists.
target=w/'smoke.py';text=target.read_text()
text=text.replace("instance_id = int(created['new_contract'])", "instance_id = int(created['new_contract'])\n        (ROOT / 'rental_history.json').write_text(json.dumps({'instance_id': instance_id, 'started': time.time(), 'offer': offer}))")
text=text.replace("STATE.unlink(missing_ok=True)", "history_path = ROOT / 'rental_history.json'\n                if history_path.exists():\n                    history = json.loads(history_path.read_text()); history['destroyed_at'] = time.time(); history_path.write_text(json.dumps(history))\n                STATE.unlink(missing_ok=True)")
text=text.replace("offers = [o for o in offers if float(o['dph_total']) <= args.max_hourly]", "offers = [o for o in offers if float(o['dph_total']) <= args.max_hourly and o.get('id') not in (48857502,49972543) and not str(o.get('geolocation','')).endswith(', CN')]")
watchdog=(root/'scripts/worker_watchdog.py').read_text()
text=text.replace('import argparse', 'import argparse\nimport shlex',1)
original="'onstart': 'touch /root/.no_auto_tmux'"
replacement="'onstart': 'touch /root/.no_auto_tmux; nohup python -c ' + shlex.quote("+repr(watchdog)+") + ' ' + str(min(time.time()+getattr(args, 'lifetime',600),1792301940.)) + ' >/tmp/campaign-watchdog.log 2>&1 &'"
text=text.replace(original,replacement)
text=text.replace("offer = min(offers, key=lambda o: float(o['dph_total']))","offer = min(offers, key=lambda o: (not str(o.get('geolocation','')).endswith((', FR', ', ES', ', DE', ', NL', ', GB')), float(o['dph_total'])))")
import ast
ast.parse(text)
target.write_text(text)
start=time.time()
try:
    process=subprocess.Popen([sys.executable,str(w/'job.py'),'--job',str(Path(a.job).resolve()),'--requirements',str(root/'requirements-worker.txt'),'--timeout-seconds',str(a.seconds),'--sync-timeout-seconds',str(transfers['sync_timeout_seconds']),'--sync-interval','30']+rest)
    while process.poll() is None:
        if budget.estimated_spend()>=24 or time.time()>=COMPUTE_DEADLINE or time.time()-start>seconds:
            budget.event({'event':'emergency_stop','worker':a.worker,'estimated_spend':budget.estimated_spend()})
            process.terminate()
            try:process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                subprocess.run([sys.executable,str(w/'smoke.py'),'--cleanup'],timeout=180)
            break
        time.sleep(10)
    result=process
finally:
    elapsed=time.time()-start
    # Conservative envelope until provider billing is reconciled; includes setup,
    # all transfers, and a fixed extra-charge allowance, not only GPU kernels.
    history=json.loads((w/'rental_history.json').read_text()) if (w/'rental_history.json').exists() else None
    if not (w/'active.json').exists():
        paid_elapsed=(history.get('destroyed_at',time.time())-history['started']) if history else 0.
        budget.reconcile(token,paid_elapsed*.60/3600+.5 if history else 0.)
    else:
        budget.event({'event':'unconfirmed_cleanup','token':token,'worker':a.worker,'reservation_remains_open':True})
    budget.event({'event':'worker_finished','worker':a.worker,'elapsed_seconds':elapsed,'cleanup_confirmed':not(w/'active.json').exists(),'billing':'conservative cap estimate; provider reconciliation pending'})
raise SystemExit(result.returncode)
