"""Bounded stale-offer retry; never retry an actual scientific job automatically."""
import subprocess,sys,json,re,datetime,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent;STATE=Path('/home/jump/vast-tools/active.json');STALE=Path('/home/jump/campaign-control/stale_offer_ids.json')
parser=argparse.ArgumentParser();parser.add_argument('--probe',action='store_true');args=parser.parse_args()
source_path=ROOT/('precision_probe_packaged.py' if args.probe else 'worker_full.py')
job_mode='precision_probe' if args.probe else 'campaign'
launch_stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
for attempt in range(10):
    ledger=json.loads((ROOT/'ledger.json').read_text())
    assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(ledger['deadline'])-datetime.timedelta(hours=4),'Insufficient deadline reservation'
    assert ledger['spent_reported_usd']+2.5<=ledger['diagnostic_cap_usd'],'Insufficient conservative reservation'
    assert not STATE.exists(),'Existing managed job must be recovered first'
    command=[sys.executable,'/home/jump/vast-tools/job.py','--job',str(source_path),'--requirements','/home/jump/deep-hedging/requirements-worker.txt','--exclude-offer','54008097','--timeout-seconds','10800','--boot-timeout-seconds','300','--sync-interval','20','--max-hourly','0.60','--require-artifact','gate_summary.json','--require-artifact','report.tex','--require-artifact','artifact_hashes.json','--','--mode','campaign']
    if args.probe:
        command=command[:command.index('--require-artifact')]+['--require-artifact','precision_probe.json']
        command[command.index('--timeout-seconds')+1]='600'
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    (ROOT/'live_launcher.json').write_text(json.dumps({'controller_pid':__import__('os').getpid(),'launcher_pid':process.pid,'attempt':attempt,'source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest()},indent=2))
    lines=[];offer=None;instance=None;run=None
    for line in process.stdout:
        print(line,end='',flush=True);lines.append(line)
        match=re.search(r'Offer (\d+):',line)
        if match:offer=int(match.group(1))
        match=re.search(r'Created instance (\d+)',line)
        if match:
            instance=int(match.group(1));ledger=json.loads((ROOT/'ledger.json').read_text());ledger['jobs'].append({'instance_id':instance,'offer_id':offer,'status':'running','mode':job_mode,'launcher_pid':process.pid,'source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest()});(ROOT/'ledger.json').write_text(json.dumps(ledger,indent=2))
        match=re.search(r'Local logs and outputs: (.+)',line)
        if match:
            run=match.group(1).strip();ledger=json.loads((ROOT/'ledger.json').read_text());ledger['jobs'][-1]['run']=Path(run).name;(ROOT/'ledger.json').write_text(json.dumps(ledger,indent=2))
    code=process.wait();log=''.join(lines)
    (ROOT/f'launch-{launch_stamp}-attempt-{attempt}.log').write_text(log)
    if code and instance is None and offer is not None and 'no_such_ask' in log and not STATE.exists():
        values=json.loads(STALE.read_text());STALE.write_text(json.dumps(sorted(set(values+[offer]))))
        ledger=json.loads((ROOT/'ledger.json').read_text());ledger['jobs'].append({'offer_id':offer,'status':'stale_offer_no_instance','log':f'launch-{launch_stamp}-attempt-{attempt}.log'});(ROOT/'ledger.json').write_text(json.dumps(ledger,indent=2));continue
    ledger=json.loads((ROOT/'ledger.json').read_text())
    if instance is not None:
        ledger['jobs'][-1]['status']='finished_destroyed' if not STATE.exists() else 'cleanup_unconfirmed';ledger['jobs'][-1]['exit_code']=code;(ROOT/'ledger.json').write_text(json.dumps(ledger,indent=2))
    raise SystemExit(code)
raise SystemExit('Ten unavailable offers; no live instance; inspect availability before next attempt')
