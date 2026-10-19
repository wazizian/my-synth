from pathlib import Path
import json,datetime,hashlib,xml.etree.ElementTree as ET
c=Path(__file__).resolve().parent
summary={}
for arm in ('baseline','easy','hard','evaluation'):
    roots=sorted((c/'launchers'/arm/'runs').glob('*'));runs=[]
    for run in roots:
        o=run/'outputs';x=dict(run=str(run))
        for name in ('validation.json','learned/selection.json','failure.json'):
            p=o/name
            if p.exists():x[name]=json.loads(p.read_text())
        x['progress']={p.parent.name:json.loads(p.read_text()) for p in o.glob('learned/*/progress.json')}
        log=run/'console.log'
        if log.exists():x['tail']=log.read_text(errors='replace').splitlines()[-3:]
        runs.append(x)
    active=c/'launchers'/arm/'active.json'
    summary[arm]=dict(active=json.loads(active.read_text()) if active.exists() else None,runs=runs)
(c/'status.latest.json').write_text(json.dumps(dict(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),arms=summary),indent=2)+'\n')
for arm,x in summary.items():
    print(arm, 'instance', (x['active'] or {}).get('instance_id'))
    for run in x['runs']:
        print(' ',Path(run['run']).name, 'validation',run.get('validation.json'), 'progress',[(k,v['steps'],v['best']) for k,v in run['progress'].items()])
        tail=run.get('tail',[])
        if tail:
            try:
                event=json.loads(tail[-1]);print(' ',{k:event[k] for k in ('event','lr','step','active_seconds','best') if k in event})
            except ValueError:print(' ',tail[-1])
        if 'failure.json' in run:print(run['failure.json'])
