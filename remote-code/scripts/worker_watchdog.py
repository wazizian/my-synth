"""Supervisor-failure compute stop using the instance's own provider key.

Vast documents CONTAINER_API_KEY/CONTAINER_ID and self-destruction in its Docker
Environment guide. No account credential is copied to the worker or logged.
"""
import os,sys,time,urllib.request,json
from pathlib import Path
end=float(sys.argv[1])
instance=os.environ.get('CONTAINER_ID');key=os.environ.get('CONTAINER_API_KEY')
if not instance or not key:raise SystemExit('Missing provider instance-local watchdog environment')
Path('/tmp/campaign-watchdog-status.json').write_text(json.dumps({'instance_id':instance,'deadline':end,'armed':True}))
time.sleep(max(0,end-time.time()))
for attempt in range(20):
    try:
        exported=Path('/workspace/job/outputs/export_ack.json').exists()
        request=urllib.request.Request('https://console.vast.ai/api/v0/instances/'+instance+'/',method='DELETE' if exported else 'PUT',data=None if exported else json.dumps({'state':'stopped'}).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=20) as response:
            if response.status<300:break
    except Exception:
        time.sleep(15)
