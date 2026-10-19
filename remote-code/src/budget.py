"""Metadata-only transactional central rental reservations and dollar watchdog."""
import fcntl,json,time,uuid
from pathlib import Path
from datetime import datetime,timezone
DEADLINE=datetime.fromisoformat('2026-10-18T05:54:00+00:00').timestamp()
COMPUTE_DEADLINE=DEADLINE-1800
class Budget:
    def __init__(self,root):
        self.root=Path(root)/'budget';self.root.mkdir(exist_ok=True)
    def reserve(self,worker,seconds,hourly,extras=.5):
        with (self.root/'lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            path=self.root/'reservations.json';state=json.loads(path.read_text()) if path.exists() else {}
            committed=sum(x['reserved_usd'] if x['status']=='reserved' else x['actual_usd'] for x in state.values())
            estimate=seconds*hourly/3600+extras
            if time.time()+seconds>COMPUTE_DEADLINE:raise RuntimeError('Reservation exceeds compute deadline')
            if committed+estimate>22:raise RuntimeError('Reservation exceeds $22 normal-work ceiling')
            token=uuid.uuid4().hex
            state[token]={'worker':worker,'reserved_usd':estimate,'status':'reserved','seconds':seconds,'hourly':hourly,'created':time.time()}
            temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2));temporary.replace(path);self.event({'event':'reserve','token':token,**state[token]});return token
    def reconcile(self,token,actual_usd):
        with (self.root/'lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);path=self.root/'reservations.json';state=json.loads(path.read_text())
            state[token].update(status='reconciled',actual_usd=actual_usd);temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2));temporary.replace(path)
            self.event({'event':'reconcile','token':token,'actual_usd':actual_usd})
    def event(self,value):
        with (self.root/'ledger.jsonl').open('a') as f:f.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),**value))+'\n');f.flush()

    def estimated_spend(self):
        path=self.root/'reservations.json'
        state=json.loads(path.read_text()) if path.exists() else {}
        estimate=0.
        for item in state.values():
            if item['status']=='reconciled':estimate+=item['actual_usd'];continue
            history=self.root.parent/'workers'/item['worker']/'rental_history.json'
            if history.exists():
                h=json.loads(history.read_text());estimate+=(h.get('destroyed_at',time.time())-h['started'])*item['hourly']/3600+.5
            else:estimate+=(time.time()-item['created'])*item['hourly']/3600+.5
        return estimate
