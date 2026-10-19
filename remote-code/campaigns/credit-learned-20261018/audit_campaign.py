"""Control-plane artifact/protocol audit; no local numerical experiment."""
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
c=Path(__file__).resolve().parent
checks=[]
def read(p):return json.loads(p.read_text())
def check(name,passed,evidence):checks.append(dict(requirement=name,passed=bool(passed),evidence=evidence))
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
expected=read(c/'training_source_hashes.json');initial=[]
for arm in ('baseline','easy','hard'):
 runs=[p for p in (c/'launchers'/arm/'runs').glob('*') if (p/'outputs/learned/complete.json').exists()]
 check('Exactly one completed scientific trainer',len(runs)==1,arm)
 if len(runs)!=1:continue
 r=runs[0];o=r/'outputs';check('Final training export complete',read(r/'sync-status.json')['final_sync_complete'],str(r))
 for row in read(r/'artifacts.json'):
  p=o/row['path'];check('Training export byte integrity',p.is_file() and p.stat().st_size==row['bytes'] and digest(p)==row['sha256'],str(p))
 check('Executed source matches frozen trainer',read(o/'source_hashes.json')==expected,arm)
 validation=read(o/'validation.json');check('GPU training validation',validation['exit_code']==0 and validation['backend']=='gpu',arm)
 xml=ET.parse(o/'tests.xml');check('All41 GPU tests passed',len(xml.findall('.//testcase'))==41 and not xml.findall('.//failure') and not xml.findall('.//error'),arm)
 init=read(o/'learned/initialization_validation.json');initial.append(init['model_sha256']);check('Exact static initialization',init['max_error']==0,arm)
 choice=read(o/'learned/selection.json');check('All tuning rates and budgets completed',len(choice['trials'])==3 and {x['lr'] for x in choice['trials']}=={.0001,.0003,.001} and all(x['active_seconds']>=600 for x in choice['trials']),arm)
 check('Initial checkpoint eligible and development-only selection',choice['selected']['entropic']<=choice['initial']['entropic'] and choice['final_test_opened'] is False,arm)
 for t in choice['trials']:
  p=o/'learned'/Path(t['checkpoint']).parent.name/'progress.json';history=read(p)['history'];check('Matched checkpoint opportunities',len(history)==4 and all(v['active_seconds']>=threshold for v,threshold in zip(history,(60,180,360,600))),str(p))
 check('Selected checkpoint bytes match lineage',digest(o/'learned/selected.eqx')==digest(o/'learned'/Path(choice['selected']['checkpoint']).parent.name/'best.eqx'),arm)
check('Paired initialization across all arms',len(initial)==3 and len(set(initial))==1,'initialization_validation.json')
evals=[p for p in (c/'launchers/evaluation/runs').glob('*') if (p/'outputs/complete.json').exists()]
check('Completed fresh evaluation and report',len(evals)==1,'launchers/evaluation/runs')
if len(evals)==1:
 r=evals[0];o=r/'outputs';e=o/'evaluation'
 check('Final evaluation export complete',read(r/'sync-status.json')['final_sync_complete'],str(r))
 for row in read(o/'scientific_manifest.json'):
  p=o/row['path'];check('Remote scientific manifest integrity',p.is_file() and p.stat().st_size==row['bytes'] and digest(p)==row['sha256'],str(p))
 for row in read(r/'artifacts.json'):
  p=o/row['path'];check('Evaluation launcher inventory integrity',p.is_file() and p.stat().st_size==row['bytes'] and digest(p)==row['sha256'],str(p))
 check('Independent tuning score reconstruction',read(e/'selection_audit.json')['passed'],str(e/'selection_audit.json'))
 check('Full-calendar selected checkpoint replay',read(e/'checkpoint_replay_audit.json')['passed'],str(e/'checkpoint_replay_audit.json'))
 audit=read(e/'independent_audit.json');check('All selected raw-record audit batches',audit['passed'] and len(audit['checks'])==969,str(e/'independent_audit.json'))
 freeze=read(o/'frozen_selection.json');check('Neural choices frozen before fresh data',freeze['evaluation_opened'] is False,str(o/'frozen_selection.json'))
 staticfreeze=read(e/'evaluation_freeze.json');check('Static choices frozen before fresh data',staticfreeze['fresh_evaluation_opened'] is False,str(e/'evaluation_freeze.json'))
 for arm in ('baseline','easy','hard'):
  check('Executed selected models match frozen checkpoint hashes',digest(o/'selected'/(arm+'.eqx'))==freeze['choices'][arm]['checkpoint_sha256']==staticfreeze['neural_checkpoints'][arm],arm)
 metrics=read(e/'metrics.json')
 check('Cross-evaluation covers all domains and hedgers',set(metrics)=={'easy','ordinary','hard','base'} and all(set(rows)=={'baseline','easy','hard','unhedged','replenish_1.0'} for rows in metrics.values()),str(e/'metrics.json'))
 for d in ('easy','ordinary','hard'):
  for n in ('baseline','easy','hard','unhedged','replenish_1.0'):
   check('8192 fresh episodes and complete raw daily batches',metrics[d][n]['n']==8192 and len(list(e.glob(f'{d}-{n}-[0-9][0-9][0-9][0-9][0-9].npz')))==64,d+'/'+n)
 check('Neural vs tuned static comparison on training distribution',(e/'in_distribution_baseline.json').is_file(),str(e/'in_distribution_baseline.json'))
 check('Specialization and residual ordering reported',(e/'learned_gate.json').is_file() and (e/'paired_gains.json').is_file(),str(e))
 check('Markdown LaTeX and figures exported',all((e/f'report.{ext}').is_file() for ext in ('md','tex')) and len(list((e/'figures').glob('*.png')))==9,str(e))
matched_runs=[p for p in (c/'launchers/matched/runs').glob('*') if (p/'outputs/complete.json').exists()]
cancelled=(c/'matched_cancelled.json').exists() and read(c/'matched_cancelled.json')['shutdown_confirmed']
check('Additional common-rate check completed or explicitly cancelled by user',len(matched_runs)==1 or cancelled,'matched_cancelled.json' if cancelled else 'launchers/matched/runs')
if len(matched_runs)==1:
 r=matched_runs[0];o=r/'outputs';m=o/'matched';freeze=read(c/'matched_freeze.json')
 check('Matched rate fixed by baseline development choice',freeze['common_learning_rate']==read(c/'frozen_selection.json')['choices']['baseline']['selection']['selected']['lr'],str(c/'matched_freeze.json'))
 check('Final matched export complete',read(r/'sync-status.json')['final_sync_complete'],str(r))
 for row in read(o/'scientific_manifest.json'):
  p=o/row['path'];check('Matched scientific manifest integrity',p.is_file() and p.stat().st_size==row['bytes'] and digest(p)==row['sha256'],str(p))
 for row in read(r/'artifacts.json'):
  p=o/row['path'];check('Matched launcher inventory integrity',p.is_file() and p.stat().st_size==row['bytes'] and digest(p)==row['sha256'],str(p))
 for arm in ('baseline','easy','hard'):
  check('Matched checkpoint hashes',digest(o/'selected'/(arm+'.eqx'))==freeze['choices'][arm]['checkpoint_sha256'],arm)
 a=read(m/'audit.json');check('Matched raw accounting and replay checks',a['passed'] and len(a['checks'])==576,str(m/'audit.json'))
 check('Matched complete entropic matrix and intervals',set(read(m/'metrics.json'))=={'easy','ordinary','hard'} and set(read(m/'paired_gains.json'))=={'easy','hard'},str(m))
b=read(c/'provider_billing.latest.json');check('Provider confirms no active campaign instances',not b['active_campaign_instances'] and not b['unreported_instance_ids'],str(c/'provider_billing.latest.json'))
check('Within fresh campaign budget',b['total_reported_usd']<=read(c/'ledger.json')['total_cap_usd'],str(c/'ledger.json'))
result=dict(passed=all(x['passed'] for x in checks),check_count=len(checks),checks=checks)
(c/'artifact_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(passed=result['passed'],check_count=len(checks),failures=[x for x in checks if not x['passed']])))
if not result['passed']:raise SystemExit(1)
