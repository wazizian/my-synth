"""Patch fresh copied managed launchers for verified export and safe cleanup."""
from pathlib import Path
import argparse,shutil

def harden(folder):
 folder=Path(folder)
 if (folder/'active.json').exists():raise RuntimeError('Refuse to patch an active launcher')
 source=Path(__file__).resolve().parent
 for name in ('export_safety.py','controller_handoff.py','external_artifacts.py','external_artifact_worker.py'):
  shutil.copy2(source/name,folder/name)
 p=folder/'smoke.py';s=p.read_text()
 if '# VERIFIED_EXPORT_GUARD' not in s:
  begin=s.index('    # Independent local watchdog survives termination of this foreground process.')
  end=s.index('    try:\n        created =',begin)
  s=s[:begin]+'    # Systemd global watchdog owns deadlines; no unverified local cleanup child.\n'+s[end:]
  s=s.replace('def cleanup(expected_label=None):','def _unsafe_cleanup(expected_label=None):',1)
  marker='\ndef run(args, workload=None):'
  guard='''\n# VERIFIED_EXPORT_GUARD
def cleanup(expected_label=None):
    from export_safety import guarded_cleanup
    return guarded_cleanup(sys.modules[__name__], expected_label)
'''
  s=s.replace(marker,guard+marker,1);p.write_text(s)
 p=folder/'job.py';s=p.read_text()
 if '# VERIFIED_EXPORT_GUARD' not in s:
  s=s.replace('import smoke\n','import smoke\nfrom export_safety import verify_export\n# VERIFIED_EXPORT_GUARD\n',1)
  s=s.replace("'instance': json.loads(smoke.STATE.read_text()), 'ssh': ssh}","'instance': json.loads(smoke.STATE.read_text()), 'ssh': ssh, 'required_artifacts': args.require_artifact}")
  marker="                remote('cd /workspace/job; nohup setsid bash supervise-workload.sh"
  if marker not in s:raise RuntimeError('Expected detached workload launch not found')
  s=s.replace(marker,"                (run_dir / 'workload-launch-intent.json').write_text(json.dumps({'instance': json.loads(smoke.STATE.read_text()), 'time': time.time()}))\n"+marker,1)
  marker="                            (run_dir / 'sync-status.json').write_text(json.dumps(sync_status, indent=2))\n                            break"
  if marker not in s:raise RuntimeError('Expected final sync hook not found')
  s=s.replace(marker,"                            (run_dir / 'sync-status.json').write_text(json.dumps(sync_status, indent=2))\n                            required = args.require_artifact or (['metrics/train.jsonl', 'checkpoints/step-000100.eqx'] if args.job.resolve() == (ROOT / 'example_jax.py').resolve() else [])\n                            verify_export(run_dir, ssh, required)\n                            break",1)
  s=s.replace('instance cleanup will proceed.','worker disk will be preserved unless export verification succeeds.')
  p.write_text(s)
 usage=folder/'USAGE.txt'
 text='\nVerified export hardening: systemd global watchdog owns deadlines; cleanup requires final sync plus matching remote/local SHA256 inventory and required artifacts. Incomplete external selfplay history or failed export stops the instance and preserves disk with deadline-alert.json.\n'
 if text not in usage.read_text():usage.write_text(usage.read_text()+text)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);args=parser.parse_args();harden(args.folder)
