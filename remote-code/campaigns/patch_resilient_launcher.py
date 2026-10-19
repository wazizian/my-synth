"""Patch campaign launchers: remote jobs survive an SSH polling disconnect."""
from pathlib import Path
import sys
old="                remote('bash -c ' + shlex.quote(command), args.timeout_seconds, stdout=log, stderr=subprocess.STDOUT)"
new='''                # The numerical job owns file-backed stdout, independent of SSH.
                # Launch at most once; retries only poll its durable exit marker.
                upload_script = command.encode()
                remote('cat > /workspace/job/workload.sh', 120, input=upload_script)
                wrapper = ('cd /workspace/job\\n'
                           'set +e\\n'
                           'timeout ' + str(args.timeout_seconds) + 's bash workload.sh > outputs/workload.log 2>&1\\n'
                           'status=$?\\n'
                           'echo "$status" > outputs/workload.exit.tmp\\n'
                           'mv outputs/workload.exit.tmp outputs/workload.exit\\n')
                remote('cat > /workspace/job/supervise-workload.sh', 120, input=wrapper.encode())
                remote('cd /workspace/job; nohup setsid bash supervise-workload.sh </dev/null >/workspace/job/outputs/remote-supervisor.log 2>&1 &',
                       120, stdout=log, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + args.timeout_seconds + 120
                while True:
                    if time.monotonic() > deadline:
                        raise TimeoutError('Detached workload exceeded configured runtime')
                    try:
                        result = remote('if test -f /workspace/job/outputs/workload.exit; then cat /workspace/job/outputs/workload.exit; else echo RUNNING; fi',
                                        min(45, max(1, deadline-time.monotonic())), capture_output=True, text=True)
                        status = result.stdout.strip()
                        if status != 'RUNNING':
                            if not status.isdigit():
                                raise RuntimeError('Invalid remote workload exit marker: ' + repr(status))
                            if int(status):
                                raise RuntimeError('Numerical workload exited ' + status + '; inspect outputs/workload.log')
                            break
                    except (OSError, subprocess.SubprocessError) as exc:
                        print('SSH poll interrupted; detached job continues: ' + str(exc), file=sys.stderr, flush=True)
                    time.sleep(15)
'''
for arg in sys.argv[1:]:
 p=Path(arg);s=p.read_text()
 if 'remote-supervisor.log' in s:continue
 assert old in s,p
 s=s.replace('import uuid','import uuid\nimport time').replace(old,new)
 compile(s,str(p),'exec');p.write_text(s)
 u=p.parent/'USAGE.txt'
 with u.open('a') as f:f.write('\n2026-10-18: campaign-only resilience change: numerical workload detached with nohup/setsid, file-backed stdout and atomic exit marker, remote timeout unchanged. SSH polling retries do not restart work. Existing output sync, final inventory and API-confirmed cleanup remain. Validate with real GPU preflight on each resumed job.\n')
 print(p)
