"""Enable acknowledged controller transfer in fresh hardened launchers only."""
from pathlib import Path
import argparse
import shutil


def enable(folder):
    folder = Path(folder)
    if (folder / 'active.json').exists():
        raise RuntimeError('Refuse to patch an active launcher')
    shutil.copy2(Path(__file__).with_name('controller_handoff.py'), folder / 'controller_handoff.py')
    p = folder / 'job.py'
    s = p.read_text()
    if '# CONTROLLER_HANDOFF' in s:
        return
    s = s.replace('import smoke\n', 'import smoke\nimport traceback\nfrom controller_handoff import request_handoff, atomic_json\n# CONTROLLER_HANDOFF\n', 1)
    s = s.replace("    parser.add_argument('job_args'", "    parser.add_argument('--controller-handoff', action='store_true')\n    parser.add_argument('--handoff-deadline-utc')\n    parser.add_argument('job_args'", 1)
    s = s.replace("    args = parser.parse_args()", "    args = parser.parse_args()\n    args.handoff_confirmed = False\n    if args.controller_handoff and not args.handoff_deadline_utc:\n        parser.error('--controller-handoff requires --handoff-deadline-utc')", 1)
    marker = "                deadline = time.monotonic() + args.timeout_seconds + 120"
    if marker not in s:
        raise RuntimeError('Detached supervisor polling hook missing')
    s = s.replace(marker, "                if args.controller_handoff:\n                    stop_sync.set()\n                    if sync_thread:\n                        sync_thread.join(timeout=args.sync_timeout_seconds + 5)\n                        if sync_thread.is_alive():\n                            raise RuntimeError('Cannot transfer while local sync remains active')\n                    request_handoff(run_dir, ssh, args)\n                    args.handoff_confirmed = True\n                    return\n" + marker, 1)
    exception_marker = '            finally:\n                stop_sync.set()'
    if exception_marker not in s:
        raise RuntimeError('Inner workload exception hook missing')
    s = s.replace(exception_marker, "            except BaseException as original_error:\n                atomic_json(run_dir / 'original-launch-error.json', {\n                    'type': type(original_error).__name__, 'error': str(original_error),\n                    'traceback': traceback.format_exc(),\n                })\n                raise\n" + exception_marker, 1)
    begin = s.index('                # Retry final metrics/checkpoint sync before instance destruction.')
    end = s.index("        print((run_dir / 'console.log')", begin)
    s = s[:begin] + '                if not args.handoff_confirmed:\n' + ''.join('    ' + line if line.strip() else line for line in s[begin:end].splitlines(True)) + s[end:]
    p.write_text(s)
    p = folder / 'smoke.py'
    s = p.read_text()
    marker = '    finally:\n        cleanup(label)'
    if marker not in s:
        raise RuntimeError('Cleanup guard hook missing')
    s = s.replace(marker, "    finally:\n        if not getattr(args, 'handoff_confirmed', False):\n            cleanup(label)", 1)
    p.write_text(s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    enable(parser.parse_args().folder)
