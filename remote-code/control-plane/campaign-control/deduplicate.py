"""Dry-run by default: hardlink identical immutable campaign artifacts safely.

No numerical imports, deletions, or scientific source edits. --apply atomically
replaces duplicate paths with hardlinks after size/SHA256 verification. At least
one name always survives. Future writers MUST preserve immutability (new names or
atomic replacement, never in-place writes to a linked artifact). Worker trees are
excluded unless individually opted in and verified terminal by local records and
absence of a matching live process. This does not verify provider billing state.
Queue archives are only reported; this program never removes archives.
"""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import tarfile
import uuid

TERMINAL = {'complete', 'completed', 'failed', 'stopped', 'not_started'}


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def leases(root):
    return {name: lease for phase in read(root / 'queue/state.json').get('phases', {}).values()
            for name, lease in phase.get('leases', {}).items()}


def live_pid(pid):
    if not isinstance(pid, int) or pid <= 0: return False
    try: os.kill(pid, 0)
    except ProcessLookupError: return False
    except PermissionError: return True
    return True


def process_references(path, worker=None):
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid(): continue
        try: command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except (OSError, PermissionError): continue
        if str(path) in command or (worker and '--worker ' + worker + ' ' in command):
            return True
    return False


def terminal_worker(root, name, all_leases):
    directory = root / 'workers' / name
    if Path(name).name != name or directory.is_symlink() or not directory.is_dir(): return False
    if (directory / 'active.json').exists(): return False
    lease = all_leases.get(name)
    if lease and (lease.get('status') not in TERMINAL or live_pid(lease.get('pid'))): return False
    if not lease and not read(directory / 'rental_history.json').get('destroyed_at'): return False
    return not process_references(directory, name)


def regular(path):
    try: return stat.S_ISREG(path.lstat().st_mode) and not any(p.is_symlink() for p in (path, *path.parents))
    except OSError: return False


def fingerprint(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode): raise RuntimeError('Not a regular file: ' + str(path))
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, stat.S_IMODE(info.st_mode))


def digest(path):
    before = fingerprint(path)
    hasher = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        if (os.fstat(stream.fileno()).st_dev, os.fstat(stream.fileno()).st_ino) != before[:2]:
            raise RuntimeError('Artifact changed while opening: ' + str(path))
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): hasher.update(chunk)
    if fingerprint(path) != before: raise RuntimeError('Artifact changed while hashing: ' + str(path))
    return hasher.hexdigest()


def candidates(project, blocked_runs):
    for run in (project / 'runs').glob('*'):
        if not run.is_dir() or run.name in blocked_runs: continue
        if read(run / 'status.json').get('status') not in TERMINAL: continue
        for pattern in ('history/*.eqx', 'checkpoints/*/state.eqx', 'episode_metrics/*.npz'):
            yield from (path for path in run.glob(pattern) if regular(path))
    for pattern in ('data/**/*.npz', 'cache/pricing/*.npz'):
        yield from (path for path in project.glob(pattern) if regular(path))


def archive_report(root, all_leases):
    """A terminal lease and byte-identical extracted members prove regeneration."""
    report = []
    for archive in sorted((root / 'queue').glob('*.tar.gz')):
        if not regular(archive): continue
        name = archive.name.removesuffix('.tar.gz')
        lease = all_leases.get(name)
        record = dict(path=str(archive), bytes=archive.stat().st_size, reclaimable=False)
        if not lease or lease.get('status') not in TERMINAL or live_pid(lease.get('pid')):
            record['reason'] = 'No verified terminal lease; retain archive'
        elif (root / 'workers' / name / 'active.json').exists() or process_references(root / 'workers' / name, name):
            record['reason'] = 'Worker may still be active; retain archive'
        else:
            missing = []
            with tarfile.open(archive, 'r:gz') as source:
                for member in source:
                    if member.isdir(): continue
                    relative = Path(member.name)
                    current = root / relative
                    if relative.is_absolute() or '..' in relative.parts or not member.isfile() or not regular(current):
                        missing.append(member.name); continue
                    if current.stat().st_size != member.size:
                        missing.append(member.name); continue
                    hasher = hashlib.sha256()
                    stream = source.extractfile(member)
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''): hasher.update(chunk)
                    if digest(current) != hasher.hexdigest(): missing.append(member.name)
            record.update(reclaimable=not missing, unmatched_members=missing,
                          reason='Every archive member has an identical current copy; terminal lease' if not missing
                          else 'Some archive members lack identical current copies; retain archive')
        report.append(record)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/jump/deep-hedging'))
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--include-terminal-worker', action='append', default=[])
    parser.add_argument('--inspect-queue-assets', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if root != Path('/home/jump/deep-hedging'):
        parser.error('This campaign helper is restricted to /home/jump/deep-hedging')
    if args.output and (args.output.resolve() == root or root in args.output.resolve().parents):
        parser.error('Write the helper report outside the frozen project tree')
    lock_path = Path(__file__).resolve().with_suffix('.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        all_leases = leases(root)
        blocked = {run for lease in all_leases.values()
                   if lease.get('status') not in TERMINAL or live_pid(lease.get('pid'))
                   for run in lease.get('run_ids', [])}
        projects = [root]
        for worker in args.include_terminal_worker:
            if not terminal_worker(root, worker, all_leases):
                raise RuntimeError('Worker not proven terminal and quiescent: ' + worker)
            projects.extend((root / 'workers' / worker / 'runs').glob('*/outputs/project'))
        buckets = defaultdict(list)
        for project in projects:
            for path in candidates(project, blocked):
                info = path.stat()
                buckets[(info.st_dev, info.st_size, stat.S_IMODE(info.st_mode), info.st_uid, info.st_gid)].append(path)
        operations, physical_reclaimable, shared_skipped = [], 0, 0
        for paths in buckets.values():
            if len({(p.stat().st_dev, p.stat().st_ino) for p in paths}) < 2: continue
            hashes = defaultdict(list)
            for path in paths: hashes[digest(path)].append(path)
            for sha256, identical in hashes.items():
                if len(identical) < 2: continue
                canonical = identical[0]
                original_inodes = defaultdict(list)
                for path in identical: original_inodes[(path.stat().st_dev, path.stat().st_ino)].append(path)
                canonical_inode = (canonical.stat().st_dev, canonical.stat().st_ino)
                for inode, names in original_inodes.items():
                    if inode == canonical_inode:
                        shared_skipped += len(names) - 1
                        continue
                    old_info = names[0].stat()
                    if old_info.st_nlink == len(names): physical_reclaimable += old_info.st_blocks * 512
                    for duplicate in names:
                        operation = dict(source=str(canonical), destination=str(duplicate), size=old_info.st_size,
                                         sha256=sha256, applied=False)
                        if args.apply:
                            # Revalidate the authority/liveness checks immediately before mutation.
                            current_leases = leases(root)
                            if any(path.is_relative_to(root / 'workers' / worker) and
                                   not terminal_worker(root, worker, current_leases)
                                   for path in (canonical, duplicate) for worker in args.include_terminal_worker):
                                raise RuntimeError('Worker became active during deduplication')
                            if any(path.is_relative_to(root / 'runs' / run) for path in (canonical, duplicate) for lease in current_leases.values()
                                   if lease.get('status') not in TERMINAL or live_pid(lease.get('pid'))
                                   for run in lease.get('run_ids', [])):
                                raise RuntimeError('Run became active during deduplication')
                            if not regular(canonical) or not regular(duplicate): raise RuntimeError('Artifact path changed')
                            if digest(canonical) != sha256 or digest(duplicate) != sha256: raise RuntimeError('Artifact content changed')
                            expected = fingerprint(duplicate)
                            temporary = duplicate.parent / ('.dedup-' + uuid.uuid4().hex)
                            try:
                                os.link(canonical, temporary, follow_symlinks=False)
                                if not regular(temporary) or digest(temporary) != sha256 or fingerprint(duplicate) != expected:
                                    raise RuntimeError('Artifact changed before publication')
                                os.replace(temporary, duplicate)
                                fd = os.open(duplicate.parent, os.O_RDONLY)
                                try: os.fsync(fd)
                                finally: os.close(fd)
                            finally:
                                if temporary.exists(): temporary.unlink()
                            operation['applied'] = True
                        operations.append(operation)
        report = dict(mode='apply' if args.apply else 'dry_run', operations=operations,
                      estimated_reclaimable_allocated_bytes=physical_reclaimable,
                      already_shared_names_skipped=shared_skipped,
                      excluded_active_run_ids=sorted(blocked), opted_in_terminal_workers=args.include_terminal_worker,
                      queue_assets=archive_report(root, all_leases) if args.inspect_queue_assets else [],
                      warning='Linked scientific artifacts must remain immutable; never overwrite them in place')
        output = json.dumps(report, indent=2) + '\n'
        if args.output: args.output.write_text(output)
        print(json.dumps({key: value for key, value in report.items() if key != 'operations'}, indent=2))
        print('Duplicate path replacements:', len(operations))


if __name__ == '__main__':
    main()
