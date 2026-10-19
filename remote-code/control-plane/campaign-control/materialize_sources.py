"""Recover missing source bytes from verified loose Git objects; stdlib only.

Preflights every snapshot before any writes. Existing files must match. Exclusive
creation and no-follow directory descriptors prohibit replacement and symlinks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from audit_recovery import Audit, HEX40, HEX64, canonical_hash, check, read, safe


def parts_for(name):
    check(isinstance(name, str), 'non-string source path')
    parts = name.split('/')
    check(parts and all(p not in ('', '.', '..') for p in parts), 'unsafe source path')
    check(not PurePosixPath(name).is_absolute() and '\\' not in name and '\0' not in name,
          'unsafe source path')
    check(parts[0] != '.git' and name != 'code_provenance.json', 'reserved source path')
    return parts


def recover(root, apply=False):
    audit = Audit(root)
    base = safe(root, 'source_snapshots', directory=True)
    pending, rows = [], []
    for snapshot in sorted(base.iterdir()):
        commit = snapshot.name
        check(bool(HEX40.fullmatch(commit)), 'invalid snapshot directory')
        safe(base, commit, directory=True)
        provenance = read(safe(snapshot, 'code_provenance.json'))
        check(provenance['commit'] == commit, 'commit provenance mismatch')
        files = provenance['files']
        check(files and canonical_hash(files) == provenance['tracked_source_sha256'], 'manifest hash mismatch')
        runtime = {n: h for n, h in files.items() if n.split('/')[0] in ('src', 'scripts', 'configs')}
        check(canonical_hash(runtime) == provenance['source_hash'], 'runtime hash mismatch')
        payload = audit.git_object(snapshot, commit, 'commit')
        tree = payload.split(b'\n', 1)[0]
        check(tree == ('tree ' + provenance['git_tree']).encode('ascii'), 'commit tree mismatch')
        committed = audit.tree_files(snapshot, provenance['git_tree'])
        check(set(committed) == set(files), 'tracked file set mismatch')
        missing = 0
        for name, expected in files.items():
            parts = parts_for(name)
            check(bool(HEX64.fullmatch(expected)), 'invalid SHA256')
            blob = audit.git_object(snapshot, committed[name], 'blob')
            check(hashlib.sha256(blob).hexdigest() == expected, 'Git blob SHA256 mismatch: ' + name)
            target = snapshot.joinpath(*parts)
            for parent in target.parents:
                if parent == snapshot: break
                check(not parent.is_symlink(), 'symlink parent')
                if parent.exists(): check(parent.is_dir(), 'non-directory parent')
            check(not target.is_symlink(), 'symlink target')
            if target.exists():
                check(audit.digest(safe(snapshot, name)) == expected, 'existing source differs: ' + str(target))
            else:
                pending.append((snapshot, parts, blob))
                missing += 1
        rows.append(dict(commit=commit, verified_files=len(files), missing_files=missing,
                         source_hash=provenance['source_hash']))
    if apply:
        for snapshot, parts, blob in pending:
            descriptor = os.open(snapshot, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                for part in parts[:-1]:
                    try: os.mkdir(part, mode=0o755, dir_fd=descriptor)
                    except FileExistsError: pass
                    child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                    os.close(descriptor)
                    descriptor = child
                fd = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o644, dir_fd=descriptor)
                with os.fdopen(fd, 'wb') as stream:
                    stream.write(blob)
                    stream.flush()
                    os.fsync(stream.fileno())
            finally: os.close(descriptor)
        for row in rows:
            check(audit.source(row['commit'])['passed'], 'post-recovery source audit failed')
    return dict(passed=True, applied=apply, recovered_files=len(pending) if apply else 0,
                missing_files=len(pending), snapshots=rows,
                scope='Git SHA1 commit/tree/blob and provenance SHA256 verified; missing regular files only')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/jump/deep-hedging'))
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.absolute(), args.output.absolute()
    check(root not in output.parents, 'report must be outside project')
    result = recover(root, args.apply)
    with output.open('x') as stream: json.dump(result, stream, indent=2)
    print(json.dumps({k: result[k] for k in ('passed', 'applied', 'recovered_files', 'missing_files')}))


if __name__ == '__main__': main()
