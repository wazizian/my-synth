"""Worker-only immutable-file staging and fresh external-storage readback checks."""
import hashlib
import errno
import json
import os
from pathlib import Path
import sys


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    request = json.loads(sys.argv[1])
    output = Path('/workspace/job/outputs')
    stage = Path(request['stage'])
    readback = Path(request['readback'])
    if not str(stage).startswith('/workspace/external-artifact-stage-') or not str(readback).startswith('/workspace/external-artifact-readback-'):
        raise RuntimeError('Invalid private staging namespace')
    rows = request['rows']
    for row in rows:
        if Path(row['object_name']).name != row['object_name'] or row['object_name'] in ('.', '..'):
            raise RuntimeError('Invalid flat volume object name')
    if request['action'] == 'prepare':
        stage.mkdir(exist_ok=True)
        # Do not stat the future readback name: a negative overlay dentry can
        # become ESTALE when the provider writes it from the host namespace.
        for row in rows:
            rel = Path(row['path'])
            if rel.is_absolute() or '..' in rel.parts:
                raise RuntimeError('Unsafe output-relative artifact path')
            source = output / rel
            if source.is_symlink() or not source.is_file():
                raise RuntimeError('Artifact must be a regular immutable output')
            if source.stat().st_size != row['bytes'] or digest(source) != row['sha256']:
                raise RuntimeError('Source artifact changed before staging')
            target = stage / row['object_name']
            if target.exists():
                if not os.path.samefile(source, target):
                    raise RuntimeError('Unexpected staging collision')
            else:
                os.link(source, target)
        (stage / 'manifest.json').write_text(json.dumps(rows, sort_keys=True, indent=2))
        print(json.dumps(dict(prepared=True, staging='hardlinks', rows=rows)))
    elif request['action'] == 'verify':
        # Enumerate existing names first, never look up a missing readback path.
        if readback.name not in os.listdir(readback.parent):
            print(json.dumps(dict(verified=False, pending=True)))
            return
        names = set(os.listdir(readback))
        if not {'manifest.json', *(row['object_name'] for row in rows)} <= names:
            print(json.dumps(dict(verified=False, pending=True)))
            return
        for row in rows:
            restored = readback / row['object_name']
            source = output / row['path']
            if restored.is_symlink() or not restored.is_file() or os.path.samefile(restored, source):
                raise RuntimeError('Readback must be an independent regular file')
            if restored.stat().st_size != row['bytes'] or digest(restored) != row['sha256']:
                raise RuntimeError('External artifact readback SHA256 mismatch')
        restored_manifest = json.loads((readback / 'manifest.json').read_text())
        if restored_manifest != rows:
            raise RuntimeError('External readback manifest mismatch')
        print(json.dumps(dict(verified=True, rows=rows, readback=str(readback))))
    else:
        raise RuntimeError('Unknown storage proof action')


if __name__ == '__main__':
    try:
        main()
    except OSError as exc:
        if exc.errno == errno.ESTALE:
            print(json.dumps(dict(verified=False, pending=True, stale_readback=True)))
        else:
            raise
