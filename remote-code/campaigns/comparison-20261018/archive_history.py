"""Lossless portable selfplay-history archives, executed only on GPU workers."""
import hashlib
import json
import os
from pathlib import Path
import tarfile


def digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()


def archive_history(history_dir, destination):
    """Retain every pre-block snapshot required by earlier full checkpoints."""
    history_dir, destination = Path(history_dir), Path(destination)
    files = sorted(history_dir.glob('*.pkl'))
    if not files:
        raise RuntimeError('Selfplay history cache is missing; cannot declare portable recovery')
    inventory = []
    for path in files:
        with path.open('rb') as stream:
            sha = digest(stream)
        inventory.append(dict(name=path.name, bytes=path.stat().st_size, sha256=sha))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.tmp')
    with tarfile.open(temporary, 'w:gz', compresslevel=1) as archive:
        for path in files:
            archive.add(path, arcname=path.name, recursive=False)
    # Verify the complete exported representation, without loading model code.
    with tarfile.open(temporary, 'r:gz') as archive:
        if archive.getnames() != [row['name'] for row in inventory]:
            raise RuntimeError('Historical snapshot archive member mismatch')
        for row in inventory:
            member = archive.getmember(row['name'])
            with archive.extractfile(member) as stream:
                actual = digest(stream)
            if member.size != row['bytes'] or actual != row['sha256']:
                raise RuntimeError('Historical snapshot archive readback mismatch')
    os.replace(temporary, destination)
    with destination.open('rb') as stream:
        archive_sha = digest(stream)
    record = dict(archive=destination.name, archive_bytes=destination.stat().st_size,
                  archive_sha256=archive_sha, snapshot_count=len(files),
                  source_history_path=str(history_dir), snapshots=inventory,
                  restore='Extract into the corresponding JOB_SELFPLAY_HISTORY_DIR '
                          'before restoring the full model/optimizer/teacher/RNG checkpoint')
    manifest = destination.with_name(destination.name + '.manifest.json')
    temporary_manifest = manifest.with_name(manifest.name + '.tmp')
    temporary_manifest.write_text(json.dumps(record, indent=2) + '\n')
    os.replace(temporary_manifest, manifest)
    return record
