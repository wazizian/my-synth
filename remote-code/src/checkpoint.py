"""Atomic immutable Equinox checkpoints and durable, replaceable named pointers.

The caller rebuilds a matching skeleton from explicit JSON configuration. Put all
mutable numerical state (including optimizer moments, PRNG keys and historical
snapshots) in ``state``; put static configuration and artifact hashes in metadata.
One process owns each run directory. Tables/datasets are referenced, not copied.
"""
from pathlib import Path
import hashlib
import json
import os
import tempfile
import uuid

import equinox as eqx
import jax
import numpy as np


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.tmp-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _typed_key(value):
    return isinstance(value, jax.Array) and jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)


def _serialize(stream, value):
    if _typed_key(value):
        np.save(stream, np.asarray(jax.random.key_data(value)), allow_pickle=False)
    else:
        eqx.default_serialise_filter_spec(stream, value)


def _deserialize(stream, value):
    if _typed_key(value):
        return jax.random.wrap_key_data(np.load(stream, allow_pickle=False), impl=jax.random.key_impl(value))
    return eqx.default_deserialise_filter_spec(stream, value)


def save_checkpoint(directory, state, metadata, *, tag='latest'):
    """Publish a complete bundle, then atomically point ``tag.json`` to it.

    ``latest`` preserves the prior valid pointer as ``previous.json``. All bundle
    names are immutable, so best/final/latest can safely share retained files.
    """
    if tag not in {'latest', 'best', 'final'}:
        raise ValueError('tag must be latest, best or final')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_id = uuid.uuid4().hex
    staging = Path(tempfile.mkdtemp(prefix='.checkpoint.tmp-', dir=directory))
    with open(staging / 'state.eqx', 'wb') as stream:
        eqx.tree_serialise_leaves(stream, state, filter_spec=_serialize)
        stream.flush()
        os.fsync(stream.fileno())
    record = dict(metadata, checkpoint_id=checkpoint_id, format_version=1,
                  state_sha256=file_hash(staging / 'state.eqx'))
    atomic_json(staging / 'metadata.json', record)
    destination = directory / checkpoint_id
    os.rename(staging, destination)
    _sync_directory(directory)
    pointer = {'checkpoint_id': checkpoint_id, 'path': checkpoint_id,
               'state_sha256': record['state_sha256'],
               'metadata_sha256': file_hash(destination / 'metadata.json')}
    if tag == 'latest' and (directory / 'latest.json').exists():
        previous = json.loads((directory / 'latest.json').read_text())
        # Never rotate a corrupt pointer over the last known previous checkpoint.
        old = directory / previous['path']
        if ((old / 'state.eqx').exists() and (old / 'metadata.json').exists()
                and file_hash(old / 'state.eqx') == previous['state_sha256']
                and file_hash(old / 'metadata.json') == previous['metadata_sha256']):
            atomic_json(directory / 'previous.json', previous)
    atomic_json(directory / (tag + '.json'), pointer)
    return dict(record, path=str(destination))


def load_checkpoint(path, skeleton, *, expected_hashes=None):
    """Restore full state, rejecting damaged bytes and mismatched input hashes."""
    path = Path(path)
    if path.is_dir() and not (path / 'state.eqx').exists():
        path = path / 'latest.json'
    pointer = None
    if path.suffix == '.json':
        pointer = json.loads(path.read_text())
        path = path.parent / pointer['path']
    if pointer and file_hash(path / 'metadata.json') != pointer['metadata_sha256']:
        raise ValueError('checkpoint metadata checksum mismatch')
    metadata = json.loads((path / 'metadata.json').read_text())
    if metadata.get('format_version') != 1:
        raise ValueError('unsupported checkpoint format')
    digest = file_hash(path / 'state.eqx')
    if digest != metadata['state_sha256'] or (pointer and digest != pointer['state_sha256']):
        raise ValueError('checkpoint state checksum mismatch')
    for name, expected in (expected_hashes or {}).items():
        if metadata.get('hashes', {}).get(name) != expected:
            raise ValueError('checkpoint input hash mismatch: ' + name)
    state = eqx.tree_deserialise_leaves(path / 'state.eqx', skeleton, filter_spec=_deserialize)
    return state, metadata
