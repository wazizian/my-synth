"""Operational authorization scoped only to the newly approved option campaign."""
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path('/home/jump/deep-hedging/campaigns/comparison-20261018')
AUTHORIZATION = ROOT / 'campaign-extension.json'
LEGACY_DEADLINE_UTC = '2026-10-19T06:05:00+00:00'
EXTENSION_NAMESPACE = ROOT / 'option-exploratory-20261019'


def _timestamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RuntimeError('Operational deadline requires an explicit timezone')
    return parsed.timestamp()


def authorized_namespace():
    if not AUTHORIZATION.exists():
        return None
    record = json.loads(AUTHORIZATION.read_text())
    if record.get('status') != 'authorized':
        return None
    namespace = Path(record['namespace']).resolve()
    if namespace != EXTENSION_NAMESPACE.resolve():
        raise RuntimeError('Extension authorization must name the new option namespace only')
    _timestamp(record['deadline_utc'])
    return namespace


def deadline_utc_for(path):
    path = Path(path).resolve()
    namespace = authorized_namespace()
    if namespace is not None and (path == namespace or namespace in path.parents):
        record = json.loads(AUTHORIZATION.read_text())
        return datetime.fromtimestamp(_timestamp(record['deadline_utc']), timezone.utc).isoformat()
    return LEGACY_DEADLINE_UTC


def deadline_for(path):
    return _timestamp(deadline_utc_for(path))


def assert_design_deadline(namespace, supplied_deadline):
    if _timestamp(supplied_deadline) != deadline_for(namespace):
        raise RuntimeError('Design deadline differs from the scoped operational authorization')
