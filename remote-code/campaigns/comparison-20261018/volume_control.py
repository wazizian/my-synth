"""Lightweight provider orchestration; private credentials never leave control plane."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, '/home/jump/vast-tools')
import smoke


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('search', 'list', 'create', 'copy', 'instance', 'cancel'))
    parser.add_argument('--offer', type=int)
    parser.add_argument('--size', type=int, default=120)
    parser.add_argument('--instance', type=int)
    parser.add_argument('--source')
    parser.add_argument('--destination')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'search':
        result = smoke.api('POST', 'volumes/search/', dict(
            verified={'eq': True}, external={'eq': False}, disk_space={'gte': 40},
            reliability={'gte': 0.99}, duration={'gte': 7776000}, allocated_storage=40,
            order=[['storage_cost', 'asc']], limit=10))
    elif args.action == 'list':
        result = smoke.api('GET', 'volumes?owner=me&type=all_volume')
    elif args.action == 'instance':
        row = next(v for v in smoke.list_instances() if v['id'] == args.instance)
        result = {k: v for k, v in row.items() if any(s in k for s in ('status', 'msg', 'copy', 'rsync'))}
    elif args.action == 'cancel':
        if not args.destination or not args.destination.startswith(('C.', 'V.')):
            parser.error('--destination C.id or V.id required')
        result = smoke.api('DELETE', 'commands/copy_direct/', dict(client_id='me', dst_id=args.destination))
    elif args.action == 'create':
        if args.offer != 49989558 or args.size != 120:
            parser.error('Only approved offer49989558, size120GB may be created')
        owned = smoke.api('GET', 'volumes?owner=me&type=all_volume')['volumes']
        existing = [v for v in owned if v.get('label', v.get('name')) == 'deep_hedging_20261018_artifacts']
        if existing:
            result = dict(adopted_existing=existing)
        else:
            args.output.with_suffix('.intent.json').write_text(json.dumps(dict(
                offer=args.offer, size=args.size, name='deep_hedging_20261018_artifacts')))
            result = smoke.api('PUT', 'volumes/', dict(size=args.size, id=args.offer,
                               name='deep_hedging_20261018_artifacts'))
    else:
        if not args.source or not args.destination:
            parser.error('--source and --destination required')
        def endpoint(value):
            identity, path = value.split(':', 1)
            if not identity.startswith(('C.', 'V.')) or not path.startswith('/'):
                parser.error('Copy endpoints must be C.instance:/path or V.volume:/path')
            return identity, path
        source_id, source_path = endpoint(args.source)
        destination_id, destination_path = endpoint(args.destination)
        result = smoke.api('PUT', 'commands/copy_direct/', dict(
            client_id='me', src_id=source_id, dst_id=destination_id,
            src_path=source_path, dst_path=destination_path))
    args.output.write_text(json.dumps(result, indent=2))
    print(args.output)


if __name__ == '__main__':
    main()
