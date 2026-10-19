"""GPU-worker-only stdlib fixtures for scoped operational deadline controls."""
import ast
from datetime import datetime, timezone
import datetime as datetime_module
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
provider_calls = []
provider = []

def api(method, path, body=None):
    provider_calls.append((method, path))
    if (method, path) == ('GET', 'users/current'):
        return {'credit': 100., 'balance': 0.}
    if (method, path) == ('POST', 'bundles/'):
        return {'offers': [{'id': index, 'dph_total': .45} for index in range(1, 10)]}
    raise AssertionError('No mutation permitted in control fixture')

sys.modules['smoke'] = SimpleNamespace(api=api, list_instances=lambda: provider)
sys.modules['offer_exclusions'] = SimpleNamespace(excluded_offer_ids=lambda: set())
import campaign_deadlines as deadlines
import budget_admission as budget

def expect_error(action):
    try:
        action()
    except (RuntimeError, ValueError, KeyError):
        return
    raise AssertionError('Expected fail-closed error')

def assigned(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)

def execute(nodes, namespace):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), '<actual-controller-block>', 'exec'), namespace)

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    namespace = root / 'option-exploratory-20261019'
    auth = root / 'campaign-extension.json'
    record = dict(status='authorized', namespace=str(namespace), deadline_utc='2026-10-19T18:00:00+00:00')
    auth.write_text(json.dumps(record))
    deadlines.ROOT = root
    deadlines.AUTHORIZATION = auth
    deadlines.EXTENSION_NAMESPACE = namespace
    legacy = datetime(2026, 10, 19, 6, 5, tzinfo=timezone.utc).timestamp()
    extended = datetime(2026, 10, 19, 18, tzinfo=timezone.utc).timestamp()
    now = datetime(2026, 10, 19, 12, tzinfo=timezone.utc).timestamp()
    assert deadlines.deadline_for(namespace) == extended
    assert deadlines.deadline_for(namespace / 'selfplay/launcher') == extended
    assert deadlines.deadline_for(root / 'credit-final/baseline/launcher') == legacy
    assert deadlines.deadline_for(root / 'option-final/evaluation/launcher') == legacy
    assert deadlines.deadline_for(root / 'option-exploratory-20261019-evil') == legacy
    deadlines.assert_design_deadline(namespace, '2026-10-19T18:00:00Z')
    expect_error(lambda: deadlines.assert_design_deadline(namespace, '2026-10-19T19:00:00Z'))
    expect_error(lambda: deadlines.assert_design_deadline(root / 'credit-final', '2026-10-19T18:00:00Z'))
    invalid = dict(record, namespace=str(root / 'credit-final'))
    auth.write_text(json.dumps(invalid))
    expect_error(lambda: deadlines.deadline_for(namespace))
    auth.write_text(json.dumps(dict(record, deadline_utc='2026-10-19T18:00:00')))
    expect_error(lambda: deadlines.deadline_for(namespace))
    auth.write_text(json.dumps(record))

    # Execute the actual watchdog per-launch deadline block, without its API loop.
    tree = ast.parse((SOURCE / 'watchdog.py').read_text())
    loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == 'launch' and isinstance(node.iter, ast.Call) and getattr(node.iter.func, 'id', '') == 'launches')
    begin = next(index for index, node in enumerate(loop.body) if assigned(node, 'authorization_error'))
    end = next(index for index, node in enumerate(loop.body) if assigned(node, 'status')) + 1
    watchdog_block = loop.body[begin:end]
    def due(path, run=Path('20261019T120000Z-fixture')):
        env = dict(deadline_for=deadlines.deadline_for, DEADLINE=legacy, launch=path, run=run,
                   meta={'timeout_seconds': 12000}, datetime=datetime_module,
                   state=SimpleNamespace(stat=lambda: SimpleNamespace(st_mtime=now)),
                   saved={'instance_id': 1, 'label': 'fixture'})
        execute(watchdog_block, env)
        return env
    assert due(namespace / 'baseline/launcher')['due'] == now + 13200
    assert due(root / 'credit-final/baseline/launcher')['due'] == legacy
    assert due(namespace / 'baseline/launcher', None)['due'] == now + 1200
    auth.write_text('{malformed')
    failed = due(namespace / 'baseline/launcher')
    assert failed['due'] == legacy and failed['authorization_error']
    auth.write_text(json.dumps(record))

    # Actual allocator assertion/deadline/time-budget block; no rental subprocess.
    tree = ast.parse((SOURCE / 'launch_arms.py').read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    begin = next(index for index, node in enumerate(main.body) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and getattr(node.value.func, 'id', '') == 'assert_design_deadline')
    end = next(index for index, node in enumerate(main.body) if assigned(node, 'timeout')) + 1
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 19, 12, tzinfo=timezone.utc)
    def allocator(path, value):
        env = dict(assert_design_deadline=deadlines.assert_design_deadline,
                   args=SimpleNamespace(namespace=path), design={'deadline_utc': value},
                   datetime=FixedDatetime, timezone=timezone)
        execute(main.body[begin:end], env)
        return env
    assert allocator(namespace, record['deadline_utc'])['timeout'] == 12000
    expect_error(lambda: allocator(namespace, '2026-10-19T19:00:00Z'))
    expect_error(lambda: allocator(root / 'credit-final', deadlines.LEGACY_DEADLINE_UTC))

    # Actual admission scan must count active arms in the newly authorized namespace.
    budget.ROOT = root
    original_time = budget.time.time
    original_sleep = budget.time.sleep
    budget.time.time = lambda: now
    budget.time.sleep = lambda _: (_ for _ in ()).throw(AssertionError('No waits in control fixture'))
    launch = namespace / 'baseline/launcher'
    run = launch / 'runs/20261019T120000Z-fixture'
    run.mkdir(parents=True)
    (launch / 'active.json').write_text(json.dumps({'instance_id': 99, 'label': 'fixture'}))
    (run / 'metadata.json').write_text(json.dumps({'instance': {'instance_id': 99, 'label': 'fixture'}}))
    provider.append({'id': 99, 'label': 'fixture', 'actual_status': 'running', 'dph_total': .45})
    snapshot = budget.snapshot('option', namespace)
    assert snapshot['admissible'] and snapshot['authorized_deadline_utc'] == record['deadline_utc']
    assert any(row['instance_id'] == 99 and row['namespace'] == namespace.name for row in snapshot['already_committed_comparisons'])
    assert 'baseline' not in snapshot['new_arms']
    before = len(provider_calls)
    expect_error(lambda: budget.wait_for_admission('option', root / 'credit-final'))
    assert len(provider_calls) == before
    budget.time.time = lambda: extended - 8999
    expect_error(lambda: budget.wait_for_admission('option', namespace))
    assert len(provider_calls) == before
    budget.time.time = original_time
    budget.time.sleep = original_sleep

result = dict(status='passed', scope='GPU-host stdlib control fixtures only; no rentals or scientific imports',
              checks=['exact-authorized-namespace-only', 'legacy-credit-and-option-deadline-preserved',
                      'timezone-and-root-scope-validation', 'actual-watchdog-runtime-and-boot-bounds',
                      'malformed-authorization-failclosed', 'actual-allocator-design-consistency-and-time-cap',
                      'actual-budget-scan-counts-new-namespace', 'expired-late-admission-before-provider-query'],
              source_hashes={name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                             for name in ('campaign_deadlines.py', 'watchdog.py', 'budget_admission.py', 'launch_arms.py')})
print(json.dumps(result, indent=2))
(SOURCE / 'deadline-control-validation.json').write_text(json.dumps(result, indent=2))
