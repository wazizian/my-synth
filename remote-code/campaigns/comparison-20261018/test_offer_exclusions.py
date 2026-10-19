"""GPU-worker-only actual helper, launcher-command and budget projection proof."""
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace

import offer_exclusions as exclusion

ROOT = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        registry = base / 'registry.json'
        registry.write_text('[1234567]')
        exclusion.excluded_offer_ids.__defaults__ = (registry,)
        assert exclusion.excluded_offer_ids() == exclusion.KNOWN | {1234567}
        registry.write_text('[7654321]')
        assert 7654321 in exclusion.excluded_offer_ids() and 1234567 not in exclusion.excluded_offer_ids()
        for invalid in ('[true]', '["123"]', '{}', '[0]', '[-1]', 'broken'):
            registry.write_text(invalid)
            try:
                exclusion.excluded_offer_ids()
            except (RuntimeError, ValueError):
                pass
            else:
                raise AssertionError('Malformed exclusion registry was accepted: ' + invalid)
        registry.unlink()
        try:
            exclusion.excluded_offer_ids()
        except FileNotFoundError:
            pass
        else:
            raise AssertionError('Missing registry did not fail closed')
        registry.write_text('[7654321]')
        wanted = exclusion.KNOWN | {7654321}
        campaign = base / 'campaigns' / 'comparison-20261018'
        campaign.mkdir(parents=True)
        template = campaign.parent / 'credit-ordinary-long-20261018' / 'launchers' / 'baseline'
        template.mkdir(parents=True)
        for name in ('job.py', 'smoke.py', 'requirements-jax.txt', 'example_jax.py', 'USAGE.txt', 'AGENTS.md'):
            (template / name).write_text('fixture')
        admitted = ('controller_handoff.py', 'enable_handoff.py', 'launch_arms.py', 'recover_credit.py',
                    'package_trial.py', 'archive_history.py', 'harden_export.py', 'export_safety.py',
                    'external_artifacts.py', 'external_artifact_worker.py', 'watchdog.py',
                    'external_export_coordinator.py', 'offer_exclusions.py')
        for name in admitted:
            (campaign / name).write_text('fixture')
        (campaign / 'controller_handoff_approval.json').write_text(json.dumps(dict(
            status='GPU_validated', source_hashes={name:hashlib.sha256((campaign/name).read_bytes()).hexdigest() for name in admitted})))
        design = campaign / 'design.json'
        design.write_text(json.dumps(dict(status='frozen_for_training', scale_status='frozen_from_development',
                                         deadline_utc=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=10)).isoformat())))
        launches = []
        namespace = campaign / 'final'
        arm_run = namespace / 'baseline' / 'launcher' / 'runs' / 'fixture'
        class Child:
            pid = 123
            def wait(self):
                if '--controller-handoff' in launches[-1]:
                    arm_run.mkdir(parents=True)
                    (arm_run / 'handoff.accepted.json').write_text('{}')
                return 0
        import subprocess
        old_run, old_popen = subprocess.run, subprocess.Popen
        subprocess.run = lambda *a, **k: SimpleNamespace(returncode=0)
        def popen(command, **kwargs):
            launches.append(command)
            return Child()
        subprocess.Popen = popen
        try:
            launch = load('launch_arms')
            launch.__file__ = str(campaign / 'launch_arms.py')
            sys.argv = ['fixture', '--benchmark', 'credit', '--source', str(base), '--design', str(design),
                        '--bp', '1', '--namespace', str(namespace), '--arm', 'baseline']
            launch.main()
            credit_source = campaign.parent / 'credit-ordinary-long-20261018' / 'source'
            credit_source.mkdir()
            (credit_source / 'fixture.py').write_text('pass')
            (campaign / 'credit_gate.py').write_text('pass')
            for arm in ('baseline', 'easy', 'hard'):
                selected = credit_source.parent / 'launchers' / arm / 'runs' / 'fixture' / 'outputs' / 'cost-1bp' / 'learned'
                selected.mkdir(parents=True)
                (selected / 'selected.eqx').write_bytes(b'x')
                (selected / 'selection.json').write_text('{}')
            sys.argv = ['fixture', '--bp', '1', '--campaign', 'credit-ordinary-long-20261018']
            exec(compile((ROOT/'launch_credit_gates.py').read_text(), 'gate-fixture', 'exec'),
                 dict(__file__=str(campaign/'launch_credit_gates.py'), __name__='gate_fixture'))
        finally:
            subprocess.run, subprocess.Popen = old_run, old_popen
        for command in launches:
            got = {int(command[index+1]) for index,value in enumerate(command) if value == '--exclude-offer'}
            assert got == wanted and {46384002,28780319} <= got
        assert len(launches) == 2
        provider = ModuleType('smoke')
        provider.list_instances = lambda: []
        def api(method, path, body=None):
            if path == 'users/current':
                return dict(credit=100)
            return dict(offers=[dict(id=46384002,dph_total=.01),dict(id=28780319,dph_total=.02),
                                dict(id=7654321,dph_total=.03),dict(id=99999999,dph_total=.45)])
        provider.api = api
        sys.modules['smoke'] = provider
        budget = load('budget_admission')
        budget.ROOT = campaign
        projection = budget.snapshot('credit', campaign/'budget-fixture')
        assert [row['id'] for row in projection['cheapest_qualifying_offers']] == [99999999]
        assert projection['unknown_offer_count_priced_at_cap'] == 7
        assert abs(projection['new_comparison_estimate_usd'] - 1.8*(.45+7*.60)) < 1e-9
    print('offer exclusion proof passed: strict registry/fresh reads; actual training and gate '
          'commands include full union; actual budget projection excludes cheap stale listings', flush=True)


if __name__ == '__main__':
    main()
