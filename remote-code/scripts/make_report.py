"""Post-selection reporting and artifact export. Execute only on a GPU worker."""
import argparse
import csv
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_artifacts import ARMS, DOMAINS, audit, read_json, read_events, expand_event, selected_runs


def csv_write(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def active_time(row):
    for name in ('active_training_seconds', 'active_seconds', 'training_seconds', 'cumulative_active_training_seconds'):
        if name in row: return row[name]
    times = row.get('timings', row.get('runtime', {}))
    return times.get('active_training_seconds', times.get('training')) if isinstance(times, dict) else None


def validate_episode_event(root, run_dir, event, datasets, scales):
    """GPU recomputation binds each successful domain event to retained raw bytes."""
    import jax.numpy as jnp
    import numpy as np
    from src.metrics import summarize_outcomes
    domain, role = event.get('domain'), event.get('split_role')
    scope = dict(event_id=event.get('event_id'), checkpoint_id=event.get('checkpoint_id'),
                 run_id=run_dir.name, domain=domain, split_role=role)
    if event.get('raw_mse') is None:
        reason = event.get('reason') or event.get('error')
        if not reason and event.get('status') == 'nonfinite_or_empty':
            reason = 'nonfinite_or_empty: sample_count=' + str(event.get('sample_count')) + ', nonfinite_count=' + str(event.get('nonfinite_count'))
        if not reason:
            return dict(scope, status='invalid', reason='Null outcome lacks a documented failure reason')
        return dict(scope, status='documented_failure', reason=str(reason))
    try:
        if event.get('status') != 'ok': raise ValueError('Non-null outcome is not marked ok')
        dataset = datasets[role + '_' + domain]
        count = dataset['count']
        if event.get('sample_count') != count or event.get('finite_count') != count:
            raise ValueError('Event sample/finite count differs from immutable dataset count')
        if event.get('data_hash') != dataset['sha256']:
            raise ValueError('Event dataset hash differs from immutable manifest')
        if event.get('reference_scale') != scales[domain]:
            raise ValueError('Event reference scale differs from frozen scale')
        relative = Path(event['episode_metrics_path'])
        if relative.is_absolute() or '..' in relative.parts: raise ValueError('Unsafe raw artifact path')
        path = root / relative
        if not path.is_relative_to(run_dir / 'episode_metrics'):
            raise ValueError('Raw artifact is outside the event run episode directory')
        if any(parent.is_symlink() for parent in (path, *path.parents)):
            raise ValueError('Symlinked raw artifact is prohibited')
        required = {'shortfall': (count,), 'pnl': (count,), 'costs': (count, 2),
                    'turnover': (count, 2), 'liquidation_costs': (count, 2),
                    'mean_abs_holdings': (count, 2), 'saturation_fraction': (count,)}
        with np.load(path, allow_pickle=False) as archive:
            missing = set(required) - set(archive.files)
            if missing: raise ValueError('Missing required raw arrays: ' + ','.join(sorted(missing)))
            arrays = {}
            for name in archive.files:
                value = jnp.asarray(archive[name])
                if value.ndim == 0 or value.shape[0] != count:
                    raise ValueError('Raw array episode count mismatch: ' + name)
                if name in required and value.shape != required[name]:
                    raise ValueError('Raw shape mismatch: ' + name)
                if not bool(jnp.all(jnp.isfinite(value))):
                    raise ValueError('Nonfinite raw values in successful event: ' + name)
                arrays[name] = value
        if not bool(jnp.array_equal(arrays['pnl'], -arrays['shortfall'])):
            raise ValueError('Raw PnL is not negative normalized shortfall')
        actual = summarize_outcomes(arrays['shortfall'], arrays, scale=scales[domain],
                                    domain=domain, split_role=role, data_hash=dataset['sha256'],
                                    checkpoint_id=event['checkpoint_id'])
        fields = ('raw_mse', 'scaled_mse', 'mean_shortfall', 'mean_pnl', 'shortfall_std',
                  'mean_absolute_error', 'var95', 'es95', 'turnover', 'costs',
                  'liquidation_costs', 'mean_abs_holdings', 'saturation_fraction', 'total_costs')
        for name in fields:
            if event.get(name) is None: raise ValueError('Successful summary lacks field: ' + name)
            expected, computed = jnp.asarray(event[name]), jnp.asarray(actual[name])
            if expected.shape != computed.shape or not bool(jnp.all(jnp.isclose(expected, computed, rtol=1e-5, atol=1e-9))):
                raise ValueError('Raw arrays disagree with saved summary: ' + name)
        return dict(scope, status='verified', episode_metrics_path=str(relative), sample_count=count,
                    tolerance={'rtol': 1e-5, 'atol': 1e-9})
    except Exception as error:
        return dict(scope, status='invalid', reason=str(error), episode_metrics_path=event.get('episode_metrics_path'))


def validate_aggregate_events(events, scales):
    """Aggregate comparisons use only the immediately preceding exact-checkpoint domains."""
    import jax.numpy as jnp
    checked, pending = [], {}
    for event in events:
        key = (event.get('run_id'), event.get('split_role'), event.get('checkpoint_id'))
        if event.get('domain') in DOMAINS:
            pending.setdefault(key, {})[event['domain']] = event
        elif event.get('event_type') == 'evaluation_aggregate':
            domains = pending.pop(key, {})
            record = dict(run_id=key[0], split_role=key[1], checkpoint_id=key[2])
            try:
                if set(domains) != set(DOMAINS): raise ValueError('Aggregate lacks a complete preceding three-domain event set')
                easy, hard = domains['easy']['raw_mse'], domains['hard']['raw_mse']
                if easy is None or hard is None: raise ValueError('Aggregate includes unavailable outcomes')
                computed = jnp.asarray([.5 * (easy + hard), .5 * (easy / scales['easy'] + hard / scales['hard'])])
                saved = jnp.asarray([event['easy_hard_raw_mean'], event['balanced_objective']])
                if not bool(jnp.all(jnp.isclose(saved, computed, rtol=1e-5, atol=1e-9))):
                    raise ValueError('Aggregate disagrees with exact-checkpoint domain summaries')
                record['status'] = 'verified'
            except Exception as error:
                record.update(status='invalid', reason=str(error))
            checked.append(record)
    for key, domains in pending.items():
        if domains and all(event.get('raw_mse') is not None for event in domains.values()):
            checked.append(dict(run_id=key[0], split_role=key[1], checkpoint_id=key[2],
                                status='invalid', reason='Successful domain events lack a completed aggregate'))
    return checked


def final_pricing_sensitivity(root, selection):
    """Diagnostic only, after selection freeze; never updates configurations/pointers."""
    import hashlib
    import subprocess
    expected = {row.get('run_id', row.get('trial_id')): row.get('checkpoint_id') for row in selection.values()}
    record = dict(status='unavailable', scope='32 fixed monitor episodes per domain; frozen selected seven checkpoints',
                  limitation='Bounded conditional sensitivity, not a full-data or rare-tail pricing-error guarantee')
    if set(selection) != set(ARMS) or None in expected or len(expected) != 7 or any(value is None for value in expected.values()):
        return dict(record, reason='Frozen selection does not identify exactly seven checkpoints')
    manifest = read_json(root / 'data/manifest.json', {})
    pricing_source = hashlib.sha256((root / 'src/pricing.py').read_bytes()).hexdigest()
    def matches(report):
        return (report.get('status') == 'complete' and report.get('episodes_per_domain') == 32
                and report.get('pricing_hash') == manifest.get('pricing_hash')
                and report.get('benchmark_hash') == manifest.get('benchmark_hash')
                and report.get('objective_scales_hash') == manifest.get('objective_scales_hash')
                and report.get('pricing_source_hash') == pricing_source
                and {name: row.get('checkpoint_id') for name, row in report.get('runs', {}).items()} == expected)
    reports = sorted((root / 'results/pricing_sensitivity').glob('*/report.json'))
    reusable = [path for path in reports if matches(read_json(path, {}))]
    if not reusable:
        result = subprocess.run([sys.executable, str(root / 'scripts/pricing_sensitivity.py'),
                                 '--runs', *expected, '--frozen-selection', '--episodes', '32'], cwd=root)
        reports = sorted((root / 'results/pricing_sensitivity').glob('*/report.json'))
        reusable = [path for path in reports if matches(read_json(path, {}))]
        if result.returncode != 0 or not reusable:
            return dict(record, status='failed', reason='Selected pricing sensitivity process failed or produced no matching complete report',
                        returncode=result.returncode)
    path = reusable[-1]
    sensitivity = read_json(path, {})
    for run in sensitivity['runs'].values():
        for domain in DOMAINS:
            outcome = run['domains'][domain]
            artifact = root / outcome['episode_archive']
            if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != outcome['episode_archive_hash']:
                return dict(record, status='failed', reason='Selected sensitivity paired archive missing or checksum mismatch')
    return dict(record, status='complete', report_path=str(path.relative_to(root)),
                gaps=sensitivity.get('gaps'), report_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main(root):
    results = root / 'results'
    figures = results / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    selection_path = results / 'frozen_selection.json'
    if not selection_path.exists():
        raise RuntimeError('Freeze selection before opening final-test reporting')
    selection = selected_runs(read_json(selection_path, {}))
    # Finish the guarded numerical subprocess before this process initializes
    # JAX, avoiding competing GPU preallocation pools on the one report device.
    sensitivity = final_pricing_sensitivity(root, selection)
    (results / 'selected_pricing_sensitivity.json').write_text(json.dumps(sensitivity, indent=2) + '\n')
    from src.runtime import require_gpu
    require_gpu()
    import jax.numpy as jnp
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    events, train_events, all_rows = [], [], []
    archive_validation = []
    raw_event_validation = []
    dataset_manifest = read_json(root / 'data/manifest.json')
    scales = read_json(root / 'objective_scales.json')['scales']
    run_dirs = sorted(path for path in (root / 'runs').glob('*') if path.is_dir())
    with zipfile.ZipFile(results / 'episode_losses.npz', 'w', zipfile.ZIP_DEFLATED) as combined:
        for directory in run_dirs:
            config = read_json(directory / 'config.json', {})
            status = read_json(directory / 'status.json', {})
            method = config.get('method', config.get('arm', directory.name))
            run_events = [dict(row, run_id=directory.name, method=method)
                          for event in read_events(directory / 'evaluation_metrics.jsonl') for row in expand_event(event)]
            events.extend(run_events)
            raw_event_validation.extend(validate_episode_event(root, directory, event, dataset_manifest['datasets'], scales)
                                        for event in run_events if event.get('domain') in DOMAINS
                                        and event.get('split_role') in ('monitor', 'selection', 'final_test'))
            training = [dict(row, run_id=directory.name, method=method)
                        for row in read_events(directory / 'train_metrics.jsonl')]
            train_events.extend(training)
            checkpoints = sorted({event.get('checkpoint_id') for event in run_events
                                  if event.get('split_role') == 'final_test' and event.get('checkpoint_id')}
                                 | {path.parent.name for path in (directory / 'checkpoints').glob('*/metadata.json')}) or [None]
            lineage_active = max([active_time(e) for e in training if active_time(e) is not None] or [None])
            for checkpoint_id in checkpoints:
                checkpoint_times = {active_time(e) for e in run_events
                                    if e.get('split_role') == 'final_test'
                                    and e.get('checkpoint_id') == checkpoint_id
                                    and e.get('domain') in DOMAINS and active_time(e) is not None}
                checkpoint_active = next(iter(checkpoint_times)) if len(checkpoint_times) == 1 else None
                row = dict(run_id=directory.name, method=method, checkpoint_id=checkpoint_id,
                           status=status.get('status', status), config=config,
                           parent_id=config.get('parent_id'), active_training_seconds=lineage_active,
                           lineage_active_training_seconds=lineage_active,
                           checkpoint_active_training_seconds=checkpoint_active,
                           checkpoint_active_time_status=('consistent' if len(checkpoint_times) == 1 else
                                                          'missing' if not checkpoint_times else 'inconsistent_evaluation_events'))
                for domain in DOMAINS:
                    matches = [e for e in run_events if e.get('split_role') == 'final_test'
                               and e.get('checkpoint_id') == checkpoint_id and e.get('domain') == domain]
                    if matches:
                        row.update({domain + '_' + key: value for key, value in matches[-1].items()
                                    if key not in ('run_id', 'method', 'domain', 'checkpoint_id')})
                    else: row[domain + '_reason'] = 'No final-test result retained'
                if all(row.get(domain + '_scaled_mse') is not None for domain in ('easy', 'hard')):
                    row['balanced_objective'] = .5 * (row['easy_scaled_mse'] + row['hard_scaled_mse'])
                    row['easy_hard_raw_mean'] = .5 * (row['easy_raw_mse'] + row['hard_raw_mse'])
                all_rows.append(row)
            for path in sorted((directory / 'episode_metrics').rglob('*.npz')):
                record = dict(path=str(path.relative_to(root)), arrays={}, status='inspected')
                try:
                    with np.load(path, allow_pickle=False) as saved:
                        for name in saved.files:
                            array = saved[name]
                            device_array = jnp.asarray(array)
                            record['arrays'][name] = dict(shape=list(array.shape), finite_count=int(jnp.sum(jnp.isfinite(device_array))), size=int(array.size))
                    with zipfile.ZipFile(path) as source:
                        prefix = str(path.relative_to(root / 'runs')).replace('/', '__').removesuffix('.npz')
                        for name in source.namelist():
                            combined.writestr(prefix + '__' + name, source.read(name))
                except Exception as error:
                    record.update(status='invalid', reason='Raw archive unreadable; original remains retained: ' + str(error))
                archive_validation.append(record)
    aggregate_validation = validate_aggregate_events(events, scales)
    validation_failures = [row for row in raw_event_validation + aggregate_validation + archive_validation if row['status'] == 'invalid']
    (results / 'raw_event_validation.json').write_text(json.dumps(dict(
        passed=not validation_failures, events=raw_event_validation, aggregates=aggregate_validation,
        failures=validation_failures), indent=2) + '\n')
    selected = []
    for arm in ARMS:
        chosen = selection.get(arm, {})
        run_id = chosen.get('run_id', chosen.get('trial_id'))
        checkpoint_id = chosen.get('checkpoint_id')
        matches = [row for row in all_rows if row['run_id'] == run_id and
                   (checkpoint_id is None or row['checkpoint_id'] == checkpoint_id)]
        if len(matches) == 1: selected.append(matches[0])
        else: selected.append(dict(method=arm, run_id=run_id, lineage_active_training_seconds=None,
                                  checkpoint_active_training_seconds=None, active_training_seconds=None, status='missing_or_ambiguous_selected_evaluation', reason='Frozen selection must identify one evaluated checkpoint'))
    baseline = next((row for row in selected if row['method'] == 'baseline'), {})
    for row in selected:
        for domain in DOMAINS:
            base, value = baseline.get(domain + '_raw_mse'), row.get(domain + '_raw_mse')
            row[domain + '_improvement_over_baseline'] = base - value if base is not None and value is not None else None
    csv_write(results / 'all_trials.csv', all_rows)
    csv_write(results / 'final_metrics.csv', selected)
    csv_write(results / 'all_events.csv', [dict(e, event_stream='evaluation') for e in events] + [dict(e, event_stream='training') for e in train_events])
    (results / 'array_integrity.json').write_text(json.dumps(archive_validation, indent=2) + '\n')
    selected_ids = {chosen.get('run_id', chosen.get('trial_id')) for chosen in selection.values()}
    for split in ('monitor', 'selection'):
        for domain in DOMAINS:
            for scope in ('selected', 'all_trials'):
                fig, ax = plt.subplots(figsize=(8, 5))
                count = 0
                for directory in run_dirs:
                    if scope == 'selected' and directory.name not in selected_ids:
                        continue
                    rows = sorted((e for e in events if e['run_id'] == directory.name and e.get('split_role') == split
                                   and e.get('domain') == domain and e.get('raw_mse') is not None
                                   and active_time(e) is not None), key=active_time)
                    if rows:
                        ax.plot([active_time(e) / 60 for e in rows], [e['raw_mse'] for e in rows], label=directory.name)
                        count += 1
                ax.set(xlabel='Cumulative active training minutes', ylabel='Normalized shortfall MSE',
                       title=domain + ' ' + split + (' — selected lineages' if scope == 'selected' else ' — all retained trials'))
                if count: ax.legend(fontsize=7 if scope == 'selected' else 5)
                else: ax.text(.5, .5, 'No retained observations', ha='center', transform=ax.transAxes)
                prefix = '' if scope == 'selected' else 'all_trials_'
                fig.tight_layout(); fig.savefig(figures / (prefix + domain + '_' + split + '.png')); plt.close(fig)
    for directory in run_dirs:
        rows = [e for e in train_events if e['run_id'] == directory.name]
        points = []
        for row in rows:
            probs = row.get('generator_probabilities', row.get('probabilities', row.get('q')))
            if probs is None and isinstance(row.get('generator'), dict): probs = row['generator'].get('probabilities')
            if isinstance(probs, list) and len(probs) == 12 and active_time(row) is not None: points.append((active_time(row) / 60, probs))
        if points:
            fig, ax = plt.subplots(figsize=(8, 5))
            for index in range(12): ax.plot([p[0] for p in points], [p[1][index] for p in points], label=str(index))
            ax.set(xlabel='Cumulative active training minutes', ylabel='Regime probability', title=directory.name)
            ax.legend(ncol=4, fontsize=7); fig.tight_layout(); fig.savefig(figures / (directory.name + '_probabilities.png')); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in selected:
        x, y = row.get('easy_raw_mse'), row.get('hard_raw_mse')
        if x is not None and y is not None:
            ax.scatter(x, y); ax.annotate(row['method'], (x, y), fontsize=7)
    ax.set(xlabel='Easy final normalized MSE', ylabel='Hard final normalized MSE', title='Frozen selected checkpoints; lower is better')
    fig.tight_layout(); fig.savefig(figures / 'easy_hard_final.png'); plt.close(fig)
    completeness = audit(root)
    completeness['raw_event_integrity_passed'] = not validation_failures
    completeness['selected_pricing_sensitivity_status'] = sensitivity['status']
    completeness['failures'].extend('raw outcome: ' + row['reason'] for row in validation_failures)
    if sensitivity['status'] != 'complete':
        completeness['failures'].append('selected pricing sensitivity: ' + sensitivity.get('reason', sensitivity['status']))
    for row in selected:
        if any(row.get(domain + '_raw_mse') is None for domain in DOMAINS):
            completeness['failures'].append('selected checkpoint lacks valid three-domain final results: ' + row['method'])
    completeness['complete'] = not completeness['failures']
    (results / 'artifact_audit.json').write_text(json.dumps(completeness, indent=2) + '\n')
    ledger = read_events(root / 'budget/ledger.jsonl')
    from scripts.profile import summarize
    previous_profile = read_json(results / 'profile.json', {})
    previous_runs = {row['run_id']: row for row in previous_profile.get('runs', [])}
    profile = summarize(root)
    for row in profile['runs']:
        previous = previous_runs.get(row['run_id'], {})
        for field in ('peak_sampled_vram_mib', 'vram_sample_count', 'vram_source'):
            if previous.get(field) is not None:
                row[field] = previous[field]
    (results / 'profile.json').write_text(json.dumps(profile, indent=2) + '\n')
    csv_write(results / 'profile.csv', profile['runs'])
    lines = ['# Adaptive deep hedging exploratory campaign', '',
             'Archive status: ' + ('metadata completeness passed.' if completeness['complete'] else 'INCOMPLETE; see artifact_audit.json for every missing requirement.'), '',
             '| Arm | Base MSE | Easy MSE | Hard MSE | Balanced scaled loss | Status |',
             '| --- | ---: | ---: | ---: | ---: | --- |']
    def number(value): return 'missing' if value is None else format(value, '.6g')
    for row in selected:
        lines.append('| ' + ' | '.join([row['method']] + [number(row.get(key)) for key in ('base_raw_mse', 'easy_raw_mse', 'hard_raw_mse', 'balanced_objective')] + [str(row.get('status'))]) + ' |')
    lines += ['', '## Selected-result interpretation', '',
              'PENDING SUPERVISOR INTERPRETATION: After checking the frozen selected results and numerical validity, describe hard-only versus mixed within DPG and SOAR, easy/hard tradeoffs, base retention, intrinsic versus target-guided learning, and whether gains justify measured compute. Include unfavorable or inconclusive outcomes and any incomplete arms. No favorable conclusion is inferred automatically.', '',
              '## Measured compute', '',
              '| Run | Compile s | ms/block | Active paths/s | Sampled VRAM MiB | Allocator peak in-use MiB | Allocator peak pool MiB | Checkpoint/history I/O s |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for measured in profile['runs']:
        lines.append('| ' + ' | '.join([measured['run_id']] + [number(measured.get(key)) for key in
            ('compile_seconds', 'milliseconds_per_block', 'end_to_end_active_paths_per_second',
             'peak_sampled_vram_mib', 'allocator_peak_in_use_mib', 'allocator_peak_pool_mib', 'checkpoint_seconds')]) + ' |')
    lines += ['', 'Teacher time includes student updates; isolated teacher overhead is unmeasured. Evaluation throughput includes summary and archive I/O. Sampled nvidia-smi VRAM values remain lower bounds. Allocator peaks come from device.memory_stats() process-lifetime high-water counters: they include earlier sequential runs and evaluation in the same process, are not isolated per-trial peaks, and do not represent total device VRAM. Unsupported counters remain missing; profile.json retains observation process/device provenance. Profile shapes are recorded in profile.csv; tiny smoke timing does not establish production throughput. Profile-derived block counts and storage projections are estimates. The runner separately enforces immutable selfplay_max_blocks, selfplay_history_limit_bytes (12 GiB default), and selfplay_storage_reserve_bytes (1 GiB default), checking projected storage before training and block/storage availability during execution. Consult each run configuration and stop_reason for the applied limits and any early stop.', '', 'lineage_active_training_seconds records the maximum cumulative active training time observed for the run; active_training_seconds is its backward-compatible alias. checkpoint_active_training_seconds comes from the final-test events for that exact checkpoint and may be earlier than the lineage endpoint. Missing or inconsistent checkpoint timing remains null and is identified by checkpoint_active_time_status. Primary monitor/selection figures show the seven selected lineages; all_trials_<domain>_<split>.png preserves every retained trial curve. The CSV tables retain scaled and raw losses, signed shortfall/PnL, tail outcomes, transaction costs, turnover, positions, sample counts, lineage and available runtime/cost metadata. Missing observations remain missing. Positive domain improvement means baseline MSE minus method MSE; compare methods within each domain. Easy/hard raw levels alone do not establish algorithm quality.', '',
              'Inspect hard-only versus mixed separately within DPG and SOAR, including base retention. Compare intrinsic curriculum to target-guided methods together with their active compute and teacher overhead. A lower hard loss with higher easy loss is specialization, not broad improvement. This report does not infer a favorable result from incomplete data.', '',
              'Selected-checkpoint pricing sensitivity status: ' + sensitivity['status'] + '. See selected_pricing_sensitivity.json and its referenced paired-outcome report. This uses the same 32 fixed monitor episodes per domain for all seven frozen checkpoints; it does not bound pricing error on the full datasets or rare tails.', '',
              '## Limitations', '',
              'One training seed; narrow tuning; synthetic common-support shifts in a finite 12-regime Heston menu. Numerical pricing/interpolation and time discretization are approximations. Pricing validation must be assessed against measured between-method differences. The recurrent policy is history conditioned; no separate ICL mechanism is established. Hard-only teacher updates omit easy target reward, but configuration selection uses easy development data. Mixed feedback has twice as many distinct target episodes. Intrinsic selfplay is development-selected. Financial adaptations of DPG, SOAR and selfplay are not literal language-model reproductions. No independent-seed uncertainty claim is supported.', '',
              '## Cost and release verification', '',
              'Supervisor must reconcile provider charges, elapsed campaign duration and terminal billing state before final signoff. See budget/ledger.jsonl and SUPERVISOR_STATE.md; process termination alone does not prove rental release.', '',
              'Retained ledger events: ' + str(len(ledger)) + '. Provider reconciliation/shutdown signoff: PENDING SUPERVISOR VERIFICATION.', '',
              '## Artifacts', '',
              '- final_metrics.csv: all seven selected-arm slots with per-domain improvement.',
              '- all_trials.csv: every retained trial/checkpoint and its status, including failures.',
              '- all_events.csv: all available training and evaluation events.',
              '- episode_losses.npz: lossless merged named raw episode arrays.',
              '- array_integrity.json: worker-side array shapes and finite counts.',
              '- raw_event_validation.json: strict event/raw schema, count, finiteness, sign, summary and aggregate checks; explicit failures remain recorded.',
              '- selected_pricing_sensitivity.json: bounded frozen-selected pricing comparison and paired-archive reference.',
              '- figures/: selected-lineage monitor/selection curves, separate all_trials_* curves, all-run generator trajectories, and easy/hard final outcomes.',
              '- artifact_audit.json: metadata completeness failures, if any.']
    (results / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'selected_slots': len(selected), 'trial_checkpoint_rows': len(all_rows), 'complete': completeness['complete']}))
    return 0 if completeness['complete'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    raise SystemExit(main(parser.parse_args().root.resolve()))
