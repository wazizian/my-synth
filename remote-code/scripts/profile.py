"""Metadata-only throughput/cap summary: no numerical imports or model execution.

Read small JSONL/status files, file sizes, and sampled nvidia-smi CSV. Safe on the
supervisor. Rates apply only to the recorded shapes/device, not production shapes
inferred from a tiny smoke run. --gpu-csv RUN_ID=PATH binds logs explicitly; use
*=PATH only for a single-worker archive. GPU timestamps are assumed UTC.
"""
import argparse
import csv
import datetime
import json
import math
from pathlib import Path


def events(path):
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def ratio(numerator, denominator):
    return numerator / denominator if numerator is not None and denominator and denominator > 0 else None


def timestamp(value):
    if not value: return None
    try:
        parsed = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        try: parsed = datetime.datetime.strptime(value.strip(), '%Y/%m/%d %H:%M:%S.%f')
        except ValueError: return None
    if parsed.tzinfo is None: parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed.timestamp()


def gpu_samples(path):
    samples = []
    with path.open(newline='') as stream:
        for raw in csv.DictReader(stream):
            row = {key.strip(): value.strip() for key, value in raw.items() if key and value}
            time = timestamp(row.get('timestamp'))
            memory = next((value for key, value in row.items() if key.startswith('memory.used')), None)
            try: used = float(memory.split()[0]) if memory else None
            except ValueError: used = None
            if time is not None and used is not None: samples.append((time, used))
    return samples


def summarize(root, active_seconds=2700., storage_gib=1., mappings=None):
    root = Path(root)
    mappings = mappings or {}
    rows = []
    for directory in sorted(path for path in (root / 'runs').glob('*') if path.is_dir()):
        config, status = read(directory / 'config.json'), read(directory / 'status.json')
        train = events(directory / 'train_metrics.jsonl')
        evaluation = events(directory / 'evaluation_metrics.jsonl')
        observations = train + ([status] if status else [])
        # Cumulative terminal counters are authoritative; never sum duplicated logs.
        def counter(name):
            values = [row[name] for row in observations if isinstance(row.get(name), (float, int))]
            return max(values) if values else None
        rounds = counter('outer_rounds')
        active = counter('active_training_seconds')
        steps = counter('learner_steps')
        paths = counter('synthetic_path_count')
        update = counter('integrated_update_seconds')
        simulation = counter('simulation_seconds')
        snapshots = list((directory / 'history').glob('*.eqx'))
        snapshot_bytes = sum(path.stat().st_size for path in snapshots)
        checkpoints = list((directory / 'checkpoints').glob('*/state.eqx'))
        ms_per_block = ratio(active * 1000 if active is not None else None, rounds)
        estimated = math.ceil(active_seconds * rounds / active) if rounds and active else None
        mean_snapshot = ratio(snapshot_bytes, len(snapshots))
        storage_cap = math.floor(storage_gib * 1024 ** 3 / mean_snapshot) if mean_snapshot else None
        proposed = min(estimated, storage_cap) if estimated is not None and storage_cap is not None else estimated
        starts = [timestamp(row.get('timestamp')) for row in train]
        starts = [value for value in starts if value is not None]
        first, last = (min(starts), timestamp(status.get('timestamp'))) if starts else (None, None)
        log_path = mappings.get(directory.name, mappings.get(status.get('worker'), mappings.get('*')))
        sampled = []
        if log_path is not None and first is not None and last is not None:
            sampled = [memory for time, memory in gpu_samples(Path(log_path)) if first <= time <= last]
        # Domain records precede an aggregate event; the aggregate advances the
        # role's cumulative timer. Count complete groups, not stale counters.
        pending = {}
        evaluation_counts, evaluation_times = {}, {}
        for event in evaluation:
            role = event.get('split_role')
            timer = {'monitor': 'monitoring_seconds', 'selection': 'selection_seconds',
                     'final_test': 'final_test_seconds'}.get(role)
            if timer is None: continue
            key = (role, event.get('checkpoint_id'))
            if event.get('event_type') == 'evaluation':
                group = pending.setdefault(key, {'initial': event.get(timer), 'count': 0})
                group['count'] += event.get('sample_count', 0)
            elif event.get('event_type') == 'evaluation_aggregate' and key in pending:
                group = pending.pop(key)
                final = event.get(timer)
                if final is not None and group['initial'] is not None and final > group['initial']:
                    evaluation_counts[role] = evaluation_counts.get(role, 0) + group['count']
                    evaluation_times[role] = evaluation_times.get(role, 0.) + final - group['initial']
        method = config.get('method', directory.name)
        family = 'fixed' if method in ('baseline', 'hard_reference') else method.split('_')[0]
        row = dict(run_id=directory.name, method=method, family=family, status=status.get('status'),
                   purpose=config.get('purpose'), gpu_model=status.get('gpu_model'), worker=status.get('worker'),
                   batch_size=config.get('batch_size'), block_steps=config.get('block_steps'),
                   hidden_size=config.get('hidden_size'), compile_seconds=counter('compile_seconds'),
                   compile_scope='cumulative training/monitor/selection compilation; reporting compiles separate',
                   active_training_seconds=active, outer_rounds=rounds, learner_steps=steps,
                   milliseconds_per_block=ms_per_block,
                   milliseconds_per_update_block=ratio(update * 1000 if update is not None else None, rounds),
                   simulated_paths_per_second=ratio(paths, simulation),
                   end_to_end_active_paths_per_second=ratio(paths, active),
                   feedback_episode_evaluation_count=counter('feedback_episode_evaluation_count'),
                   integrated_update_seconds=update, teacher_seconds=counter('teacher_seconds'),
                   teacher_overhead_seconds=None, teacher_overhead_scope='not separately measured; adaptive teacher_seconds includes student updates',
                   feedback_seconds=counter('feedback_seconds'),
                   evaluation_pipeline_paths_per_second={role: ratio(evaluation_counts[role], duration)
                       for role, duration in evaluation_times.items()},
                   evaluation_scope='synchronized rollout plus GPU summary and raw archive I/O; excludes compilation',
                   checkpoint_seconds=counter('checkpoint_seconds'),
                   checkpoint_scope='checkpoint serialization plus selfplay snapshot writes/history reads; not isolated main-checkpoint I/O',
                   checkpoint_bundle_count=len(checkpoints), checkpoint_state_bytes=sum(p.stat().st_size for p in checkpoints),
                   selfplay_snapshot_count=len(snapshots), selfplay_history_bytes=snapshot_bytes,
                   selfplay_mean_snapshot_bytes=mean_snapshot,
                   peak_sampled_vram_mib=max(sampled) if sampled else None,
                   allocator_peak_bytes_in_use=counter('allocator_peak_bytes_in_use'),
                   allocator_peak_pool_bytes=counter('allocator_peak_pool_bytes'),
                   allocator_peak_in_use_mib=ratio(counter('allocator_peak_bytes_in_use'),1024**2),
                   allocator_peak_pool_mib=ratio(counter('allocator_peak_pool_bytes'),1024**2),
                   allocator_memory_scope='maximum observed current-process device-allocator high-water mark at this run terminal; includes earlier sequential runs and evaluation; not isolated trial or total device VRAM',
                   allocator_memory_observations=[{key:event.get(key) for key in
                       ('timestamp','worker','instance','allocator_memory_pid','allocator_memory_device',
                        'allocator_peak_bytes_in_use','allocator_peak_pool_bytes','allocator_memory_supported','allocator_memory_error')}
                       for event in train if event.get('event_type')=='allocator_memory'],
                   vram_sample_count=len(sampled), vram_scope='nvidia-smi 5-second samples in run interval; true peak may be higher',
                   vram_source=str(log_path) if log_path else None,
                   proposed_active_seconds=active_seconds, measured_rate_estimated_blocks=estimated,
                   proposed_max_blocks=proposed, history_storage_limit_gib=storage_gib,
                   estimated_history_bytes_at_cap=proposed * mean_snapshot if proposed is not None and mean_snapshot else None,
                   cap_scope='proposal only; same architecture, batch, K and GPU required; not an enforced runner limit',
                   profile_sufficient_for_production=(config.get('batch_size') == 256 and config.get('block_steps') == 4
                       and config.get('hidden_size') == 64 and rounds is not None and rounds >= 5))
        rows.append(row)
    return dict(runs=rows, numerical_work_executed=False,
                missing_families=sorted({'fixed', 'dpg', 'soar', 'selfplay'} - {row['family'] for row in rows}),
                limitations=['No isolated teacher or feedback timer is inferred from integrated counters.',
                             'Timing counters omit history I/O from active training; inspect checkpoint_seconds separately.',
                             'Sampled VRAM is a lower bound on the true peak.',
                             'Allocator high-water marks are process-lifetime counters, may include earlier sequential runs, and do not measure total device VRAM or an isolated trial.',
                             'No cap is extrapolated between shapes or enforced by this reporting script.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--active-seconds', type=float, default=2700.)
    parser.add_argument('--history-storage-gib', type=float, default=1.)
    parser.add_argument('--gpu-csv', action='append', default=[], metavar='RUN_OR_WORKER=PATH')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.active_seconds <= 0 or args.history_storage_gib <= 0: parser.error('Caps must be positive')
    mappings = {}
    for value in args.gpu_csv:
        key, separator, path = value.partition('=')
        if not separator: parser.error('--gpu-csv requires RUN_OR_WORKER=PATH')
        mappings[key] = path
    result = summarize(args.root, args.active_seconds, args.history_storage_gib, mappings)
    output = args.output or args.root / 'results/profile.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'runs': len(result['runs']), 'missing_families': result['missing_families'], 'output': str(output)}))


if __name__ == '__main__':
    main()
