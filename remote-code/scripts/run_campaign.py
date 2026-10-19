"""Central metadata-only multi-GPU queue with resumable phase leases.

Preparation must already have passed on a GPU worker. The command screens all
arms before common-runtime extensions, seals selection, archives every retained
checkpoint test, and exports a worker-generated report. Rental launchers own
cleanup and transactional budget reservations. No numerical stack is imported.
"""
import argparse,fcntl,datetime,hashlib,json,os,shutil,subprocess,sys,tarfile,time,tempfile,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ARMS=['baseline','hard_reference','dpg_hard','soar_hard','selfplay','dpg_mixed','soar_mixed']
RUNS=[m+'-lr001' for m in ARMS]
COMPUTE_DEADLINE=1792301040.
LAUNCH_OVERHEAD_SECONDS=600+900+4*120+120
LINK_BASE=Path('/home/jump/campaign-link-base')
ASSET_STORAGE_RESERVE=1024**3

def transfer_plan(archive_bytes=0,logical_bytes=0,file_count=0,export_growth_bytes=0,export_growth_files=0):
    """Conservative bounded transfer allowances; no network speed guarantee."""
    upload=0 if not archive_bytes else max(300,math.ceil(120+archive_bytes/(8*1024**2)))
    if min(archive_bytes,logical_bytes,file_count,export_growth_bytes,export_growth_files)<0:
        raise ValueError('Negative transfer estimate')
    sync=max(120,math.ceil(120+(logical_bytes+export_growth_bytes)/(16*1024**2)+(file_count+export_growth_files)/1000))
    if upload>1800 or sync>1200:
        raise RuntimeError('Archive exceeds bounded transfer policy; split the handoff before rental')
    return dict(upload_timeout_seconds=upload,sync_timeout_seconds=sync,
        launcher_overhead_seconds=600+900+upload+4*sync+120,
        archive_bytes=archive_bytes,logical_bytes=logical_bytes,file_count=file_count,
        export_growth_bytes=export_growth_bytes,export_growth_files=export_growth_files,
        export_logical_bytes=logical_bytes+export_growth_bytes,export_file_count=file_count+export_growth_files,
        assumption='upload 8MiB/s; full export including planned growth 16MiB/s plus metadata; all four bounded sync waits reserved')

def asset_roots(run_ids):
    run_ids=list(run_ids)
    if set(run_ids)=={'baseline-lr001','hard_reference-lr001'}:
        run_ids+=['profile_'+arm for arm in ARMS]
    names=['data','cache/pricing','cache/jax','objective_scales.json','runs/manifest.jsonl',
           'budget/ledger.jsonl','budget/reservations.json','results/integrated_smoke.json',
           'results/termination_validation.json','results/preparation_gate.json',
           'results/production_profile_gate.json','results/profile.json','results/pricing_sensitivity',
           'results/worker_environments',
           'results/development_gate.json','results/residency_profile_transition.json',
           'queue/selected_configs.json','validation.json','validation_followup.json',
           'dependency-lock-or-export.txt','requirements-worker.txt','DECISIONS.md',
           'code_provenance.json','.git']+['runs/'+run_id for run_id in run_ids]+['results/frozen_selection.json']
    return [ROOT/name for name in names if (ROOT/name).exists()]

def asset_estimate(run_ids):
    total=count=0
    for source in asset_roots(run_ids):
        paths=source.rglob('*') if source.is_dir() else (source,)
        for path in paths:
            if path.is_symlink():raise RuntimeError('Refusing symlinked asset handoff')
            if path.is_file():total+=path.stat().st_size;count+=1
    # Deliberately ignore beneficial gzip compression and within-tar hardlinks.
    upper=math.ceil(total*1.01)+count*4096+8*1024**2
    return dict(logical_bytes=total,file_count=count,archive_upper_bound_bytes=upper)

def export_growth(run_ids,seconds=0,evaluate=False):
    """Bound future selfplay history by its frozen block plan, not input size.

    Reserve the entire remaining immutable plan even for a shorter lease: active
    time does not impose a hard bound on the number of generated snapshots.
    """
    growth_bytes=growth_files=0
    if evaluate or seconds<=0:return dict(export_growth_bytes=0,export_growth_files=0)
    for run_id in run_ids:
        if run_id.rsplit('-lr',1)[0]!='selfplay':continue
        directory=ROOT/'runs'/run_id
        config_path=directory/'config.json'
        if config_path.exists():
            config=json.loads(config_path.read_text())
            cap=int(config['selfplay_max_blocks'])
            size=int(config['selfplay_snapshot_estimate_bytes'])
            limit=int(config['selfplay_history_limit_bytes'])
            planned_seconds=float(config['selfplay_planned_active_seconds'])
        else:
            rows=[r for r in json.loads((ROOT/'results/profile.json').read_text())['runs']
                  if r.get('run_id')=='profile_selfplay' and r.get('status')=='completed']
            if len(rows)!=1:raise RuntimeError('Selfplay export planning requires completed profile')
            row=rows[0];planned_seconds=2700.
            if min(row['outer_rounds'],row['active_training_seconds'],row['selfplay_mean_snapshot_bytes'])<=0:
                raise RuntimeError('Invalid selfplay export profile')
            cap=max(2,math.ceil(2*planned_seconds*row['outer_rounds']/row['active_training_seconds']))
            size=math.ceil(row['selfplay_mean_snapshot_bytes']);limit=12*1024**3
        if min(cap,size,limit)<=0 or cap*size>limit or seconds>planned_seconds:
            raise RuntimeError('Selfplay export estimate conflicts with immutable history plan')
        existing=list((directory/'history').glob('*.eqx'))
        # Header stat only; never deserialize scientific arrays on supervisor.
        current_bytes=sum(p.stat().st_size for p in existing)
        growth_bytes+=max(0,cap*size-current_bytes)+64*1024**2
        growth_files+=max(0,cap-len(existing))+1024
    return dict(export_growth_bytes=growth_bytes,export_growth_files=growth_files)


def asset_transfer_plan(path):
    path=Path(path)
    sidecar=path.with_suffix(path.suffix+'.metadata.json')
    compressed=path.stat().st_size
    growth={}
    if sidecar.exists():
        metadata=json.loads(sidecar.read_text())
        if metadata.get('archive_bytes')!=compressed:raise RuntimeError('Asset size differs from published handoff metadata')
        logical,count=metadata['logical_bytes'],metadata['file_count']
        growth={key:metadata.get(key,0) for key in ('export_growth_bytes','export_growth_files')}
    else:
        # Legacy caller: inspect tar headers only, never deserialize scientific arrays.
        logical=count=0
        with tarfile.open(path,'r:*') as archive:
            for entry in archive:
                if entry.isfile():logical+=entry.size;count+=1
    return transfer_plan(compressed,logical,count,**growth)

def immutable_artifact(relative):
    """Only committed scientific arrays, never mutable pointers or log files."""
    p=Path(relative).parts
    return ((len(p)==4 and p[0]=='runs' and p[2]=='history' and p[3].endswith('.eqx')) or
            (len(p)==5 and p[0]=='runs' and p[2]=='checkpoints' and p[4]=='state.eqx' and not p[3].startswith('.')) or
            (len(p)==4 and p[0]=='runs' and p[2]=='episode_metrics' and p[3].endswith('.npz')) or
            (len(p)==2 and p[0]=='data' and p[1].endswith('.npz')) or
            (len(p)==3 and p[:2]==('cache','pricing') and p[2].endswith('.npz')))

def atomic_copy(source,destination):
    """Publish a fresh inode, so a previous hardlink can never be overwritten."""
    source,destination=Path(source),Path(destination)
    if source.is_symlink() or destination.is_symlink():raise RuntimeError('Refusing symlinked artifact import')
    destination.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix='.'+destination.name+'.',dir=destination.parent)
    os.close(fd)
    try:
        shutil.copy2(source,temporary)
        with open(temporary,'rb') as stream:os.fsync(stream.fileno())
        os.replace(temporary,destination)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)

def immutable_link(source,destination):
    """Atomic link creation; conflicting immutable names are an integrity error."""
    source,destination=Path(source),Path(destination)
    if source.is_symlink() or destination.is_symlink():raise RuntimeError('Refusing symlinked immutable artifact')
    destination.parent.mkdir(parents=True,exist_ok=True)
    try:
        os.link(source,destination)
    except FileExistsError:
        if not os.path.samefile(source,destination) and (
            source.stat().st_size!=destination.stat().st_size or digest(source)!=digest(destination)):
            raise RuntimeError('Conflicting immutable scientific artifact: '+str(destination))
    # EXDEV/disk errors deliberately fail closed instead of silently doubling disk.

def import_tree(source,destination):
    source,destination=Path(source),Path(destination)
    if source.is_symlink() or destination.is_symlink():raise RuntimeError('Refusing symlinked artifact directory')
    for path in sorted(source.rglob('*')):
        if path.is_symlink():raise RuntimeError('Refusing symlinked artifact tree')
        target=destination/path.relative_to(source)
        if path.is_dir():target.mkdir(parents=True,exist_ok=True);continue
        if not path.is_file():continue
        relative=target.relative_to(ROOT)
        (immutable_link if immutable_artifact(relative) else atomic_copy)(path,target)

def build_link_base(root=ROOT,base=LINK_BASE):
    """Future-launch preparation only; no rewriting or dedup of existing files."""
    root,base=Path(root).resolve(),Path(base)
    count=0
    patterns=('runs/*/history/*.eqx','runs/*/checkpoints/*/state.eqx',
              'runs/*/episode_metrics/*.npz','data/*.npz','cache/pricing/*.npz')
    for pattern in patterns:
        for source in root.glob(pattern):
            relative=source.relative_to(root)
            if not immutable_artifact(relative):continue
            if source.is_symlink() or not source.resolve().is_relative_to(root):
                raise RuntimeError('Immutable artifact escapes canonical archive')
            immutable_link(source,base/'project'/relative)
            count+=1
    return {'base':str(base),'immutable_files':count}

def optional_tuning_storage(state):
    """Conservative metadata-only projection, including one full handoff copy."""
    if state.get('storage_policy',{}).get('skip_optional_second_lr'):
        return False,dict(reason=state['storage_policy'].get('reason','Supervisor storage policy preserves all seven final runs'))
    try:
        transition=json.loads((ROOT/'results/residency_profile_transition.json').read_text())
        before,after=transition['before']['selfplay'],transition['after']['selfplay']
        elapsed=after['active_training_seconds']-before['active_training_seconds']
        rounds=after['outer_rounds']-before['outer_rounds']
        if elapsed<=0 or rounds<=0:raise ValueError('Missing corrected residency profile interval')
        rows=json.loads((ROOT/'results/profile.json').read_text())['runs']
        size=next(row['selfplay_mean_snapshot_bytes'] for row in rows if row['run_id']=='profile_selfplay')
        if not size or size<=0:raise ValueError('Missing observed snapshot size')
        rate=rounds/elapsed
        history=int(1.10*rate*(2700+600)*size)
        current=sum(path.stat().st_size for prefix in ('runs','data','cache/pricing')
                    for path in (ROOT/prefix).rglob('*') if path.is_file() and
                    immutable_artifact(path.relative_to(ROOT)))
        handoff=current+history
        reserve=2*1024**3
        required=history+handoff+reserve
        free=shutil.disk_usage(ROOT).free
        evidence=dict(corrected_rounds_per_second=rate,snapshot_bytes=size,
            projected_selected_and_extra_screen_history_bytes=history,
            handoff_uncompressed_equivalent_bytes=handoff,reserve_bytes=reserve,
            required_additional_bytes=required,available_bytes=free,
            reason='Preserve all seven final runs and one complete handoff within supervisor storage')
        return required<=free,evidence
    except (OSError,ValueError,KeyError,TypeError,StopIteration) as error:
        return False,dict(reason='Storage fit for optional second LR is unproven',error=str(error))

def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()

def preparation_gate():
    required=['cache/pricing/heston.validation.json','cache/pricing/fallback.validation.json',
              'results/integrated_smoke.json','results/termination_validation.json',
              'results/production_profile_gate.json']
    reports={name:json.loads((ROOT/name).read_text()) for name in required}
    for name,report in reports.items():
        if not report.get('passed'):raise RuntimeError('Required GPU preparation gate failed: '+name)
    if set(reports[required[-1]].get('runs',{}))!=set(ARMS):raise RuntimeError('Production profile must cover all seven arms')
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    if reports[required[0]].get('sha256')!=manifest['pricing_hash']:raise RuntimeError('Pricing/table evidence mismatch')
    fallback=reports[required[1]]
    if fallback.get('pricing_hash')!=manifest['pricing_hash']:raise RuntimeError('Fallback/table evidence mismatch')
    if fallback.get('pricing_source_hash')!=digest(ROOT/'src/pricing.py'):raise RuntimeError('Fallback source changed since validation')
    if manifest.get('objective_scales_hash')!=digest(ROOT/'objective_scales.json'):raise RuntimeError('Frozen scales changed')
    for name,record in manifest['datasets'].items():
        if digest(ROOT/record['path'])!=record['sha256']:raise RuntimeError('Immutable dataset changed: '+name)
    return {'artifacts':{name:digest(ROOT/name) for name in required},
            'benchmark_hash':manifest['benchmark_hash'],'verified_at':time.time()}

def source_snapshot():
    return {str(path.relative_to(ROOT)):digest(path) for directory in ('src','scripts','tests','configs')
            for path in sorted((ROOT/directory).rglob('*')) if path.is_file() and '__pycache__' not in path.parts}

def final_complete(directory,checkpoint_id):
    path=directory/f'final_test_{checkpoint_id}.json'
    if not path.exists():return False
    result=json.loads(path.read_text())
    for domain in ('base','easy','hard'):
        row=result.get('domains',{}).get(domain,{})
        artifact=row.get('episode_metrics_path')
        if (row.get('status')!='ok' or row.get('checkpoint_id')!=checkpoint_id or
            row.get('raw_mse') is None or not artifact or not (ROOT/artifact).exists()):return False
    return True

def phase_timeout(run_ids,seconds,evaluate=False):
    """Reserve remaining work, compilation and measured evaluation overhead."""
    estimate=120.
    if not evaluate and set(run_ids)=={'baseline-lr001','hard_reference-lr001'}:
        estimate+=600.  # Bounded paired direct-pricing diagnostics after training.
        estimate+=600.  # All-seven resident-table profile continuations to 15s.
    for run_id in run_ids:
        directory=ROOT/'runs'/run_id
        status=json.loads((directory/'status.json').read_text()) if (directory/'status.json').exists() else {}
        if evaluate:
            pending=sum(not final_complete(directory,p.parent.name)
                        for p in (directory/'checkpoints').glob('*/metadata.json'))
            # Three 8192-episode domains; bounded but conservative until profiled.
            estimate+=120+pending*90
        else:
            remaining=max(0.,seconds-status.get('active_training_seconds',0.))
            profile=ROOT/'runs'/('profile_'+run_id.rsplit('-lr',1)[0])
            timing=status
            timing_directory=directory
            if not status and (profile/'status.json').exists():
                timing=json.loads((profile/'status.json').read_text());timing_directory=profile
            aggregate=[x for x in events(timing_directory/'evaluation_metrics.jsonl') if x.get('event_type')=='evaluation_aggregate']
            measured=timing.get('monitoring_seconds',0)+timing.get('selection_seconds',0)
            eval_seconds=max(10.,measured/max(len(aggregate),1))
            config=json.loads((directory/'config.json').read_text()) if (directory/'config.json').exists() else {}
            monitor=config.get('monitor_interval_seconds',180)
            selection=config.get('selection_interval_seconds',900)
            evaluations=2+int(remaining/max(monitor,1))+int(remaining/max(selection,1))
            if not status:evaluations+=2
            estimate+=remaining+max(120.,timing.get('compile_seconds',0.))+eval_seconds*evaluations+60
    requested=int(estimate*1.15)+120
    estimate=asset_estimate(run_ids)
    overhead=transfer_plan(estimate['archive_upper_bound_bytes'],estimate['logical_bytes'],estimate['file_count'],**export_growth(run_ids,seconds,evaluate))['launcher_overhead_seconds']
    available=int(COMPUTE_DEADLINE-time.time()-overhead-30)
    if requested>available:raise RuntimeError(f'Insufficient deadline for lease: need {requested}s, available {available}s; retain partial results')
    return requested

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2));os.replace(temporary,path)

def events(path):
    if not path.exists():return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def assets(destination,run_ids=(),seconds=0,evaluate=False):
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    estimate=asset_estimate(run_ids)
    free=shutil.disk_usage(destination.parent).free
    if free<estimate['archive_upper_bound_bytes']+ASSET_STORAGE_RESERVE:
        raise RuntimeError('Insufficient archive capacity before handoff: '+json.dumps(dict(estimate,
            free_bytes=free,reserve_bytes=ASSET_STORAGE_RESERVE)))
    fd,temporary=tempfile.mkstemp(prefix='.'+destination.name+'.',dir=destination.parent)
    os.close(fd)
    try:
        with tarfile.open(temporary,'w:gz') as archive:
            for source in asset_roots(run_ids):
                if shutil.disk_usage(destination.parent).free<ASSET_STORAGE_RESERVE:
                    raise RuntimeError('Archive construction reached reserved free-space margin')
                archive.add(source,arcname=str(source.relative_to(ROOT)))
        if shutil.disk_usage(destination.parent).free<ASSET_STORAGE_RESERVE:
            raise RuntimeError('Completed archive leaves insufficient export reserve')
        with open(temporary,'rb') as stream:os.fsync(stream.fileno())
        size=os.path.getsize(temporary)
        plan=transfer_plan(size,estimate['logical_bytes'],estimate['file_count'],**export_growth(run_ids,seconds,evaluate))
        os.replace(temporary,destination)
        atomic(destination.with_suffix(destination.suffix+'.metadata.json'),dict(estimate,**plan,
            created_at=time.time(),storage_free_before_bytes=free,storage_reserve_bytes=ASSET_STORAGE_RESERVE))
        return plan
    finally:
        if os.path.exists(temporary):os.unlink(temporary)

def copy_result(project,run_ids):
    # The provider exports these alongside project/, not inside it. Preserve
    # actual worker dependency/device evidence for the final report handoff.
    environment_files=('environment.txt','requirements.lock','uv-version.txt','nvidia-smi.txt')
    present=[name for name in environment_files if (project.parent/name).is_file()]
    if present:
        identity=hashlib.sha256(str(project.resolve()).encode()).hexdigest()[:16]
        destination=ROOT/'results/worker_environments'/identity
        for name in present:atomic_copy(project.parent/name,destination/name)
        atomic(destination/'provenance.json',{'source_output_directory':str(project.parent),
            'files':{name:digest(destination/name) for name in present},'imported_at':time.time()})
    run_ids=list(run_ids)
    if set(run_ids)=={'baseline-lr001','hard_reference-lr001'}:
        run_ids+=['profile_'+arm for arm in ARMS]
        for name in ('profile.json','production_profile_gate.json','residency_profile_transition.json'):
            if (project/'results'/name).exists():atomic_copy(project/'results'/name,ROOT/'results'/name)
    # Trusted campaign-generated cache, matched RTX4090 and pinned dependency stack.
    if (project/'cache/jax').exists():import_tree(project/'cache/jax',ROOT/'cache/jax')
    sensitivity=project/'results/pricing_sensitivity'
    if sensitivity.exists():
        import_tree(sensitivity,ROOT/'results/pricing_sensitivity')
    if (project/'code_provenance.json').exists():
        record=json.loads((project/'code_provenance.json').read_text())
        destination=ROOT/'source_snapshots'/record['commit']
        destination.mkdir(parents=True,exist_ok=True)
        atomic_copy(project/'code_provenance.json',destination/'code_provenance.json')
        if (project/'.git').exists():import_tree(project/'.git',destination/'.git')
    for run_id in run_ids:
        source=project/'runs'/run_id
        if source.exists():import_tree(source,ROOT/'runs'/run_id)
    # Append immutable manifests; preserve provenance of each worker record.
    source=project/'runs/manifest.jsonl'
    if source.exists():
        destination=ROOT/'runs/manifest.jsonl'
        known=set(destination.read_text().splitlines()) if destination.exists() else set()
        merged=list(destination.read_text().splitlines()) if destination.exists() else []
        for line in source.read_text().splitlines():
            if line and line not in known:merged.append(line);known.add(line)
        fd,temporary=tempfile.mkstemp(prefix='.manifest-',dir=destination.parent)
        try:
            with os.fdopen(fd,'w') as stream:
                stream.write(''.join(line+'\n' for line in merged));stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,destination)
        finally:
            if os.path.exists(temporary):os.unlink(temporary)

def run_phase(state,name,groups,seconds,evaluate=False):
    saved=state.setdefault('phases',{}).setdefault(name,{'leases':{}})
    if saved.get('status')=='complete':return
    planned=[(index,run_ids) for index,run_ids in enumerate(groups) if run_ids and
             saved['leases'].get(f'{name}-{index}',{}).get('status')!='complete']
    timeouts={index:phase_timeout(run_ids,seconds,evaluate) for index,run_ids in planned}
    reservation_path=ROOT/'budget/reservations.json'
    reservations=json.loads(reservation_path.read_text()) if reservation_path.exists() else {}
    committed=sum(value['reserved_usd'] if value.get('status')=='reserved' else value.get('actual_usd',0)
                  for value in reservations.values())
    projected=0.
    for index,run_ids in planned:
        estimate=asset_estimate(run_ids)
        overhead=transfer_plan(estimate['archive_upper_bound_bytes'],estimate['logical_bytes'],estimate['file_count'],**export_growth(run_ids,seconds,evaluate))['launcher_overhead_seconds']
        projected+=(timeouts[index]+overhead)*.60/3600+.5
    if committed+projected>22:raise RuntimeError(f'{name} phase reservation exceeds normal budget: {committed+projected:.2f} USD')
    processes={}
    for index,run_ids in enumerate(groups):
        if not run_ids:continue
        worker=f'{name}-{index}';lease=saved['leases'].get(worker,{})
        if lease.get('status')=='complete':continue
        if (ROOT/'workers'/worker/'active.json').exists():
            raise RuntimeError(f'Worker {worker} still has an active rental record; verify provider cleanup before restarting')
        if lease.get('pid'):
            try:os.kill(lease['pid'],0)
            except ProcessLookupError:pass
            else:raise RuntimeError(f'Lease {worker} process {lease["pid"]} still exists; inspect it before restarting')
        bundle=ROOT/'queue'/f'{worker}.tar.gz';assets(bundle,run_ids,seconds,evaluate)
        timeout=timeouts[index]
        command=[sys.executable,str(ROOT/'scripts/launch_worker.py'),'--worker',worker,'--job',str(ROOT/'campaign_job.py'),'--assets',str(bundle),'--seconds',str(timeout),'--require-artifact','project/group_status.json','--','--runs',*run_ids,'--seconds',str(seconds)]
        if evaluate:command.append('--evaluate')
        log=(ROOT/'queue'/f'{worker}.log').open('a')
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes[worker]=(process,log,run_ids)
        saved['leases'][worker]={'pid':process.pid,'run_ids':run_ids,'status':'running','command':command,'started':time.time()}
        atomic(ROOT/'queue/state.json',state)
    while processes:
        for worker,(process,log,run_ids) in list(processes.items()):
            result=process.poll()
            if result is None:continue
            log.close();lease=saved['leases'][worker]
            matches=sorted((ROOT/'workers'/worker/'runs').glob('*/outputs/project'))
            valid_group=False
            if matches:
                copy_result(matches[-1],run_ids)
                group_path=matches[-1]/'group_status.json'
                if group_path.exists():
                    group=json.loads(group_path.read_text())
                    valid_group=(group.get('status')=='complete' and not group.get('failures') and
                                 set(group.get('completed_runs',[]))==set(run_ids))
            active=ROOT/'workers'/worker/'active.json'
            lease.update(status='complete' if result==0 and valid_group and not active.exists() else 'failed',returncode=result,finished=time.time())
            atomic(ROOT/'queue/state.json',state);del processes[worker]
        if processes:time.sleep(10)
    if any(v['status']!='complete' for v in saved['leases'].values()):raise RuntimeError(f'{name} failed; artifacts retained; fix before resume')
    saved['status']='complete';atomic(ROOT/'queue/state.json',state)

def freeze(selected_runs):
    if len(selected_runs)!=len(ARMS) or len(set(selected_runs))!=len(ARMS):raise RuntimeError('Selection must retain exactly seven distinct arms')
    data=json.loads((ROOT/'data/manifest.json').read_text());selected={}
    for arm,run_id in zip(ARMS,selected_runs):
        config=json.loads((ROOT/'runs'/run_id/'config.json').read_text())
        status=json.loads((ROOT/'runs'/run_id/'status.json').read_text())
        if config['method']!=arm or status.get('status')!='completed':raise RuntimeError('Invalid selected arm: '+arm)
        if status.get('active_training_seconds',0)<2700:raise RuntimeError('Selected arm has not reached common cumulative cap: '+arm)
        pointer=json.loads((ROOT/'runs'/run_id/'checkpoints/best.json').read_text())
        bundle=ROOT/'runs'/run_id/'checkpoints'/pointer['path']
        if digest(bundle/'state.eqx')!=pointer['state_sha256'] or digest(bundle/'metadata.json')!=pointer['metadata_sha256']:
            raise RuntimeError('Selected checkpoint checksum mismatch: '+run_id)
        scores=[row.get('balanced_objective') for row in events(ROOT/'runs'/run_id/'evaluation_metrics.jsonl')
                if row.get('event_type')=='evaluation_aggregate' and row.get('split_role')=='selection'
                and row.get('checkpoint_id')==pointer['checkpoint_id']]
        if not any(isinstance(score,(int,float)) for score in scores):raise RuntimeError('Selected checkpoint lacks selection evidence')
        selected[arm]={'run_id':run_id,'checkpoint_id':pointer['checkpoint_id']}
    value={'selected':selected,'checkpoint_policy':'lowest_balanced_selection_score',
           'benchmark_hash':data['benchmark_hash'],'objective_scales_hash':data['objective_scales_hash'],
           'frozen_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'final_active_seconds':2700}
    atomic(ROOT/'results/frozen_selection.json',value)

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared-root');p.add_argument('--dry-run',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',choices=['diagnostics','screens','extensions','tests']);a=p.parse_args()
    if a.dry_run:
        print(json.dumps({'arms':ARMS,'max_workers':3,'screen_active_seconds':600,'final_cumulative_active_seconds':2700,'budget_usd':25,'normal_ceiling_usd':22,'hard_deadline':'2026-10-18T05:54:00Z','phases':['diagnostics','screens','extensions','tests','report']},indent=2));return
    (ROOT/'queue').mkdir(exist_ok=True)
    (ROOT/'results').mkdir(exist_ok=True)
    controller_lock=(ROOT/'queue/controller.lock').open('a')
    fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.prepared_root:
        src=Path(a.prepared_root)
        validation=json.loads((src/'cache/pricing/heston.validation.json').read_text())
        if not validation.get('passed'):raise RuntimeError('Market gate not passed')
        for name in ('data','cache/pricing','cache/jax'):
            if (src/name).exists():import_tree(src/name,ROOT/name)
        atomic_copy(src/'objective_scales.json',ROOT/'objective_scales.json')
        if (src/'runs').exists():import_tree(src/'runs',ROOT/'runs')
        for name in ('integrated_smoke.json','termination_validation.json','production_profile_gate.json','profile.json'):
            if (src/'results'/name).exists():atomic_copy(src/'results'/name,ROOT/'results'/name)
    if not (ROOT/'data/manifest.json').exists():raise RuntimeError('GPU preparation artifacts required')
    atomic(ROOT/'results/preparation_gate.json',preparation_gate())
    state_path=ROOT/'queue/state.json'
    prior=json.loads(state_path.read_text()) if state_path.exists() else {}
    if prior.get('phases') and not a.resume:raise RuntimeError('Existing campaign phases require --resume')
    state=prior if a.resume else {'phase':'campaign','phases':{}}
    source=source_snapshot()
    if state.get('source_files') and state['source_files']!=source:raise RuntimeError('Frozen campaign source changed; explicit supervisor migration/revalidation required')
    state['source_files']=source
    atomic(state_path,state)
    subprocess.run([sys.executable,str(ROOT/'scripts/package_job.py'),'--entry','scripts/run_group.py','--output',str(ROOT/'campaign_job.py')],check=True)
    run_phase(state,'diagnostics',[[RUNS[0],RUNS[1]]],600)
    if a.stop_after=='diagnostics':return
    # The supervising process must inspect the diagnostics and freeze common
    # learner settings before authorizing adaptive work through this gate file.
    gate=ROOT/'results/development_gate.json'
    if not gate.exists() or not json.loads(gate.read_text()).get('passed'):raise RuntimeError('Development gate requires validated reference learning and finite prices/signals')
    run_phase(state,'screens',[[RUNS[2],RUNS[5]],[RUNS[3],RUNS[6]],[RUNS[4]]],600)
    if a.stop_after=='screens':return
    selected_path=ROOT/'queue/selected_configs.json'
    if selected_path.exists():selected_runs=json.loads(selected_path.read_text())['run_ids']
    else:
        # A second allowed generator LR is screened only with a broad remaining
        # envelope for all seven continuations and every archived final test.
        remaining=COMPUTE_DEADLINE-time.time()
        reservations=json.loads((ROOT/'budget/reservations.json').read_text())
        committed=sum(v.get('actual_usd',v['reserved_usd']) for v in reservations.values())
        candidates=[m+'-lr01' for m in ARMS[2:]]
        storage_ok,storage_evidence=optional_tuning_storage(state)
        state['optional_tuning_storage']=storage_evidence
        atomic(ROOT/'queue/state.json',state)
        if remaining>4*3600 and committed<12 and storage_ok:
            run_phase(state,'second_lr',[candidates[::3],candidates[1::3],candidates[2::3]],600)
            reason='Second allowed LR affordable under four-hour continuation/evaluation reserve'
        else:
            candidates=[];reason=('Second LR omitted for all adaptive arms to preserve all-seven-run storage/export envelope'
                                  if not storage_ok else 'Second LR omitted to preserve all-arm runtime/evaluation envelope')
        selected_runs=RUNS[:2]
        for arm,default in zip(ARMS[2:],RUNS[2:]):
            options=[default]+[x for x in candidates if x.startswith(arm+'-')]
            selected_runs.append(min(options,key=lambda run:json.loads((ROOT/'runs'/run/'status.json').read_text())['selection_score']))
        atomic(selected_path,{'run_ids':selected_runs,'reason':reason,'storage_evidence':storage_evidence,
                             'selection_objective':'fixed balanced easy/hard selection score','chosen_at':time.time()})
    run_phase(state,'extensions',[selected_runs[::3],selected_runs[1::3],selected_runs[2::3]],2700)
    if a.stop_after=='extensions':return
    if not (ROOT/'results/frozen_selection.json').exists():freeze(selected_runs)
    archived=[p.name for p in sorted((ROOT/'runs').iterdir()) if p.is_dir() and (p/'checkpoints').exists()]
    run_phase(state,'tests',[archived[::3],archived[1::3],archived[2::3]],0,evaluate=True)
    for run_id in archived:
        directory=ROOT/'runs'/run_id
        for metadata in (directory/'checkpoints').glob('*/metadata.json'):
            if not final_complete(directory,metadata.parent.name):raise RuntimeError('Incomplete final checkpoint archive: '+run_id+'/'+metadata.parent.name)
    if a.stop_after=='tests':return
    assets(ROOT/'queue/report.tar.gz',archived)
    subprocess.run([sys.executable,str(ROOT/'scripts/package_job.py'),'--entry','scripts/make_report.py','--output',str(ROOT/'report_job.py')],check=True)
    sensitivity_reports=[json.loads(path.read_text()) for path in (ROOT/'results/pricing_sensitivity').glob('*/report.json')]
    sensitivity_seconds=max((record.get('elapsed_seconds',0) for record in sensitivity_reports),default=0)
    report_seconds=max(900,math.ceil(1.25*sensitivity_seconds)+600)
    subprocess.run([sys.executable,str(ROOT/'scripts/launch_worker.py'),'--worker','report','--job',str(ROOT/'report_job.py'),'--assets',str(ROOT/'queue/report.tar.gz'),'--seconds',str(report_seconds),'--require-artifact','project/results/report.md'],check=True)
    output=sorted((ROOT/'workers/report/runs').glob('*/outputs/project/results/report.md'))[-1].parent
    import_tree(output,ROOT/'results')
    subprocess.run([sys.executable,str(ROOT/'scripts/audit_artifacts.py'),'--root',str(ROOT),'--output',str(ROOT/'results/completeness.json')],check=True)
    state['phase']='reported_pending_shutdown_and_billing_audit';atomic(ROOT/'queue/state.json',state)
    print('Report exported. Supervisor must verify provider shutdown and reconcile billing.',flush=True)
if __name__=='__main__':main()
