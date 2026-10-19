"""Lightweight source packaging only; never imports scientific modules."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', choices=('credit', 'option'), required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--arm', required=True)
    rate = parser.add_mutually_exclusive_group(required=True)
    rate.add_argument('--lr', type=float)
    rate.add_argument('--all-learning-rates', action='store_true')
    parser.add_argument('--bp', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    files = {str(p.relative_to(args.source)): p.read_text()
             for p in args.source.rglob('*')
             if p.is_file() and p.suffix in ('.py', '.json')}
    for name in ('adaptive.py', 'train_adaptive.py', 'archive_history.py', f'{args.benchmark}_adapter.py'):
        files[name] = (root / name).read_text()
    design = json.loads(args.design.read_text())
    manifest = {name: hashlib.sha256(value.encode()).hexdigest()
                for name, value in files.items()}
    worker = '''"""Standalone frozen GPU training payload."""
import os,sys,pathlib,json,traceback
root=pathlib.Path(os.environ['JOB_OUTPUT_DIR'])
source=root/'source';source.mkdir(exist_ok=True)
for name,value in FILES.items():
    p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)
(root/'source_hashes.json').write_text(json.dumps(MANIFEST,indent=2))
sys.path.insert(0,str(source))
import jax
jax.config.update('jax_enable_x64',True)
assert jax.default_backend()=='gpu'
from train_adaptive import run
try:
    if BENCHMARK=='credit':
        from credit_adapter import CreditAdapter
        from credit_hedging.learned_policy import TargetHedger
        adapter=CreditAdapter(source,BP)
        def make_model(seed):return TargetHedger(jax.random.key(seed))
    else:
        from option_adapter import OptionAdapter
        adapter=OptionAdapter(source,BP)
        make_model=adapter.make_model
    if ALL_LEARNING_RATES:
        import gc,shutil
        trials=[]
        for lr in DESIGN['learning_rates']:
            folder=root/'training'/f'lr-{lr:g}'
            if ARM=='selfplay':
                os.environ['JOB_SELFPLAY_HISTORY_DIR']=str(root.parent/'selfplay-history'/ARM/f'lr-{lr:g}')
            run(adapter,make_model,folder,DESIGN,ARM,lr)
            if ARM=='selfplay':
                from archive_history import archive_history
                history=pathlib.Path(os.environ.get('JOB_SELFPLAY_HISTORY_DIR',str(folder.parent.parent/'selfplay-history'/ARM/f'lr-{lr:g}')))
                marker=root/'recovery-requirements.json'
                def mark_history(required,**details):
                    temporary=marker.with_suffix('.tmp')
                    temporary.write_text(json.dumps(dict(external_history_required_until_complete=required,external_history_path=str(history),arm=ARM,lr=lr,**details)))
                    os.replace(temporary,marker)
                mark_history(True,reason='Portable history archive awaiting complete readback')
                record=archive_history(history,root/'histories'/f'lr-{lr:g}.tar.gz')
                mark_history(False,reason='Complete portable history archive verified on worker',archive=record['archive'],archive_sha256=record['archive_sha256'])
            row=json.loads((folder/'selection.json').read_text())
            row['folder']=str(folder.relative_to(root));trials.append(row)
            (root/'trials.json').write_text(json.dumps(trials,indent=2))
            jax.clear_caches();gc.collect()
        selected=min(trials,key=lambda row:row['selection_score'])
        shutil.copy2(root/selected['folder']/selected['checkpoint'],root/'selected.pkl')
        (root/'selection.json').write_text(json.dumps(dict(arm=ARM,selected=selected,trials=trials,final_test_opened=False),indent=2))
        (root/'complete.json').write_text(json.dumps(dict(status='complete',arm=ARM,final_test_opened=False)))
    else:
        run(adapter,make_model,root/'training',DESIGN,ARM,LR)
except BaseException:
    (root/'failure.json').write_text(json.dumps({'traceback':traceback.format_exc()}))
    raise
'''
    prefix = '\n'.join(f'{k}={v!r}' for k, v in dict(
        FILES=files, MANIFEST=manifest, BENCHMARK=args.benchmark, BP=args.bp,
        DESIGN=design, ARM=args.arm, LR=args.lr,
        ALL_LEARNING_RATES=args.all_learning_rates).items())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(prefix + '\n' + worker)
    args.output.with_suffix('.manifest.json').write_text(json.dumps(dict(
        source_hashes=manifest, design=design, benchmark=args.benchmark, bp=args.bp,
        arm=args.arm, lr=args.lr), indent=2) + '\n')


if __name__ == '__main__':
    main()
