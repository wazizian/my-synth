"""Run/resume one retained trial on its centrally assigned GPU."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--config',required=True,help='JSON immutable trial configuration')
    parser.add_argument('--active-seconds',type=float,required=True,help='Cumulative active-training cap')
    parser.add_argument('--no-resume',action='store_true')
    args = parser.parse_args()
    from src.runner import train_one
    result = train_one(args.root,json.loads(Path(args.config).read_text()),args.active_seconds,
                       resume=not args.no_resume)
    print(json.dumps({k:result.get(k) for k in ('run_id','status','last_checkpoint_id','active_training_seconds','error')}),flush=True)
    return 1 if result['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
