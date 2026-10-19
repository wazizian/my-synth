"""GPU worker: choose among the bounded credit designs using development only.

Input files are packaged development summaries, never final test episodes.
"""
from pathlib import Path
import hashlib
import json
import os

import jax


def main(inputs, output):
    if jax.default_backend() != 'gpu':
        raise RuntimeError('Selection processing belongs on GPU worker')
    candidates = []
    for path in sorted(Path(inputs).glob('*.json')):
        row = json.loads(path.read_text())
        result = row['results']
        confirmation = result['streams']['confirmation']
        lower = [confirmation[d]['gain']['ci95'][0] for d in ('easy', 'hard')]
        point = [confirmation[d]['gain']['gain'] for d in ('easy', 'hard')]
        candidates.append(dict(campaign=row['campaign'], bp=result['bp'],
            passed=result['passed'], minimum_confirmation_lower_bound=min(lower),
            mean_confirmation_gain=sum(point) / 2,
            source_path=str(path), input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            scales=[max(.001, result['streams']['screen'][d]['metrics']['baseline']['terminal_std'])
                    for d in ('easy', 'hard')]))
    if len(candidates) != 6:
        raise RuntimeError('Require all two campaigns by three costs; no early cherry picking')
    chosen = max(candidates, key=lambda c: (c['passed'],
                  c['minimum_confirmation_lower_bound'], c['mean_confirmation_gain']))
    result = dict(selected=chosen, candidates=candidates,
        status='benchmark_frozen',
        comparison_scope='gated' if chosen['passed'] else 'exploratory; original specialist gate failed',
        rule='gate pass first, then maximum minimum confirmation gain lower bound, then mean point gain',
        scale_rule='baseline screen terminal SD in each target domain, floored at .001',
        final_test_seed=520261017, final_test_opened=False)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + '.tmp')
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, out)


if __name__ == '__main__':
    main(os.environ['CREDIT_GATE_INPUTS'], Path(os.environ['JOB_OUTPUT_DIR']) / 'credit_freeze.json')
