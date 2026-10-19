"""Immutable saved episode splits. Test reads require frozen selection manifest."""
import json
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from .market import simulate, DEFAULT_CONFIG as MARKET_CONFIG, mixture_probabilities, REGIMES
from .objective import base_premium_fraction
from .protocol import DOMAINS, ROLES, stream_key, canonical_hash, file_hash
from .pricing import load_tables, table_hash

@eqx.filter_jit
def generate(q,key,n,tables):
    a,b=jax.random.split(key)
    ids=jax.random.categorical(a,jnp.log(q),shape=(n,))
    return simulate(ids,b,MARKET_CONFIG,tables)

def prepare(root):
    from .runtime import require_gpu
    require_gpu()
    root=Path(root); directory=root/'data'; directory.mkdir(exist_ok=True)
    tables=load_tables(root/'cache/pricing/heston.npz')
    validity=json.loads((root/'cache/pricing/heston.validation.json').read_text())
    if not validity['passed'] or validity['sha256'] != table_hash(tables): raise ValueError('Pricing validity gate failed or table hash mismatch')
    if (directory/'manifest.json').exists():
        existing=json.loads((directory/'manifest.json').read_text())
        if existing['pricing_hash']!=table_hash(tables): raise ValueError('Stale dataset pricing hash')
        for record in existing['datasets'].values():
            if file_hash(root/record['path'])!=record['sha256']: raise ValueError('Stale dataset content')
        if file_hash(root/'objective_scales.json')!=existing['objective_scales_hash']: raise ValueError('Stale scales')
        return existing
    fallback=json.loads((root/'cache/pricing/fallback.validation.json').read_text())
    if not fallback['passed'] or fallback['pricing_hash']!=table_hash(tables) or fallback['pricing_source_hash']!=file_hash(root/'src/pricing.py'):
        raise ValueError('Fallback pricing validation missing or stale')
    qs=mixture_probabilities()
    manifest={'pricing_hash':table_hash(tables),'market':MARKET_CONFIG,'regimes':REGIMES,
        'distributions':{d:np.asarray(q).tolist() for d,q in zip(DOMAINS,qs)},'datasets':{}}
    scales={}
    for role,n in ROLES.items():
        for i,(domain,q) in enumerate(zip(DOMAINS,qs)):
            if role=='feedback' and domain=='base': continue
            key=stream_key(role,i); path=directory/f'{role}_{domain}.npz'
            pieces=[]
            for start in range(0,n,256):
                batch=generate(q,jax.random.fold_in(key,start//256),min(256,n-start),tables)
                if not bool(jnp.isfinite(batch['call']).all()): raise ValueError(f'Invalid prices in {role}/{domain}')
                batch=jax.device_get(batch)
                pieces.append(batch)
            batch={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
            # IDs identify role/domain/index; never accepted by training generator.
            batch['episode_id']=np.arange(n,dtype=np.int64)+(list(ROLES).index(role)+1)*1000000+i*100000
            temporary=path.with_name(path.name+'.tmp')
            with temporary.open('wb') as stream:np.savez_compressed(stream,**batch)
            temporary.replace(path)
            manifest['datasets'][f'{role}_{domain}']={'path':str(path.relative_to(root)),
                'sha256':file_hash(path),'count':n,'key':np.asarray(jax.random.key_data(key)).tolist(),
                'role':role,'domain':domain,'id_min':int(batch['episode_id'].min()),'id_max':int(batch['episode_id'].max())}
            if role=='feedback' or (role=='monitor' and domain=='base'):
                s=jnp.asarray(batch['spot']); loss=(jnp.maximum(s[:,-1]-s[:,64],0)/s[:,64]-base_premium_fraction())**2
                scales[domain]=max(float(jnp.mean(loss)),1e-12)
    manifest['benchmark_hash']=canonical_hash({k:v for k,v in manifest.items() if k!='datasets'})
    (root/'objective_scales.json').write_text(json.dumps({'scales':scales,'floor':1e-12,'reference':'zero holdings; fixed premium; normalized by inception spot','sources':{'base':'monitor_base','easy':'feedback_easy','hard':'feedback_hard'}},indent=2))
    manifest['objective_scales_hash']=file_hash(root/'objective_scales.json')
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest

def load_dataset(root,role,domain,*,allow_final=False):
    root=Path(root)
    if role=='final_test' and (not allow_final or not (root/'results/frozen_selection.json').exists()):
        raise PermissionError('Final tests sealed until selection manifest is frozen')
    manifest=json.loads((root/'data/manifest.json').read_text())
    if role=='final_test':
        frozen=json.loads((root/'results/frozen_selection.json').read_text())
        if not frozen.get('selected') or frozen.get('benchmark_hash')!=manifest['benchmark_hash'] or frozen.get('objective_scales_hash')!=manifest['objective_scales_hash']:
            raise PermissionError('Final selection missing or does not bind this benchmark/scales')
    record=manifest['datasets'][f'{role}_{domain}']; path=root/record['path']
    if file_hash(path)!=record['sha256']: raise ValueError('Dataset content hash mismatch')
    with np.load(path) as f: data={k:f[k] for k in f.files}
    return data,record
