"""Frozen scientific protocol and disjoint, auditable PRNG namespaces."""
import hashlib
import json
from pathlib import Path
import jax

ARMS = ('baseline','hard_reference','dpg_hard','soar_hard','selfplay','dpg_mixed','soar_mixed')
DOMAINS = ('base','easy','hard')
ROLES = {'feedback':256,'monitor':512,'selection':1024,'final_test':8192}
SEED = 20261017
STREAMS = {'initialization':1,'training_regimes':2,'training_noise':3,
           'feedback':10,'monitor':20,'selection':30,'final_test':40,'teacher':50}

def stream_key(role, index=0):
    return jax.random.fold_in(jax.random.fold_in(jax.random.key(SEED),STREAMS[role]),index)

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def source_hash(root):
    return canonical_hash({str(p.relative_to(root)):file_hash(p) for folder in ('src','scripts','configs') for p in sorted((Path(root)/folder).rglob('*')) if p.is_file() and '__pycache__' not in str(p)})

DEFAULT_CONFIG = dict(seed=SEED,batch_size=256,block_steps=4,hidden_size=64,
    learner_lr=.001,generator_lr=.001,kl_coefficient=.01,epsilon=.05,
    expert_coefficient=1.,monitor_interval_seconds=180,checkpoint_interval_seconds=300,
    selection_interval_seconds=900,evaluation_batch_size=256,
    checkpoint_policy='lowest_balanced_selection_score',screen_seconds=600,
    final_active_seconds=2700,task={},market={})
