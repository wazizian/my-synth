"""Metadata-only checks, safe on the supervisor; imports no numerical stack."""
import sys,tempfile,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.budget import Budget
with tempfile.TemporaryDirectory() as d:
    b=Budget(d);first=b.reserve('a',60,.6,extras=0)
    try:b.reserve('too-expensive',60,2000,extras=0)
    except RuntimeError:pass
    else:raise AssertionError('Overbudget reservation accepted')
    try:b.reserve('too-long',86400,.6,extras=0)
    except RuntimeError:pass
    else:raise AssertionError('Late reservation accepted')
    b.reconcile(first,.01)
    s=json.loads((Path(d)/'budget/reservations.json').read_text())
    assert s[first]['status']=='reconciled'
print('PASS: metadata-only budget reservation/rejection/reconciliation')
