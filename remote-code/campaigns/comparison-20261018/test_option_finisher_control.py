"""GPU-host stdlib proof of actual idle-finisher control statements."""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace
source=Path(__file__).with_name('finish_recovered.py')
tree=ast.parse(source.read_text())
harden=next(n for n in tree.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='run')
branch=next(n for n in tree.body if isinstance(n,ast.If) and isinstance(n.test,ast.UnaryOp) and isinstance(n.test.operand,ast.Name) and n.test.operand.id=='complete')
body=[]
for node in branch.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='complete' for t in node.targets): break
    body.append(node)
for halt in (False,True):
    calls=[]
    def child(args,**kwargs):
        calls.append(('child',args))
        assert kwargs=={'check':True}
    def funding():
        calls.append(('funding',))
        if halt: raise RuntimeError('funding halt')
    def exclusions():
        calls.append(('exclusions',))
        return ['--exclude-offer','123456789']
    env={'C':Path('/campaign/volatility'),'l':Path('/campaign/volatility/launchers/evaluation'),'sys':sys,'subprocess':SimpleNamespace(run=child),'assert_funding_available':funding,'exclusion_arguments':exclusions}
    code=ast.fix_missing_locations(ast.Module(body=[harden]+body,type_ignores=[]))
    try:exec(compile(code,'actual-finisher-controls','exec'),env)
    except RuntimeError:assert halt
    else:assert not halt
    assert calls[0][0]=='child' and 'harden_export.py' in calls[0][1][1]
    assert calls[0][1][0]==sys.executable
    assert [row[0] for row in calls][:3]==['child','exclusions','funding']
    if halt:assert len(calls)==3
    else:
        args=calls[3][1]
        assert args[0]==sys.executable
        assert args[-4:]==['--exclude-offer','123456789','--','evaluation']
        assert args[args.index('--timeout-seconds')+1]=='7200'
        assert args[args.index('--max-hourly')+1]=='.60'
        assert args[args.index('--job')+1]=='/campaign/volatility/evaluation_worker.py'
print(json.dumps({'status':'PASS','actual_hardener_before_allocation':True,'funding_halt_blocks_allocation':True,'fresh_exclusions_before_workload_separator':True,'sys_executable_no_uv_dependency':True,'scientific_arguments_unchanged':True}),flush=True)
