"""Package all source fragments, preserving validation-v1 submitted source."""
from pathlib import Path
import json,hashlib,ast
root=Path(__file__).resolve().parent
names=['worker.py','reference_fragment.py','teacher_fragment.py','protocol_fragment.py','gate_fragment.py','residual_fragment.py','adaptive_fragment.py','report_fragment.py']
source=(root/'worker.py').read_text().split('def main():')[0]
for name in names[1:]: source+='\n\n'+(root/name).read_text()
source+='''
def main():
    assert all(d.platform=='gpu' for d in jax.devices()),'GPU required'
    save(OUT/'design.json',D)
    save(OUT/'numerics.json',NUMERICS)
    save(OUT/'residual_review_protocol.json',RESIDUAL_PROTOCOL)
    (OUT/'actual_source.py').write_bytes(pathlib.Path(__file__).read_bytes())
    save(OUT/'actual_source_sha256.json',dict(sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()))
    parser=argparse.ArgumentParser(); parser.add_argument('--mode',default='validate',choices=['validate','gate','campaign']); args=parser.parse_args()
    if args.mode in ('gate','campaign'):
        run_gate()
        if args.mode=='campaign' and json.loads((OUT/'gate_summary.json').read_text())['accepted']:
            run_final()
        exec(AUDIT_SOURCE,{'__name__':'independent_audit'})
        make_report()
        return
    reports=[]
    for family in D['families']:
        path=OUT/family['name']; path.mkdir(exist_ok=True)
        print('BUILD PRICER',family['name'],flush=True); pricer=Pricer(family)
        numerical=validate(family,pricer,path)
        independent=independent_prices(family,pricer,path)
        objectives=validate_objectives(family,path)
        reports.append(dict(family=family['name'],numerical=numerical,independent=independent,objectives=objectives,passed=numerical['passed'] and independent['passed'] and objectives['passed']))
    save(OUT/'validation_summary.json',dict(passed=all(x['passed'] for x in reports),families=reports))
if __name__=='__main__':
    try: main()
    except Exception:
        save(OUT/'failure.json',dict(traceback=traceback.format_exc())); raise
'''
source=source.replace("json.loads('__DESIGN_JSON__')",'json.loads('+repr((root/'design.json').read_text())+')')
source='NUMERICS = '+repr(json.loads((root/'numerics.json').read_text()))+'\n'+source
audit=(root/'independent_audit.py').read_text().replace("complete=(ROOT/'artifact_hashes.json').exists()",'complete=True').replace('raise SystemExit(bool(failures))','assert not failures, failures')
source='AUDIT_SOURCE = '+repr(audit)+'\n'+source
source='RESIDUAL_PROTOCOL = '+repr(json.loads((root/'residual_review_protocol.json').read_text()))+'\n'+source
ast.parse(source)
(root/'worker_full.py').write_text(source)
hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names+['design.json','numerics.json','worker_full.py']}
(root/'source_full_hashes.json').write_text(json.dumps(hashes,indent=2))
print('Packaged full worker:',hashes['worker_full.py'])
