from pathlib import Path
import json,ast,hashlib
root=Path(__file__).resolve().parent
source=(root/'worker.py').read_text().split('def main():')[0]+'\n'+(root/'teacher_fragment.py').read_text()+'\n'+(root/'precision_probe_fragment.py').read_text()
source=source.replace("json.loads('__DESIGN_JSON__')",'json.loads('+repr((root/'design.json').read_text())+')')
source+='\nif __name__=="__main__":\n    assert all(d.platform=="gpu" for d in jax.devices())\n    precision_probe()\n'
ast.parse(source);(root/'precision_probe_packaged.py').write_text(source);print(hashlib.sha256(source.encode()).hexdigest())
