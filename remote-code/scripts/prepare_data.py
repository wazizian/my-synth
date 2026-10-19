from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import require_gpu
require_gpu()
from src.datasets import prepare
if __name__=='__main__':
    m=prepare(Path(__file__).resolve().parents[1]); print('Prepared',len(m['datasets']),'immutable datasets')
