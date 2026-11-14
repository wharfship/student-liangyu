"""Read-only verification of the exact executed source snapshots."""
import json
from inspect_sources import remote_python
from prepare import HERE,ROOT,save,read,sha

CODE='''from pathlib import Path
import hashlib,json,subprocess
p=Path('/home/chenliangyu/project/new_agent');e=p/'glomerulus_tools/experiments/sclerosis_focus_20261114'
def sha(q):return hashlib.sha256(q.read_bytes()).hexdigest()
result={'snapshots':{},'baseline':{}}
for name in ['execution_snapshot','execution_snapshot_memoryfix','specialist_execution_snapshot']:
 d=e/name;manifest=json.loads((d/'manifest.json').read_text())
 for fn,h in manifest.items():assert sha(d/'glomerulus_tools'/fn)==h,(name,fn)
 result['snapshots'][name]=manifest
for name in ['run.py','models.json','configs/segformer_mit-b5_8xb2-160k_kidney_1024x1024_d4.py']:
 result['baseline'][name]=sha(p/'nepathtk_tools_20261105'/name)
result['completions']={name:json.loads((e/name).read_text()).get('status','present') for name in
 ['inference_complete.json','specialist_inference_complete.json','seam_validation/completed.json']}
result['final_gpu_status']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
print(json.dumps(result))
'''

def main():
    r=remote_python(CODE);assert r.returncode==0,r.stderr
    data=json.loads(r.stdout)
    for name,manifest in data['snapshots'].items():
        for fn,h in manifest.items():assert sha(HERE/name/'glomerulus_tools'/fn)==h,(name,fn)
    for fn,h in data['baseline'].items():assert sha(ROOT/'research/nepathtk_tools_20261105'/fn)==h,fn
    data['all_local_execution_sources_equal_remote']=True
    save(HERE/'executed_source_verification.json',data)
    print('3 execution snapshots and baseline sources match local copies')

if __name__=='__main__':main()
