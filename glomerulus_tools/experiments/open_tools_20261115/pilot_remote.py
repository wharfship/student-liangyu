"""Frozen small-batch acceptance; sequential CUDA jobs under one checked GPU."""
from pathlib import Path
import os,json,subprocess,time
P=Path(__file__).resolve().parent

def main():
 assert json.loads((P/'smoke_complete.json').read_text())['status']=='complete'
 free=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
 candidates=[int(s.split(',')[0]) for s in free.splitlines() if int(s.split(',')[1])>12000]
 assert candidates,'No free GPU';gpu=candidates[0];os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 py=str(P/'tool_venv/bin/python');runs=[]
 for tid,taskname,outname in [('instanseg_brightfield_nuclei_v011','nuclei','nuclei_hospital16'),
  ('instanseg_brightfield_nuclei_v011','tnbc','nuclei_TNBC8'),
  ('dinov2_vits14_features','feature','features77'),
  ('grandqc_tissue_mpp10','qc_paired','tissue24'),('grandqc_artifact_mpp15','qc_paired','artifact24')]:
  command=[py,'-u','-m','glomerulus_tools','run','--tool',tid,'--task',str(P/(taskname+'_remote_NO_GT.json')),'--output',str(P/'results'/outname),'--research']
  start=time.monotonic();r=subprocess.run(command,cwd=P/'execution_snapshot',env=os.environ.copy())
  run=dict(tool=tid,task=taskname,output=outname,exit_code=r.returncode,seconds=time.monotonic()-start,command=command,gpu=gpu);runs.append(run)
  (P/'pilot_runs.json').write_text(json.dumps(runs,indent=2));assert r.returncode==0,run
 (P/'pilot_complete.json').write_text(json.dumps(dict(status='complete',runs=runs,GT_read=False),indent=2))
 freeze=subprocess.check_output([py,'-m','pip','freeze'],text=True);(P/'requirements-executed-freeze.txt').write_text(freeze)
 print('All5 formal batches completed',flush=True)

if __name__=='__main__':main()
