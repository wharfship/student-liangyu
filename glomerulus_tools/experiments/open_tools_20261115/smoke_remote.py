"""Four real CUDA inferences and model-package regression, run on the server."""
from pathlib import Path
import os,subprocess,json,time,sys,hashlib
P=Path(__file__).resolve().parent
PY=str(P/'tool_venv/bin/python')

def main():
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
 available=[int(line.split(',')[0]) for line in gpu.splitlines() if int(line.split(',')[1])>12000]
 assert available,'No free GPU; no CPU fallback';physical=available[0];os.environ['CUDA_VISIBLE_DEVICES']=str(physical)
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 snapshot=P/'execution_snapshot';runs=[]
 for tid,taskname in [('instanseg_brightfield_nuclei_v011','nuclei'),('dinov2_vits14_features','feature'),('grandqc_tissue_mpp10','qc'),('grandqc_artifact_mpp15','qc')]:
  task=json.loads((P/(taskname+'_remote_NO_GT.json')).read_text());task['samples']=task['samples'][:1]
  path=P/(tid+'_smoke_NO_GT.json');path.write_text(json.dumps(task))
  cmd=[PY,'-u','-m','glomerulus_tools','run','--tool',tid,'--task',str(path),'--output',str(P/'smoke'/tid),'--research']
  begin=time.monotonic();r=subprocess.run(cmd,cwd=snapshot,env=os.environ.copy());runs.append(dict(tool=tid,exit_code=r.returncode,seconds=time.monotonic()-begin,command=cmd))
  (P/'smoke_runs.json').write_text(json.dumps(runs,indent=2));assert r.returncode==0,tid
 # The official test-output is a reference prediction, not an expert GT.
 import torch,numpy as np
 sys.path.insert(0,str(next((P/'assets/source').glob('instanseg-*'))))
 from instanseg.utils.utils import percentile_normalize
 model=torch.jit.load(str(P/'assets/instanseg_model/instanseg.pt'),map_location='cuda').eval()
 raw=torch.from_numpy(np.load(P/'assets/instanseg_model/test-input.npy')).cuda()
 x=torch.stack([percentile_normalize(i) for i in raw])
 with torch.inference_mode():pred=model(x).cpu().numpy()
 ref=np.load(P/'assets/instanseg_model/test-output_instance_segmentation.npy')
 same=np.array_equal(pred,ref);binary=float(np.mean((pred>0)==(ref>0)))
 np.save(P/'reference_prediction.npy',pred)
 data=dict(status='complete',runs=runs,gpu=physical,reference_exact_label_equal=same,reference_binary_agreement=binary,
  reference_is_prediction_not_GT=True,GT_read=False)
 (P/'smoke_complete.json').write_text(json.dumps(data,indent=2));print(json.dumps(data),flush=True)

if __name__=='__main__':main()
