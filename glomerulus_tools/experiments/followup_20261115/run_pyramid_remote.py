from pathlib import Path
import subprocess,os,json,time
P=Path(__file__).resolve().parent
def main():
 free=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 choices=[tuple(map(int,l.split(','))) for l in free.splitlines()];choices=[c for c in choices if c[1]>18000 and c[2]<10];assert choices
 gpu=max(choices,key=lambda c:c[1])[0];env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu));runs=[]
 python=P.parent/'sclerosis_focus_20261114/specialist_venv/bin/python'
 for mode,ext in [('PNG','png'),('JPG','jpeg')]:
  task=P/('pyramid_'+mode+'_NO_GT.json');assert task.exists()
  cmd=[str(python),'-u','-m','glomerulus_tools','run','--tool','mescnn_cascade_r50_pyramid_'+ext,'--task',str(task),'--output',str(P/('pyramid_results_'+mode)),'--research']
  record=dict(mode=mode,command=cmd,cwd=str(P/'execution_snapshot'),gpu=gpu,resource=free,start=time.time())
  with (P/('pyramid_'+mode+'.log')).open('w') as log:
   proc=subprocess.Popen(cmd,cwd=P/'execution_snapshot',env=env,stdout=log,stderr=subprocess.STDOUT);record['pid']=proc.pid
   result=proc.wait()
  record.update(exit_code=result,end=time.time());runs.append(record);(P/'pyramid_runs.json').write_text(json.dumps(runs,indent=2));assert result==0
 (P/'pyramid_complete.json').write_text(json.dumps(dict(status='complete',runs=runs)))

if __name__=='__main__':main()
