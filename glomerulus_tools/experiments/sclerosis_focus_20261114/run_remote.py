"""Run the two unchanged registered recipes against the frozen image-only cohort."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time

EXP=Path(__file__).resolve().parent
SNAP=EXP/'execution_snapshot'
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')

def main():
    assert subprocess.check_output(['hostname'],text=True).strip()=='netxge'
    assert subprocess.check_output(['whoami'],text=True).strip()=='chenliangyu'
    assert not (EXP/'inference_complete.json').exists()
    for file,want in json.loads((SNAP/'manifest.json').read_text()).items():
        assert hashlib.sha256((SNAP/'glomerulus_tools'/file).read_bytes()).hexdigest()==want,file
    batch=json.loads((EXP/'batch_NO_GT.json').read_text())
    for t in batch['cases']:
        for k in ['image','baseline','target']:assert Path(t[k]).is_file()
    # These are input-size smoke checks, selected without evaluation labels.
    ordered=sorted(batch['cases'],key=lambda x:x['native_size'][0]*x['native_size'][1])
    smoke=list({t['case_id']:t for t in [ordered[0],ordered[len(ordered)//2],ordered[-1]]}.values())
    save(EXP/'smoke_NO_GT.json',dict(schema='local_mask_refinement_batch_v1',cases=smoke))
    runs=[]
    for method in ['samrefiner_vitb_paper_fp16','ritm_hrnet32_external_mask_point1']:
        for phase in ['smoke','results']:
            q=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            cards=[tuple(map(int,line.split(','))) for line in q.splitlines()]
            cards=[c for c in cards if c[1]>=12500]
            assert cards,'No GPU with >=12500MiB free; no CPU fallback'
            gpu=max(cards,key=lambda x:x[1])[0]
            out=EXP/phase/method;out.parent.mkdir(exist_ok=True)
            cmd=[sys.executable,'-u','-m','glomerulus_tools','refine','--tool',method,'--task',
                 str(EXP/('smoke_NO_GT.json' if phase=='smoke' else 'batch_NO_GT.json')),'--output',str(out),'--research']
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu));env.pop('PYTHONPATH',None)
            logfile=EXP/(method+'_'+phase+'.log')
            record=dict(method=method,phase=phase,command=cmd,cwd=str(SNAP),gpu=gpu,resource_check=q,log=str(logfile),start=time.time())
            with logfile.open('w') as log:
                p=subprocess.Popen(cmd,cwd=SNAP,env=env,stdout=log,stderr=subprocess.STDOUT)
                record['pid']=p.pid;runs.append(record);save(EXP/'runs.json',runs)
                record['exit_code']=p.wait();record['end']=time.time()
            save(EXP/'runs.json',runs)
            assert record['exit_code']==0,logfile.read_text()[-3500:]
            print(json.dumps(record),flush=True)
    save(EXP/'inference_complete.json',dict(status='complete',cases=len(batch['cases']),methods=2,GT_read=False,runs=runs))

if __name__=='__main__':main()
