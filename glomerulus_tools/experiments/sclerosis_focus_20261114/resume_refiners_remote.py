from pathlib import Path
import hashlib,json,os,subprocess,sys,time,shutil

P=Path(__file__).resolve().parent
def save(p,x):p.write_text(json.dumps(x,indent=2))
def gpu():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    cards=[tuple(map(int,l.split(','))) for l in text.splitlines()]
    cards=[c for c in cards if c[1]>=12500];assert cards,text
    return max(cards,key=lambda c:c[1])[0],text

def main():
    assert subprocess.check_output(['whoami'],text=True).strip()=='chenliangyu'
    assert subprocess.check_output(['hostname'],text=True).strip()=='netxge'
    failure=P/'engineering_failures/attempt01';failure.mkdir(parents=True,exist_ok=False)
    for name in ['runs.json','controller.log','samrefiner_vitb_paper_fp16_results.log']:
        shutil.copy2(P/name,failure/name)
    old=P/'results/samrefiner_vitb_paper_fp16';assert old.exists() and not (old/'completed.json').exists()
    shutil.move(str(old),str(failure/'samrefiner_vitb_paper_fp16'))
    snap=P/'execution_snapshot_memoryfix';shutil.copytree(P/'execution_snapshot',snap)
    shutil.copy2(P/'mask_refinement_memoryfix.py',snap/'glomerulus_tools/mask_refinement.py')
    manifest={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (snap/'glomerulus_tools').iterdir() if f.is_file()}
    save(snap/'manifest.json',manifest)
    batch=json.loads((P/'batch_NO_GT.json').read_text());runs=[]
    for method in ['samrefiner_vitb_paper_fp16','ritm_hrnet32_external_mask_point1']:
        prior=failure/method
        saved={x.name for x in prior.iterdir() if x.is_dir() and (x/'record.json').exists()} if prior.exists() else set()
        remaining=[t for t in batch['cases'] if t['case_id'] not in saved]
        task=P/(method+'_remaining_NO_GT.json');save(task,dict(schema='local_mask_refinement_batch_v1',cases=remaining))
        physical,resource=gpu();out=P/'results'/method
        cmd=[sys.executable,'-u','-m','glomerulus_tools','refine','--tool',method,'--task',str(task),'--output',str(out),'--research']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(physical),PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True');env.pop('PYTHONPATH',None)
        rec=dict(method=method,command=cmd,cwd=str(snap),gpu=physical,resource=resource,start=time.time(),reused_completed=len(saved),new_cases=len(remaining))
        with (P/(method+'_resume.log')).open('w') as log:
            process=subprocess.Popen(cmd,cwd=snap,env=env,stdout=log,stderr=subprocess.STDOUT);rec['pid']=process.pid;runs.append(rec);save(P/'resume_runs.json',runs)
            rec['exit_code']=process.wait();rec['end']=time.time()
        save(P/'resume_runs.json',runs);assert rec['exit_code']==0
        completed=json.loads((out/'completed.json').read_text())
        if saved:
            for cid in sorted(saved):shutil.copytree(prior/cid,out/cid)
            records=[json.loads((out/t['case_id']/'record.json').read_text()) for t in batch['cases']]
            completed.update(cases=len(records),records=records,seconds=sum(r['seconds'] for r in records)+completed['loading_seconds'],
                             resumed_from=str(prior),reused_completed=len(saved))
            save(out/'completed.json',completed)
        print(json.dumps(rec),flush=True)
    save(P/'inference_complete.json',dict(status='complete',cases=len(batch['cases']),methods=2,GT_read=False,resumed=True,runs=runs))

if __name__=='__main__':main()
