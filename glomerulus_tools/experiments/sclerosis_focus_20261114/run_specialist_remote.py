from pathlib import Path
import json,os,subprocess,sys,time,traceback

P=Path(__file__).resolve().parent
SNAP=P/'specialist_execution_snapshot'
def main():
    assert subprocess.check_output(['hostname'],text=True).strip()=='netxge'
    assert subprocess.check_output(['whoami'],text=True).strip()=='chenliangyu'
    batch=json.loads((P/'specialist_batch_NO_GT.json').read_text())
    cases=batch['cases'];smoke=[cases[0],max(cases,key=lambda t:t['native_size'][0])]
    (P/'specialist_smoke_NO_GT.json').write_text(json.dumps(dict(schema=batch['schema'],cases=smoke)))
    runs=[]
    for phase in ['smoke','results']:
        resource=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        cards=[tuple(map(int,l.split(','))) for l in resource.splitlines()];cards=[c for c in cards if c[1]>12500];assert cards,resource
        gpu=max(cards,key=lambda c:c[1])[0]
        cmd=[sys.executable,'-u','-m','glomerulus_tools','run','--tool','mescnn_cascade_r50',
             '--task',str(P/('specialist_smoke_NO_GT.json' if phase=='smoke' else 'specialist_batch_NO_GT.json')),
             '--output',str(P/('specialist_'+phase)),'--research']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu));env.pop('PYTHONPATH',None)
        record=dict(phase=phase,command=cmd,cwd=str(SNAP),gpu=gpu,resource=resource,start=time.time())
        with (P/('specialist_'+phase+'.log')).open('w') as log:
            proc=subprocess.Popen(cmd,cwd=SNAP,env=env,stdout=log,stderr=subprocess.STDOUT);record['pid']=proc.pid;runs.append(record)
            (P/'specialist_runs.json').write_text(json.dumps(runs,indent=2));record['exit_code']=proc.wait();record['end']=time.time()
        (P/'specialist_runs.json').write_text(json.dumps(runs,indent=2))
        assert record['exit_code']==0,(P/('specialist_'+phase+'.log')).read_text()[-4000:]
    (P/'specialist_inference_complete.json').write_text(json.dumps(dict(status='complete',cases=len(cases),runs=runs),indent=2))

if __name__=='__main__':
    try:main()
    except Exception as e:
        (P/'specialist_exit.json').write_text(json.dumps(dict(exit_code=1,error=repr(e))));raise
    else:(P/'specialist_exit.json').write_text(json.dumps(dict(exit_code=0)))
