"""Paired grid-phase intervention, original SegFormer, image/prediction only."""
from pathlib import Path
import os,sys,json,subprocess,time,hashlib
import numpy as np
from PIL import Image

P=Path('/home/chenliangyu/project/new_agent');EXP=Path(__file__).resolve().parent
OUT=EXP/'seam_validation';OUT.mkdir(exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2))
def main():
    assert subprocess.check_output(['hostname'],text=True).strip()=='netxge'
    assert subprocess.check_output(['whoami'],text=True).strip()=='chenliangyu'
    resource=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    cards=[tuple(map(int,l.split(','))) for l in resource.splitlines()];cards=[c for c in cards if c[1]>12500];assert cards
    gpu=max(cards,key=lambda x:x[1])[0];os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    import torch
    assert torch.cuda.is_available();torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.35,0)
    torch.manual_seed(20261105);np.random.seed(20261105);torch.backends.cudnn.benchmark=False
    sys.path.insert(0,str(P/'nepathtk_tools_20261105'))
    from run import load,probabilities,starts
    model,card=load('segformer_b5');plans=json.loads((EXP/'seam_plan_NO_GT.json').read_text());rows=[];cache_shas={}
    for case in plans['cases']:
        tick=time.monotonic();tid=case['task_id'];cid=case['case_id'];dest=OUT/cid;dest.mkdir(exist_ok=False)
        frozen=json.loads((P/'full_networks_20261105/predictions/segformer_b5'/tid/'frozen.json').read_text())
        assert sha(P/'nepathtk_tools_20261105/run.py')==frozen['network_code_sha256']
        assert card['sha256']==frozen['checkpoint_sha256']
        assert sha(P/'nepathtk_tools_20261105'/card['config'])==frozen['config_sha256']
        cache=P/'full_networks_20261105/image_cache'/case['cache_key']/'rgb.npy'
        if tid not in cache_shas:cache_shas[tid]=sha(cache)
        assert cache_shas[tid]==frozen['cache_sha256']
        rgb=np.load(cache,mmap_mode='r');h,w=rgb.shape[:2];nw,nh=frozen['native_size'];x,y,xx,yy=case['box']
        nx=np.arange(x,xx,dtype=np.int64)*w//nw;ny=np.arange(y,yy,dtype=np.int64)*h//nh
        l,t,r,b=int(nx.min()),int(ny.min()),int(nx.max())+1,int(ny.max())+1
        size=card['tile_pixels'];stride=card['stride_pixels'];reports=[];original=None
        for phase in [0,256]:
            xs=starts(w,size,stride) if phase==0 else sorted(set([0,max(0,w-size)]+list(range(phase,max(phase,w-size+1),stride))))
            ys=starts(h,size,stride) if phase==0 else sorted(set([0,max(0,h-size)]+list(range(phase,max(phase,h-size+1),stride))))
            xs=[v for v in xs if v<r and v+size>l];ys=[v for v in ys if v<b and v+size>t]
            sums=np.zeros((4,b-t,r-l),np.float32);count=np.zeros((b-t,r-l),np.uint16);tiles=0
            with torch.inference_mode():
                for ty in ys:
                    for tx in xs:
                        crop=np.array(rgb[ty:ty+size,tx:tx+size]);ch,cw=crop.shape[:2]
                        tile=np.full((size,size,3),255,np.uint8);tile[:ch,:cw]=crop
                        p=probabilities(model,card,tile);ll,tt,rr,bb=max(l,tx),max(t,ty),min(r,tx+cw),min(b,ty+ch)
                        sums[:,tt-t:bb-t,ll-l:rr-l]+=p[:,tt-ty:bb-ty,ll-tx:rr-tx]
                        count[tt-t:bb-t,ll-l:rr-l]+=1;tiles+=1
            assert count.min()>0
            working=(sums/count[None]).argmax(0)==1
            native=working[np.ix_(ny-t,nx-l)]
            Image.fromarray(native.astype('uint8')*255).save(dest/f'phase{phase}.png')
            if phase==0:
                old=np.load(P/'full_networks_20261105/predictions/segformer_b5'/tid/'working_mask.npy',mmap_mode='r')
                exact=bool(np.array_equal(working,old[t:b,l:r]>0));assert exact,(cid,'old-grid mismatch')
                original=native
            reports.append(dict(phase=phase,tiles=tiles,coverage_min=int(count.min()),foreground=int(native.sum())))
        rec=dict(case_id=cid,task_id=tid,reports=reports,original_grid_exact=True,GT_read=False,
                 changed_pixels=int(np.count_nonzero(original!=native)),seconds=time.monotonic()-tick)
        save(dest/'record.json',rec);rows.append(rec);save(OUT/'progress.json',dict(completed=len(rows),total=len(plans['cases'])))
        print(json.dumps(rec),flush=True)
    save(OUT/'completed.json',dict(status='complete',cases=len(rows),records=rows,gpu=gpu,resource_check=resource,
         snapshot=str(EXP/'environment_snapshot.json'),torch=torch.__version__,torch_cuda=torch.version.cuda,
         script_sha256=sha(__file__),GT_read=False,model_sha256=card['sha256']))

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(OUT/'exit.json',dict(exit_code=1,error=repr(e)));raise
    else:save(OUT/'exit.json',dict(exit_code=0))
