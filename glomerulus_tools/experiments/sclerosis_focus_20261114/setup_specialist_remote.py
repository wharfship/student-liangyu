"""Isolated Detectron2 build; existing sam packages are inherited, never modified."""
from pathlib import Path
import json,os,subprocess,sys,time,zipfile

P=Path('/home/chenliangyu/project/new_agent/glomerulus_tools/experiments/sclerosis_focus_20261114')
OUT=P/'specialist_assets';LOG=P/'specialist_setup.log'
def main():
    assert subprocess.check_output(['whoami'],text=True).strip()=='chenliangyu'
    assert subprocess.check_output(['hostname'],text=True).strip()=='netxge'
    records=[]
    envdir=P/'specialist_venv'
    cmd=[sys.executable,'-m','venv','--system-site-packages',str(envdir)]
    subprocess.run(cmd,check=True)
    for name in ['detectron2','mescnn']:
        z=next(OUT.glob(name+'_*.zip'))
        with zipfile.ZipFile(z) as a:
            a.extractall(OUT,filter='data') if sys.version_info>=(3,12) else a.extractall(OUT)
    source=next(OUT.glob('detectron2-*/setup.py')).parent
    python=envdir/'bin/python'
    cmds=[['nvidia-smi'],['/usr/bin/nvcc','--version'],
          [str(python),'-m','pip','install','--disable-pip-version-check','--index-url','https://pypi.tuna.tsinghua.edu.cn/simple',
           'fvcore>=0.1.5,<0.1.6','iopath>=0.1.7,<0.1.10','yacs','omegaconf','hydra-core','black','tensorboard'],
          [str(python),'-m','pip','install','--no-deps','--no-build-isolation','--disable-pip-version-check',str(source)]]
    # CPU-built extension suffices for standard ResNet/FPN non-rotated operations;
    # actual CUDA model and torchvision operator checks are required before inference.
    env=dict(os.environ,MAX_JOBS='2',CUDA_VISIBLE_DEVICES='',FORCE_CUDA='0')
    for cmd in cmds:
        start=time.time();r=subprocess.run(cmd,env=env)
        records.append(dict(command=cmd,exit_code=r.returncode,start=start,end=time.time()))
        (P/'specialist_setup_runs.json').write_text(json.dumps(records,indent=2))
        assert r.returncode==0,cmd
    (P/'specialist_setup_complete.json').write_text(json.dumps(dict(status='complete',python=str(python),source=str(source)),indent=2))

if __name__=='__main__':main()
