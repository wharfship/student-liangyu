import subprocess,json
from prepare import HERE,read,save
from inspect_sources import remote_python

REMOTE='/home/chenliangyu/project/new_agent/glomerulus_tools/experiments/sclerosis_focus_20261114'
assets=HERE/'specialist_assets'
assert all(r['status']=='downloaded' for r in read(assets/'source_receipt.json'))
assert read(assets/'download_mirror_receipt.json')['status']=='downloaded_and_verified'
r=remote_python("from pathlib import Path;p=Path("+repr(REMOTE)+");(p/'specialist_assets').mkdir(exist_ok=True);print((p/'samrefiner_vitb_paper_fp16_results.log').read_text()[-6500:])")
assert r.returncode==0,r.stderr;print(r.stdout,flush=True)
files=[assets/'mescnn_cascade_r50_internal_model_final.pth',assets/'source_receipt.json',assets/'download_mirror_receipt.json',*assets.glob('*.zip')]
subprocess.run(['scp','-o','BatchMode=yes',*[str(p) for p in files],'chenliangyu@10.103.238.124:'+REMOTE+'/specialist_assets/'],check=True)
subprocess.run(['scp','-o','BatchMode=yes',str(HERE/'setup_specialist_remote.py'),'chenliangyu@10.103.238.124:'+REMOTE+'/'],check=True)
code='''from pathlib import Path
import subprocess,json,sys
p=Path(REMOTE)
assert not (p/'specialist_setup_start.json').exists()
with (p/'specialist_setup.log').open('w') as log:
 proc=subprocess.Popen([sys.executable,'-u',str(p/'setup_specialist_remote.py')],cwd=p,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
rec=dict(pid=proc.pid,log=str(p/'specialist_setup.log'));(p/'specialist_setup_start.json').write_text(json.dumps(rec));print(json.dumps(rec))
'''.replace('REMOTE',repr(REMOTE))
r=remote_python(code);assert r.returncode==0,r.stderr;save(HERE/'specialist_setup_start.json',json.loads(r.stdout));print(r.stdout)
