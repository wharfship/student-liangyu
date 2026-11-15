from pathlib import Path
import sys,json,hashlib,subprocess,base64
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
REPORT=ROOT/'已经整理好的内容/医院分割结果分析/开源工具扩展_20261115'
REMOTE='/home/chenliangyu/project/new_agent'

def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def remote(code,timeout=90):
 data=base64.b64encode(code.encode()).decode()
 cmd="/home/chenliangyu/miniconda3/envs/sam/bin/python -c 'import base64;exec(base64.b64decode(\""+data+"\"))'"
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','chenliangyu@10.103.238.124',cmd],capture_output=True,text=True,encoding='utf8',timeout=timeout)
 assert r.returncode==0,r.stderr
 return r.stdout
