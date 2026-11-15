"""Bounded author-source retrieval, with explicit HTTP errors and archived bytes."""
import requests,concurrent.futures,time
from common import *
QUERIES={
 'grandqc_search':'https://api.github.com/search/repositories?q=GrandQC',
 'instanseg_repo':'https://api.github.com/repos/instanseg/instanseg',
 'instanseg_readme':'https://raw.githubusercontent.com/instanseg/instanseg/main/README.md',
 'instanseg_tree':'https://api.github.com/repos/instanseg/instanseg/git/trees/main?recursive=1',
 'instanseg_releases':'https://api.github.com/repos/instanseg/instanseg/releases',
 'phikon_card':'https://huggingface.co/owkin/phikon/raw/main/README.md',
 'phikon_api':'https://huggingface.co/api/models/owkin/phikon',
}
def fetch(item):
 name,url=item;t=time.monotonic()
 try:
  r=requests.get(url,timeout=(10,25));p=HERE/'sources'/(name+'.txt');p.parent.mkdir(exist_ok=True);p.write_bytes(r.content)
  rec=dict(name=name,url=url,status=r.status_code,bytes=len(r.content),sha256=sha(p),seconds=time.monotonic()-t)
  print(name,r.status_code,len(r.content),flush=True)
  if name=='grandqc_search' and r.ok:print([(x['full_name'],x['html_url']) for x in r.json().get('items',[])[:5]],flush=True)
  if name=='instanseg_releases' and r.ok:print([(x['tag_name'],[(a['name'],a['size']) for a in x['assets']]) for x in r.json()[:3]],flush=True)
  return rec
 except Exception as e:print(name,type(e).__name__,flush=True);return dict(name=name,url=url,error=type(e).__name__+': '+str(e),seconds=time.monotonic()-t)
def main():
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(fetch,QUERIES.items()))
 save(HERE/'discovery.json',records)
if __name__=='__main__':main()
