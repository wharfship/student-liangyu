"""Bounded public metadata and official WSI checkpoint probe; no paid APIs."""
import requests,time,concurrent.futures
from common import *

def search(query):
 url='https://api.openalex.org/works';r=requests.get(url,params={'search':query,'per-page':8},timeout=35);r.raise_for_status()
 return dict(query=query,results=[dict(title=x['title'],year=x['publication_year'],doi=x['doi'],url=x['id'],
   primary_location=x.get('primary_location')) for x in r.json()['results']])

def main():
 records=[]
 for query in ['VISTA visual reasoning agent active perception','VISTA visual tool agent segmentation']:
  try:records.append(search(query))
  except Exception as e:records.append(dict(query=query,error=str(e)))
 save(HERE/'vista_metadata_search.json',records)
 url='https://drive.usercontent.google.com/download?id=1Rb00BRu3adhEVt47ciVZ3WCWQ938eQyV&export=download&confirm=t'
 tick=time.monotonic();rec=dict(url=url,scope='official Jiang WSI pretrained link; availability probe')
 p=HERE/'assets/jiang_wsi_checkpoint.download'
 try:
  with requests.get(url,stream=True,timeout=(20,40)) as r:
   rec.update(http_status=r.status_code,headers=dict(r.headers));r.raise_for_status()
   ctype=r.headers.get('Content-Type','');assert 'html' not in ctype,'HTML response, no model claimed'
   with p.with_suffix('.part').open('wb') as f:
    for chunk in r.iter_content(2**20):
     assert time.monotonic()-tick<300,'bounded300s';f.write(chunk)
   p.with_suffix('.part').replace(p);rec.update(status='downloaded',bytes=p.stat().st_size,sha256=sha(p))
 except Exception as e:rec.update(status='unavailable_this_probe',error=str(e))
 rec['seconds']=time.monotonic()-tick;save(HERE/'jiang_probe.json',rec);print(json.dumps(rec,ensure_ascii=False)[:2000])
 for r in records:
  print(r['query'],[(x['title'],x['year']) for x in r.get('results',[])])

if __name__=='__main__':main()
