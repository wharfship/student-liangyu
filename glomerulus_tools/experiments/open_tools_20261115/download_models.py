from assets_research import download
from common import *
import hashlib,requests,zipfile,concurrent.futures

def main():
 jobs=[]
 for record,file in [('grandqc_zenodo_tissue','Tissue_Detection_MPP10.pth'),('grandqc_zenodo_artifacts','GrandQC_MPP15.pth')]:
  meta=read(HERE/'sources'/(record+'.txt'));f=next(x for x in meta['files'] if x['key']==file);jobs.append((f['links']['self'],file,f['checksum']))
 jobs.append(('https://github.com/kaiko-ai/towards_large_pathology_fms/releases/download/0.0.1/vits16.pth','kaiko_vits16.pth',None))
 jobs.append(('https://codeload.github.com/kaiko-ai/towards_large_pathology_fms/zip/'+read(HERE/'sources/kaiko_tree.txt')['sha'],'kaiko_source.zip',None))
 def work(job):
  url,file,checksum=job
  try:r=download(url,file)
  except Exception as e:return dict(url=url,file=file,error=str(e),status='download_failed_not_runnable')
  if checksum:
   assert 'md5:'+hashlib.md5((HERE/'assets'/file).read_bytes()).hexdigest()==checksum
   r['upstream_md5_matched']=True;r['upstream_md5']=checksum
  return r
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(work,jobs))
 save(HERE/'model_download_receipts.json',records)
 with zipfile.ZipFile(HERE/'assets/kaiko_source.zip') as z:z.extractall(HERE/'assets/source')
 print('Downloads recorded; failures do not count as runnable tools')
if __name__=='__main__':main()
