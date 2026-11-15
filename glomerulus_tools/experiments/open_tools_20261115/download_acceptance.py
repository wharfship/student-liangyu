from assets_research import download
from common import *
import zipfile,concurrent.futures,hashlib

def main():
 jobs=[('https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth','dinov2_vits14_pretrain.pth'),
  ('https://codeload.github.com/facebookresearch/dinov2/zip/'+read(HERE/'sources/dinov2_repo.txt')['sha'],'dinov2_source.zip')]
 t=read(HERE/'sources/tnbc_meta.txt')['files'][0];jobs.append((t['links']['self'],'TNBC_acceptance.zip'))
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(lambda j:download(*j),jobs))
 assert 'md5:'+hashlib.md5((HERE/'assets/TNBC_acceptance.zip').read_bytes()).hexdigest()==t['checksum']
 for r in records:
  if r['file'].endswith('.zip'):
   with zipfile.ZipFile(HERE/'assets'/r['file']) as z:
    assert all(not n.startswith('/') and '..' not in Path(n).parts for n in z.namelist())
    z.extractall(HERE/'assets'/('acceptance' if 'TNBC' in r['file'] else 'source'))
 save(HERE/'acceptance_download_receipts.json',records)
 print('Reference assets ready; TNBC is part of InstanSeg training corpus, not independent test')
if __name__=='__main__':main()
