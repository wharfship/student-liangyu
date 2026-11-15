"""Verify final evidence, ready-client results, JPEGs, browser and source records."""
from common import *
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
from PIL import Image
from playwright.sync_api import sync_playwright
import numpy as np,subprocess

class Parser(HTMLParser):
 def __init__(self):super().__init__();self.images=[];self.links=[]
 def handle_starttag(self,tag,attrs):
  d=dict(attrs)
  if tag=='img':self.images.append(d['src'])
  if tag=='a':self.links.append(d['href'])

def main():
 parser=Parser();parser.feed((REPORT/'gallery.html').read_text(encoding='utf8'))
 for link in parser.images+parser.links:
  part=urlsplit(link)
  if not part.scheme and part.path:assert (REPORT/unquote(part.path)).is_file(),link
 for link in set(parser.images):
  with Image.open(REPORT/link) as im:assert im.format=='JPEG';im.verify()
 with sync_playwright() as p:
  browser=p.chromium.launch(executable_path='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless=True)
  page=browser.new_page(viewport={'width':1500,'height':1100});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  page.goto((REPORT/'gallery.html').as_uri(),wait_until='load');assert page.locator('.case').count()==113
  for key,n in [('nuclei',16),('tnbc',8),('quality',12),('features',77)]:
   page.select_option('#g',key);assert page.locator('.case:visible').count()==n
  page.select_option('#g','tnbc');page.fill('#q','TNBC01');assert page.locator('.case:visible').count()==1
  page.wait_for_function("Array.from(document.querySelectorAll('#tnbc_TNBC01 img')).every(e=>e.complete&&e.naturalWidth>0)")
  page.screenshot(path=str(HERE/'browser_check.png'));assert not errors,errors;browser.close()
 # End-to-end local clients were tested on default and actual upload branches.
 client=[]
 for d in sorted((HERE/'manual_tests').iterdir()):
  r=read(d/'execution_receipt.json');assert r['exit_code']==0 and sha(d/'results.tgz')==r['archive_sha256']
  complete=read(d/'results/completed.json');assert complete['status']=='complete' and complete['samples']==1
  client.append(dict(run_id=r['run_id'],tool=complete['tool_id'],seconds=r['seconds'],exit_code=0))
 assert {'instanseg_brightfield_nuclei_v011','dinov2_vits14_features','grandqc_artifact_mpp15'}<= {r['tool'] for r in client}
 # No-GT query helper is exercised on an actually recorded vector.
 result=subprocess.run([__import__('sys').executable,'-X','utf8',str(HERE/'search_features.py'),'--query',str(HERE/'results/features77/P001/embedding_l2.npy'),'--index',str(HERE/'results/features77'),'--top-k','5'],capture_output=True,text=True,encoding='utf8');assert result.returncode==0,result.stderr
 data=__import__('json').loads(result.stdout);assert len(data['neighbors'])==5 and data['labels_read'] is False and all(r['sample_id']!='P001' for r in data['neighbors'])
 # Preserve both exact executed package and final delivery-source identities.
 snapshot=read(HERE/'execution_snapshot/manifest.json');assert sha(ROOT/'glomerulus_tools/pathology_utilities.py')==snapshot['pathology_utilities.py']
 sources={p.relative_to(ROOT).as_posix():sha(p) for p in HERE.glob('*.py')}
 for p in [HERE/'README.md',ROOT/'glomerulus_tools/pathology_utilities.py',ROOT/'glomerulus_tools/catalog.csv',REPORT/'结果报告.md']:sources[p.relative_to(ROOT).as_posix()]=sha(p)
 save(HERE/'final_source_manifest.json',dict(kind='delivery source; distinct from actual execution snapshot',files=sources))
 save(HERE/'delivery_verification.json',dict(status='passed',gallery_cards=113,JPEG_unique=len(set(parser.images)),all_HTML_targets_exist=True,
  browser='Edge/Playwright',javascript_errors=errors,all4_filters=True,search=True,actual_clients=client,feature_search_NO_GT=True,
  worker_unchanged_from_tested_snapshot=True,formal_samples=226,GUI_and_engineering_not_accuracy=True))
 print('Delivery passed:113 gallery cards, JPEGs,3 real client runs, NO_GT search, exact tested worker')

if __name__=='__main__':main()
