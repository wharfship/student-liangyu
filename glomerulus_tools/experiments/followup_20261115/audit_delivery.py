"""Verify local evidence, JPEG transport and decision contracts; no inference."""
import copy
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
from PIL import Image
from playwright.sync_api import sync_playwright
from common import *
from score_decisions import validate


class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.images=[]
 def handle_starttag(self,tag,attrs):
  d=dict(attrs)
  if tag=='a' and 'href' in d:self.links.append(d['href'])
  if tag=='img':self.images.append(d['src'])


def main():
 pages=['人工复核.html','三例遗漏与旧工具.html','金字塔配对.html','无GT观察样例.html']
 results=[]
 for name in pages:
  page=REPORT/name;p=Links();p.feed(page.read_text(encoding='utf8'))
  for link in p.links+p.images:
   u=urlsplit(link)
   if u.scheme or not u.path:continue
   asset=(page.parent/unquote(u.path)).resolve();assert asset.is_file(),asset
  for link in set(p.images):
   asset=(page.parent/unquote(urlsplit(link).path)).resolve()
   assert asset.suffix.lower() in {'.jpg','.jpeg'}
   with Image.open(asset) as im:assert im.format=='JPEG';im.verify()
  results.append(dict(page=name,images=len(p.images),links=len(p.links),all_targets_exist=True,JPEG_valid=True))
 packages=read(HERE/'observation_packages_NO_GT.json');assert len(packages)==77
 for p in packages:
  assert p['GT_read'] is False and len(p['views'])==7
  for v in p['views'].values():
   asset=Path(v['path']);assert asset.suffix.lower()=='.jpg' and sha(asset)==v['sha256']
 ds=[dict(candidate_id=p['candidate_id'],decision='keep',views_requested=[],reason='contract check') for p in packages]
 validate(ds,packages)
 mutations={
  'unknown_id':lambda x:x[0].update(candidate_id='UNKNOWN'),
  'duplicate_id':lambda x:x[0].update(candidate_id=x[1]['candidate_id']),
  'extra_GT_field':lambda x:x[0].update(target_dice=1),
  'invalid_decision':lambda x:x[0].update(decision='replace_all'),
  'unknown_view':lambda x:x[0].update(views_requested=['GT']),
  'duplicate_view':lambda x:x[0].update(views_requested=[packages[0]['allowed_view_names'][0]]*2),
  'over_budget':lambda x:x[0].update(views_requested=packages[0]['allowed_view_names'][:4]),
  'missing_record':lambda x:x.pop(),
  'empty_reason':lambda x:x[0].update(reason=''),
 }
 rejected=[]
 for name,mutate in mutations.items():
  bad=copy.deepcopy(ds);mutate(bad)
  try:validate(bad,packages)
  except ValueError:rejected.append(name)
  else:raise AssertionError(name)
 browser_checks=[]
 with sync_playwright() as p:
  browser=p.chromium.launch(executable_path='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless=True)
  for name in pages:
   page=browser.new_page(viewport={'width':1600,'height':1000});errors=[]
   page.on('pageerror',lambda e:errors.append(str(e)))
   page.goto((REPORT/name).as_uri(),wait_until='load')
   if name=='金字塔配对.html':
    assert page.locator('.case').count()==80
    page.select_option('#g','sclerosis');assert page.locator('.case:visible').count()==40
    page.fill('#q','S019');assert page.locator('.case:visible').count()==1
    page.wait_for_function("Array.from(document.querySelectorAll('#S019 img')).every(e => e.complete && e.naturalWidth > 0)")
    page.screenshot(path=str(HERE/'paired_browser_check.png'))
    page.fill('#q','');page.select_option('#g','normal_control');assert page.locator('.case:visible').count()==40
   else:
    assert page.locator('section').count()=={'人工复核.html':40,'三例遗漏与旧工具.html':3,'无GT观察样例.html':77}[name]
   assert page.locator('img').first.evaluate('(e) => e.complete && e.naturalWidth > 0')
   assert not errors,errors
   browser_checks.append(dict(page=name,javascript_errors=errors,first_image_rendered=True))
   page.close()
  browser.close()
 save(HERE/'delivery_verification.json',dict(status='passed',html_assets=results,browser=browser_checks,
  observation_packages=77,named_JPEG_SHA_verified=539,invalid_decision_cases_rejected=rejected,
  no_VLM_calls=True,no_GT_sent=True,scope='contracts and evidence integrity, not segmentation accuracy or policy efficacy'))
 print('Passed: 4 HTML pages, all referenced assets, 539 named JPEG SHAs,',len(rejected),'invalid decision contracts, Edge rendering and filters')


if __name__=='__main__':main()
