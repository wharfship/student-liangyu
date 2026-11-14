"""Offline local gallery filter and render verification, no external requests."""
from playwright.sync_api import sync_playwright
from prepare import HERE,REPORT,save

def main():
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless=True)
        page=browser.new_page(viewport={'width':1600,'height':1000});errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto((REPORT/'gallery.html').as_uri(),wait_until='load')
        assert page.locator('.case').count()==80
        page.select_option('#group','sclerosis');assert page.locator('.case:visible').count()==40
        page.fill('#query','S019');assert page.locator('.case:visible').count()==1
        page.locator('#S019 img').first.wait_for();page.screenshot(path=str(HERE/'gallery_browser_check.png'))
        page.fill('#query','');page.select_option('#group','normal_control');assert page.locator('.case:visible').count()==40
        page.select_option('#group','ALL');assert page.locator('.case:visible').count()==80
        assert not errors,errors
        save(HERE/'browser_verification.json',dict(browser='Edge headless Playwright',all80=True,group40=True,search1=True,
            javascript_errors=errors,offline_local=True))
        browser.close()

if __name__=='__main__':main()
