from pathlib import Path
import json,urllib.request,time,zipfile
from prepare import HERE,save,sha

out=HERE/'specialist_assets';out.mkdir(exist_ok=True)
records=[]
for name,repo in [('detectron2','facebookresearch/detectron2'),('mescnn','Nicolik/MESCnn')]:
    try:
        headers={'User-Agent':'glomerulus-research'}
        api='https://api.github.com/repos/'+repo+'/commits/HEAD'
        with urllib.request.urlopen(urllib.request.Request(api,headers=headers),timeout=40) as r:
            data=json.load(r)
        commit=data['sha'];url='https://codeload.github.com/'+repo+'/zip/'+commit
        file=out/(name+'_'+commit+'.zip')
        with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=60) as r:
            file.write_bytes(r.read())
        with zipfile.ZipFile(file) as a:
            assert a.testzip() is None
        records.append(dict(name=name,repo=repo,commit=commit,url=url,path=str(file),sha256=sha(file),status='downloaded'))
        print(name,commit,file.stat().st_size,flush=True)
    except Exception as e:
        records.append(dict(name=name,repo=repo,status='failed',error=repr(e)))
save(out/'source_receipt.json',records)
