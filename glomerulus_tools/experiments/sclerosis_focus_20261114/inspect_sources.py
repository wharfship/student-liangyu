from pathlib import Path
import csv, importlib.util, json, subprocess, base64

ROOT=Path(__file__).resolve().parents[3]
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def csvread(p): return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))

def remote_python(code, timeout=90):
    data=base64.b64encode(code.encode()).decode()
    command="/home/chenliangyu/miniconda3/envs/sam/bin/python -c 'import base64;exec(base64.b64decode(\""+data+"\"))'"
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','chenliangyu@10.103.238.124',command],capture_output=True,text=True,timeout=timeout)

if __name__=='__main__':
    rows=csvread(ROOT/'已经整理好的内容/医院分割结果分析/重点failure失败原因/objects_EVAL_ONLY.csv')
    selected=[x for x in rows if x['dataset']=='hospital' and x['source_class']=='球性硬化']
    catalog={x['task']['task_id']:x['task'] for x in read(ROOT/'research/segformer_complementarity_20261106/catalog.json')}
    print('sclerosis',len(selected),'wsi',len({x['task_id'] for x in selected}),'groups',len({x['group_id'] for x in selected}))
    spec=importlib.util.spec_from_file_location('prepare',ROOT/'glomerulus_tools/experiments/problem_to_repair_20261111/prepare_cases.py')
    prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
    row=selected[0];t=catalog[row['task_id']]
    base=Path('D:/数据集/医院肾小球数据/2026_11_1_many/glomerulis_for_agents')/t['stain']
    for p in [base/'images'/Path(t['image']).name,ROOT/'research/full_networks_20261105/predictions/segformer_b5'/t['task_id']/'mask.tif']:
        print(str(p),'exists',p.exists(),flush=True)
        if p.exists():
            a=prep.roi(p,json.loads(row['box']));print('crop',a.shape,str(a.dtype),flush=True)
    result=remote_python('''from pathlib import Path
import importlib.util, json, subprocess
p=Path('/home/chenliangyu/project/new_agent')
print(json.dumps({'identity':[subprocess.check_output([x],text=True).strip() for x in ['hostname','whoami']], 'modules':{x:importlib.util.find_spec(x) is not None for x in ['detectron2','tensorflow','torch']},'dirs':[str(x) for x in p.iterdir() if any(k in x.name.lower() for k in ['mes','gncnn','skin','glo'])]},indent=2))
print(subprocess.check_output(['df','-h',str(p)],text=True))
''')
    print('remote exit',result.returncode,result.stdout,result.stderr)
