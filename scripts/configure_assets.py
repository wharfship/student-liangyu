"""Change deployment paths in the sole catalog; preserve tool/weight identities."""
import argparse,csv,json,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ALLOWED={'python','io_python','source_dir','checkpoint','config','extra_paths','required_assets','source_receipt','matrix_adapter'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--overrides',type=Path,required=True);a=p.parse_args()
    overrides=json.loads(a.overrides.read_text(encoding='utf-8-sig'))
    path=ROOT/'glomerulus_tools/catalog.csv'
    rows=list(csv.DictReader(path.open(encoding='utf-8-sig',newline='')));byid={r['tool_id']:r for r in rows}
    for key,changes in overrides.items():
        if key not in byid or not isinstance(changes,dict) or set(changes)-ALLOWED:raise ValueError('Unknown tool or non-path change')
        for field,value in changes.items():
            values=value if field in {'extra_paths','required_assets'} else [value]
            if not isinstance(values,list) or not all(isinstance(v,str) and Path(v).is_absolute() for v in values):raise ValueError('Require absolute path(s): '+field)
        execution=json.loads(byid[key]['execution_json']);execution.update(changes)
        byid[key]['execution_json']=json.dumps(execution,ensure_ascii=False)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    subprocess.run([sys.executable,'-m','glomerulus_tools.build'],cwd=ROOT,check=True)

if __name__=='__main__':main()
