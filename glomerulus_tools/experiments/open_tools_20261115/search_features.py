"""Pure cosine retrieval on recorded embeddings. No pathology labels are read."""
import argparse,json
from pathlib import Path
import numpy as np

def search(query,index,top_k=5):
 q=np.load(query);q=q.reshape(-1).astype(float)
 if q.shape!=(384,) or not np.isfinite(q).all() or np.linalg.norm(q)==0:raise ValueError('Require finite nonzero384-vector')
 q/=np.linalg.norm(q);ranked=[]
 for p in sorted(Path(index).glob('*/embedding_l2.npy')):
  if p.resolve()==Path(query).resolve():continue
  v=np.load(p).reshape(-1)
  if v.shape!=(384,) or not np.isfinite(v).all():raise ValueError('Invalid index feature')
  ranked.append(dict(sample_id=p.parent.name,cosine=float(q@v/np.linalg.norm(v)),embedding=str(p)))
 return sorted(ranked,key=lambda r:-r['cosine'])[:top_k]

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--query',type=Path,required=True);p.add_argument('--index',type=Path,required=True);p.add_argument('--top-k',type=int,default=5);p.add_argument('--output',type=Path)
 a=p.parse_args()
 if not 1<=a.top_k<=100:p.error('top-k must be1..100')
 result=dict(neighbors=search(a.query,a.index,a.top_k),labels_read=False,not_a_disease_classifier=True,
  note='Raw index search excludes only same file; caller must separately constrain source/stain if needed. Scores are similarities, not calibrated probabilities.')
 text=json.dumps(result,ensure_ascii=False,indent=2)
 if a.output:a.output.write_text(text,encoding='utf8')
 print(text)

if __name__=='__main__':main()
