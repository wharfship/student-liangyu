from pathlib import Path
import os,sys,time,json,subprocess,hashlib
P=Path(__file__).resolve().parent
def main():
 resource=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 choices=[tuple(map(int,l.split(','))) for l in resource.splitlines()];choices=[c for c in choices if c[1]>18000 and c[2]<10];assert choices
 gpu=max(choices,key=lambda x:x[1])[0];os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),TF_CPP_MIN_LOG_LEVEL='2')
 import tensorflow as tf
 assert tf.config.list_physical_devices('GPU'),'No CPU fallback'
 for g in tf.config.list_physical_devices('GPU'):tf.config.experimental.set_memory_growth(g,True)
 sys.path.insert(0,str(P/'assets/Skinet-MEST-C-main'))
 from mrcnn.TensorflowDetector import TensorflowDetector
 import numpy as np
 from PIL import Image
 image=np.asarray(Image.open(P.parent/'sclerosis_focus_20261114/specialist_online/S019/image.png').convert('RGB'))
 records=[]
 for name in ['skinet_cortex','skinet_mest_main']:
  tick=time.monotonic();model=TensorflowDetector(str(P/'assets'/name))
  pred=model.process(image,normalizedCoordinates=False,score_threshold=.5)
  record=dict(model=name,classes=pred['class_ids'].tolist(),scores=pred['scores'].tolist(),rois=pred['rois'].tolist(),
    mask_shape=list(pred['masks'].shape),seconds=time.monotonic()-tick)
  records.append(record);print(json.dumps(record),flush=True)
  del model
 save=dict(status='complete',gpu=gpu,resource=resource,tf=tf.__version__,records=records,
    source='SkinetTeam/Skinet-MEST-C main archive SHA recorded',scale='unvalidated nominal10x patch, not full author workflow')
 (P/'skinet_smoke_complete.json').write_text(json.dumps(save,indent=2))

if __name__=='__main__':main()
