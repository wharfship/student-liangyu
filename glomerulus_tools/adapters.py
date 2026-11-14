"""Preserve the audited model graphs; add image-only arbitrary-task wrappers."""
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import subprocess

from .catalog import ROOT, sha


def module_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeRGB:
    def __init__(self, path):
        import numpy as np
        from PIL import Image
        self.slide = self.array = None
        try:
            import openslide
        except ImportError:
            openslide = None
        try:
            if openslide is None: raise ImportError('OpenSlide unavailable')
            self.slide = openslide.OpenSlide(str(path))
            self.width, self.height = self.slide.dimensions
        except Exception as exc:
            # Only TIFF memory mapping is a supported fallback, never an arbitrary full-WSI decode.
            if openslide is not None and not isinstance(exc,openslide.OpenSlideUnsupportedFormatError):raise
            if Path(path).suffix.lower() not in ('.tif', '.tiff'): raise exc
            import tifffile
            self.array = tifffile.memmap(path, page=0, mode='r')
            if self.array.ndim != 3 or self.array.dtype != np.uint8:
                raise ValueError('WSI TIFF fallback requires uint8 RGB page0')
            if self.array.shape[0] == 3:
                self.array = np.moveaxis(self.array, 0, -1)
            if self.array.shape[2] != 3: raise ValueError('WSI TIFF must have three channels')
            self.height, self.width = self.array.shape[:2]

    def read(self, x0, y0, x1, y1):
        import numpy as np
        from PIL import Image
        if self.slide is None: return np.asarray(self.array[y0:y1, x0:x1])
        rgba = self.slide.read_region((x0, y0), 0, (x1-x0, y1-y0))
        rgb = Image.new('RGB', rgba.size, 'white'); rgb.paste(rgba, mask=rgba.getchannel('A'))
        return np.asarray(rgb)

    def close(self):
        if self.slide is not None: self.slide.close()
        self.array = None


def wsi_nepath(network, model, card, task, output):
    import cv2
    import numpy as np
    import tifffile
    import torch
    width, height = task['native_size']; factor = task['source_mpp'] / card['target_mpp']
    ww, wh = max(1, round(width*factor)), max(1, round(height*factor))
    size, stride = card['patch'], card['stride']
    ys, xs = network.starts(wh, size, stride), network.starts(ww, size, stride)
    if ww*wh > 1_000_000_000 or len(ys)*len(xs) > 20000:
        raise ValueError('WSI exceeds fixed working-grid/window budget; create explicit spatial tasks')
    required_bytes = ww*wh*22 + width*height
    if shutil.disk_usage(output).free < required_bytes + 1024**3:
        raise ValueError('Insufficient disk for bounded-memory WSI working buffers and native mask')
    request=output/'wsi_preparation_task.json';request.write_text(json.dumps(task),encoding='utf-8')
    subprocess.run([card['io_python'],'-m','glomerulus_tools.prepare_wsi','--task',str(request),
                    '--output',str(output/'working_rgb.npy'),'--target-mpp',str(card['target_mpp'])],
                   cwd=ROOT.parent,check=True,timeout=1800)
    preparation=json.loads((output/'working_rgb.json').read_text())
    rgb=np.load(output/'working_rgb.npy',mmap_mode='r')
    if rgb.shape!=(wh,ww,3):raise ValueError('I/O cache dimensions differ from model grid')
    sums=np.lib.format.open_memmap(output/'probability_sums.npy',mode='w+',dtype=np.float32,shape=(4,wh,ww));sums[:]=0
    counts=np.lib.format.open_memmap(output/'coverage.npy',mode='w+',dtype=np.uint8,shape=(wh,ww));counts[:]=0
    with torch.inference_mode():
        for y in ys:
            for x in xs:
                crop=rgb[y:y+size,x:x+size];ch,cw=crop.shape[:2]
                tile=np.full((size,size,3),255,np.uint8);tile[:ch,:cw]=crop
                pred=network.probabilities(model,card['_model_card'],tile)
                sums[:,y:y+ch,x:x+cw]+=pred[:,:ch,:cw];counts[y:y+ch,x:x+cw]+=1
            (output/'progress.json').write_text(json.dumps(dict(row=y,total_rows=wh)),encoding='utf-8')
    minimum,maximum=int(counts.min()),int(counts.max())
    if minimum<1: raise ValueError('Incomplete WSI tile coverage')
    work=np.lib.format.open_memmap(output/'working_mask.npy',mode='w+',dtype=np.uint8,shape=(wh,ww))
    for y in range(0,wh,256):work[y:y+256]=(np.argmax(sums[:,y:y+256]/counts[None,y:y+256],axis=0)==1).astype(np.uint8)*255
    work.flush()
    # Uncompressed BigTIFF is deliberate: the existing Agent uses memory-mapped mask readers.
    mask=tifffile.memmap(output/'mask.tif',shape=(height,width),dtype=np.uint8,bigtiff=True,photometric='minisblack')
    xpos=np.minimum(np.arange(width,dtype=np.int64)*ww//width,ww-1)
    for y in range(0,height,128):
        ypos=np.minimum(np.arange(y,min(y+128,height),dtype=np.int64)*wh//height,wh-1)
        mask[y:y+128]=work[np.ix_(ypos,xpos)]
    mask.flush();mask._mmap.close()
    for arr in (rgb,sums,counts,work): arr._mmap.close()
    for name in ('working_rgb.npy','probability_sums.npy','coverage.npy','working_mask.npy'):
        (output/name).unlink()
    return dict(mask_path=str(output/'mask.tif'),native_size=[width,height],working_size=[ww,wh],
                coverage_min=minimum,coverage_max=maximum,windows=len(ys)*len(xs),io_preparation=preparation)


def infer(card, task, output):
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(4);torch.manual_seed(20261105);np.random.seed(20261105)
    torch.backends.cudnn.benchmark=False
    os.environ.update(card.get('environment',{}))
    for folder in reversed(card.get('extra_paths', [])):
        sys.path.insert(0,folder)
    if sha(card['checkpoint']) != card['checkpoint_sha256']:
        raise ValueError('Checkpoint SHA256 differs from registered tool identity')
    if card.get('config_sha256') and sha(card['config']) != card['config_sha256']:
        raise ValueError('Config SHA256 differs from registered tool identity')
    adapter=card['adapter'];start=time.monotonic()
    if adapter=='nepath':
        network=module_file(Path(card['source_dir'])/'run.py','_unified_nepath')
        model,model_card=network.load(card['model_id'])
        if model_card['sha256'] != card['checkpoint_sha256']: raise ValueError('Model card checkpoint mismatch')
        if (card['patch'],card['stride'],card['target_mpp']) != (model_card['tile_pixels'],model_card['stride_pixels'],1.):
            raise ValueError('NePath grid differs from frozen adapter protocol; register and validate a new recipe')
        if task['input_form']=='wsi':
            result=wsi_nepath(network,model,dict(card,_model_card=model_card),task,output)
        else:
            with Image.open(task['image']) as image:rgb=np.asarray(image.convert('RGB'))
            if rgb.shape[0]*rgb.shape[1]>100_000_000:raise ValueError('ROI exceeds 100M pixels; declare WSI')
            mask,labels,coverage=network.segment(model,model_card,rgb,task['source_mpp'],card['target_mpp'])
            Image.fromarray(mask).save(output/'mask.png');Image.fromarray(labels).save(output/'labels.png')
            result=dict(mask_path=str(output/'mask.png'),labels_path=str(output/'labels.png'),**coverage)
    elif adapter=='mescnn':
        from .mescnn import infer_tile
        result=infer_tile(card,task,output)
    elif adapter=='glo_matrix':
        source=module_file(card['matrix_adapter'],'_unified_glo_matrix')
        result=source.glo2(dict(task,source_mpp_for_crops=task['source_mpp']),card,output)
        result['source_mpp_measured']=task.get('scale_basis')=='measured_metadata'
        result['virtual_working_grid_calibrated_to_training']=False
    elif adapter=='qupath':
        source=module_file(Path(card['source_dir'])/'run_no_gt.py','_unified_qupath')
        model,loading=source.load_model('qupath_monai')
        if loading['checkpoint_sha256']!=card['checkpoint_sha256']:raise ValueError('QuPath checkpoint mismatch')
        with Image.open(task['image']) as image:rgb=np.asarray(image.convert('RGB'))
        if rgb.shape[0]*rgb.shape[1]>100_000_000:raise ValueError('ROI exceeds 100M pixels')
        mask,labels,coverage=source.segment(rgb,'qupath_monai',model)
        Image.fromarray((mask*255).astype(np.uint8)).save(output/'mask.png')
        result=dict(mask_path=str(output/'mask.png'),**coverage)
    elif adapter=='likiseg':
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        predictor=nnUNetPredictor(tile_step_size=.5,use_gaussian=True,use_mirroring=False,perform_everything_on_device=True,
                                 device=torch.device('cuda:0'),verbose=False,verbose_preprocessing=False,allow_tqdm=False)
        predictor.initialize_from_trained_model_folder(card['model_folder'],use_folds=(0,),checkpoint_name='checkpoint_final.pth')
        predictor.network.load_state_dict(predictor.list_of_parameters[0],strict=True)
        with Image.open(task['image']) as image:rgb=np.asarray(image.convert('RGB'))
        if rgb.shape[0]*rgb.shape[1]>100_000_000:raise ValueError('ROI exceeds 100M pixels')
        data=rgb.transpose(2,0,1)[:,None].astype(np.float32)
        label=predictor.predict_single_npy_array(data,{'spacing':[999.,1.,1.]},None,None,False)[0]
        if label.shape!=rgb.shape[:2] or not np.isin(label,[0,1,2]).all():raise ValueError('Invalid LiKiSeg labels')
        Image.fromarray(((label==2)*255).astype(np.uint8)).save(output/'mask.png')
        Image.fromarray(label.astype(np.uint8)).save(output/'labels.png')
        result=dict(mask_path=str(output/'mask.png'),labels_path=str(output/'labels.png'),native_size=task['native_size'],
                    working_size=task['native_size'],coverage_min=1,windows=len(predictor._internal_get_sliding_window_slicers(data.shape[1:])))
    else:raise ValueError('No new-model adapter: '+adapter)
    result.update(valid=True,status='completed',GT_read=False,gt_read=False,strict_load=True,
                  source_mpp=task['source_mpp'],scale_basis=task.get('scale_basis'),scale_evidence=task.get('scale_evidence'),
                  target_working_mpp=card.get('target_mpp'),checkpoint_sha256=card['checkpoint_sha256'],
                  config_sha256=card.get('config_sha256'),adapter_sha256=sha(__file__),
                  seconds=time.monotonic()-start,random_seed=20261105,torch=torch.__version__,torch_cuda=torch.version.cuda,
                  physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),needs_review=True,accuracy_verified=False)
    result['model_environment']=card.get('environment',{})
    return result


def finalize_record(card,task,output,result):
    """Verify the delivered native mask, never infer biological accuracy from validity."""
    import numpy as np
    from PIL import Image
    path=Path(result['mask_path'])
    if path.suffix.lower() in ('.tif','.tiff'):
        import tifffile
        mask=tifffile.memmap(path,mode='r')
    else:
        with Image.open(path) as image:mask=np.asarray(image)
    if mask.ndim!=2 or mask.dtype!=np.uint8:raise ValueError('Output mask must be 2D uint8')
    size=[mask.shape[1],mask.shape[0]]
    if task.get('native_size'):
        expected=list(task['native_size'])
    elif task.get('input_form','patch')=='wsi':
        from .contract import read_wsi_header
        expected=read_wsi_header(task['image'])['native_size']
    else:
        with Image.open(task['image']) as image:expected=list(image.size)
    if size!=expected:raise ValueError('Mask not in native input coordinates')
    count=0
    for y in range(0,mask.shape[0],512):
        block=mask[y:y+512]
        if not np.isin(block,[0,255]).all():raise ValueError('Output must be binary uint8 0/255')
        count+=int(np.count_nonzero(block))
    if hasattr(mask,'_mmap'):mask._mmap.close()
    from .catalog import load_registry
    result.update(schema_version=1,tool_id=card['id'],task_id=task['task_id'],native_size=size,
                  coordinate_frame='input_image_level0_pixels',mask_values=[0,255],foreground_pixels=count,
                  mask_sha256=sha(path),catalog_sha256=load_registry()['catalog_sha256'],
                  gt_read=False,accuracy_verified=False,needs_review=True,
                  existence_state='unknown',contour_state='unverified',
                  training_overlap=card.get('training_overlap_status','exact checkpoint membership unresolved'))
    if card.get('checkpoint'):result['checkpoint_sha256']=sha(card['checkpoint'])
    if card.get('config'):result['config_sha256']=sha(card['config'])
    result.update(source_mpp=task.get('source_mpp') or task.get('source_mpp_for_crops'),
                  scale_evidence=task.get('scale_evidence'),scale_basis=task.get('scale_basis'),
                  protocol_runner_sha256=sha(ROOT.parent/'ops/tool_protocol_runner.py'))
    return result
