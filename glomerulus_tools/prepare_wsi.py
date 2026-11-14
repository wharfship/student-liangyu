"""CPU image preparation in the existing I/O environment, separate from model dependencies."""
import argparse
import json
import math
from pathlib import Path
import time

from .catalog import sha
from .contract import prepare_task


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--target-mpp',type=float,required=True)
    args=parser.parse_args()
    import cv2
    import numpy as np
    from .adapters import NativeRGB
    task=prepare_task(json.loads(args.task.read_text(encoding='utf-8')))
    width,height=task['native_size'];factor=task['source_mpp']/args.target_mpp
    ww,wh=max(1,round(width*factor)),max(1,round(height*factor))
    cv2.setNumThreads(2);tick=time.monotonic()
    if args.output.exists():raise FileExistsError('Working RGB cache must be new')
    rgb=np.lib.format.open_memmap(args.output,mode='w+',dtype=np.uint8,shape=(wh,ww,3))
    reader=NativeRGB(task['image'])
    try:
        if [reader.width,reader.height]!=[width,height]:raise ValueError('WSI reader dimensions changed')
        for y in range(0,wh,256):
            end=min(y+256,wh);ypos=(np.arange(y,end,dtype=np.float64)+.5)*height/wh-.5
            y0=max(0,math.floor(ypos.min()));y1=min(height,math.ceil(ypos.max())+2)
            for x in range(0,ww,2048):
                stop=min(x+2048,ww);xpos=(np.arange(x,stop,dtype=np.float64)+.5)*width/ww-.5
                x0=max(0,math.floor(xpos.min()));x1=min(width,math.ceil(xpos.max())+2)
                mx,my=np.meshgrid((xpos-x0).astype(np.float32),(ypos-y0).astype(np.float32))
                rgb[y:end,x:stop]=cv2.remap(reader.read(x0,y0,x1,y1),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    finally:reader.close()
    rgb.flush();rgb._mmap.close()
    receipt=dict(GT_read=False,native_size=[width,height],working_size=[ww,wh],source_mpp=task['source_mpp'],
                 scale_evidence=task['scale_evidence'],target_working_mpp=args.target_mpp,code_sha256=sha(__file__),
                 cache_sha256=sha(args.output),seconds=time.monotonic()-tick)
    args.output.with_suffix('.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')


if __name__=='__main__':main()
