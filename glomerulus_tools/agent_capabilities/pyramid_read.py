"""Read the same native-coordinate ROI from an explicit SVS JPEG pyramid page.

Read-only building block, no automatic level choice or inference. Returned RGB
keeps that page's resolution. Supports the audited tiled RGB uint8/JPEG SVS
layout; other formats fail explicitly rather than silently using page zero.
"""
from pathlib import Path
import io, math
import numpy as np
from PIL import Image
import tifffile


def read_page_roi(path, native_box, page_index):
    if not isinstance(page_index, int) or page_index < 0:
        raise ValueError('Expected explicit nonnegative TIFF page index')
    with tifffile.TiffFile(Path(path)) as f:
        base=f.pages[0]
        nw,nh=base.imagewidth,base.imagelength
        x,y,xx,yy=native_box
        if not (0 <= x < xx <= nw and 0 <= y < yy <= nh):
            raise ValueError('ROI must lie inside native page zero coordinates')
        p=f.pages[page_index];w,h=p.imagewidth,p.imagelength
        if not (p.is_tiled and p.planarconfig==1 and p.samplesperpixel==3
                and p.dtype==np.dtype('uint8') and int(p.compression)==7):
            raise ValueError('Only audited tiled uint8 RGB JPEG pages are supported')
        if int(p.tags['Orientation'].value if 'Orientation' in p.tags else 1)!=1:
            raise ValueError('Nonstandard TIFF orientation needs a separate mapping')
        if 'AppMag' not in p.description:
            raise ValueError('Associated thumbnail/label is not a spatial pyramid page')
        l,t=max(0,math.floor(x*w/nw)),max(0,math.floor(y*h/nh))
        r,b=min(w,math.ceil(xx*w/nw)),min(h,math.ceil(yy*h/nh))
        if (r-l)*(b-t)>16_000_000:
            raise ValueError('Local page ROI exceeds 16M pixels')
        rgb=np.zeros((b-t,r-l,3),np.uint8);tw,th=p.tilewidth,p.tilelength
        nx=math.ceil(w/tw);tiles=[]
        for ty in range(t//th,(b-1)//th+1):
            for tx in range(l//tw,(r-1)//tw+1):
                idx=ty*nx+tx
                f.filehandle.seek(p.dataoffsets[idx]);payload=f.filehandle.read(p.databytecounts[idx])
                tables=p.jpegtables
                jpeg=tables[:-2]+payload[2:] if tables else payload
                with Image.open(io.BytesIO(jpeg)) as im:
                    tile=np.asarray(im.convert('RGB'))
                ll,tt,rr,bb=max(l,tx*tw),max(t,ty*th),min(r,(tx+1)*tw),min(b,(ty+1)*th)
                rgb[tt-t:bb-t,ll-l:rr-l]=tile[tt-ty*th:bb-ty*th,ll-tx*tw:rr-tx*tw]
                tiles.append(dict(tile_xy=[tx,ty],compressed_bytes=len(payload),pure_white=bool(np.all(tile==255))))
        return dict(rgb=rgb,page_index=page_index,native_box=list(native_box),
                    page_box=[l,t,r,b],page_size=[w,h],native_size=[nw,nh],
                    downsample_xy=[nw/w,nh/h],tiles=tiles,
                    description=p.description,original_modified=False)


def read_working_roi(path, page_index, working_size, working_box):
    """Resample an explicit page onto the ORIGINAL whole-slide working grid.

    No GT or automatic quality decision. Preserve both whole-grid dimensions
    and global ROI origin; do not resize a tight crop and shift its coordinates.
    """
    ww,wh=working_size;l,t,r,b=working_box
    if not (0 <= l < r <= ww and 0 <= t < b <= wh) or (r-l)*(b-t)>1_000_000:
        raise ValueError('Expected bounded ROI inside original working grid')
    with tifffile.TiffFile(path) as f:
        nw,nh=f.pages[0].imagewidth,f.pages[0].imagelength
        pw,ph=f.pages[page_index].imagewidth,f.pages[page_index].imagelength
    # Include one page pixel of interpolation context on either side.
    native_box=[max(0,math.floor((l+.5)*nw/ww-.5)-math.ceil(nw/pw)),
                max(0,math.floor((t+.5)*nh/wh-.5)-math.ceil(nh/ph)),
                min(nw,math.ceil((r-.5)*nw/ww-.5)+math.ceil(nw/pw)+1),
                min(nh,math.ceil((b-.5)*nh/wh-.5)+math.ceil(nh/ph)+1)]
    rec=read_page_roi(path,native_box,page_index)
    rgb=rec.pop('rgb');px,py,_,_=rec['page_box']
    # Map pixel centres via native edge coordinates; ratio also handles rounding
    # of odd native/page dimensions in the audited SVS.
    xs=(np.arange(l,r,dtype=np.float64)+.5)*pw/ww-.5-px
    ys=(np.arange(t,b,dtype=np.float64)+.5)*ph/wh-.5-py
    import cv2
    mx,my=np.meshgrid(xs.astype(np.float32),ys.astype(np.float32))
    result=cv2.remap(rgb,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    rec.update(rgb=result,working_size=list(working_size),working_box=list(working_box),
               output_role='alternative_input_only',GT_read=False)
    return rec
