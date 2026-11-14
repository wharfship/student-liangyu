"""Deterministic, read-only views of ONE predicted candidate; no annotation input."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .catalog import sha


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


FIELDS = {'candidate_id', 'image', 'image_sha256', 'native_size', 'stain', 'source_mpp',
          'scale_basis', 'box', 'support_npz', 'support_sha256'}
VIEWS = ('context', 'native_center', 'native_NW', 'native_NE', 'native_SW', 'native_SE')


class CandidateViews:
    def __init__(self, task, output):
        if set(task) != FIELDS:
            raise ValueError('Candidate fields must exactly match the no-GT contract')
        if task['stain'] not in ('HE', 'PAS'):
            raise ValueError('This research reader currently supports HE/PAS only')
        self.task = task
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        for key, digest in [('image', 'image_sha256'), ('support_npz', 'support_sha256')]:
            if sha(task[key]) != task[digest]:
                raise ValueError('Source hash mismatch: '+key)
        with Image.open(task['image']) as im:
            self.image = im.convert('RGB')
        if list(self.image.size) != task['native_size']:
            raise ValueError('Native dimensions mismatch')
        x, y, xx, yy = task['box']
        w, h = self.image.size
        if not all(type(v) is int for v in (x, y, xx, yy)) or not (0 <= x < xx <= w and 0 <= y < yy <= h):
            raise ValueError('Invalid candidate coordinates')
        with np.load(task['support_npz'], allow_pickle=False) as z:
            if set(z.files) != {'a', 'b'}:
                raise ValueError('Only two predicted masks are accepted')
            if any(z[k].dtype != np.bool_ or z[k].shape != (yy-y, xx-x) for k in ('a', 'b')):
                raise ValueError('Support shape/dtype mismatch')
            self.support = z['a'] | z['b']
        if not self.support.any():
            raise ValueError('Empty candidate support')
        self.receipts = {}

    def render(self, name):
        if name not in ('initial',) + VIEWS:
            raise ValueError('Unknown view')
        if name in self.receipts:
            raise ValueError('Repeated view request')
        x, y, xx, yy = self.task['box']
        cx, cy = (x+xx)//2, (y+yy)//2
        w, h = self.image.size
        if name in ('initial', 'context'):
            pad = max(32, round(max(xx-x, yy-y)*(.2 if name == 'initial' else 1.0)))
            box = [max(0, x-pad), max(0, y-pad), min(w, xx+pad), min(h, yy+pad)]
            limit = 640 if name == 'initial' else 1024
        else:
            if name != 'native_center':
                cy = y if name.endswith(('NW', 'NE')) else yy
                cx = x if name.endswith(('NW', 'SW')) else xx
            bx, by = max(0, min(max(0, w-768), cx-384)), max(0, min(max(0, h-768), cy-384))
            box = [bx, by, min(w, bx+768), min(h, by+768)]
            limit = 768
        bx, by, ex, ey = box
        raw = self.image.crop(box)
        support = Image.new('L', raw.size)
        ix, iy, ixx, iyy = max(x, bx), max(y, by), min(xx, ex), min(yy, ey)
        if ix < ixx and iy < iyy:
            patch = Image.fromarray(self.support[iy-y:iyy-y, ix-x:ixx-x].astype('uint8')*255)
            support.paste(patch, (ix-bx, iy-by))
        native = list(raw.size)
        raw.thumbnail((limit, limit), Image.Resampling.LANCZOS)
        support = support.resize(raw.size, Image.Resampling.NEAREST)
        edge = np.asarray(support) > np.asarray(support.filter(ImageFilter.MinFilter(3)))
        overlay = np.array(raw).copy()
        overlay[edge] = [255, 230, 0]
        paths = [self.output/(name+'_raw.png'), self.output/(name+'_support.png')]
        raw.save(paths[0]); Image.fromarray(overlay).save(paths[1])
        receipt = dict(name=name, source_box=box, native_size=native, returned_size=list(raw.size),
                       candidate_box=self.task['box'], source_sha256=self.task['image_sha256'],
                       images=[str(p.resolve()) for p in paths], sha256=[sha(p) for p in paths],
                       scale=[raw.width/native[0], raw.height/native[1]], gt_read=False,
                       target_clipped=not (bx <= x and by <= y and ex >= xx and ey >= yy))
        self.receipts[name] = receipt
        dump(self.output/'view_receipts_NO_GT.json', self.receipts)
        return paths, receipt
