"""Image-only local grid recheck. Returns a candidate; never accepts/replaces it.

Coordinates are xyxy in the existing RGB working grid, right/bottom exclusive.
The caller supplies the original registered model's probability function and
inference context. No GT, model loading, SSH or file mutation happens here.
"""
import numpy as np


def plan_windows(image_size, box, tile_size=1024, stride=512, phase=256, max_tiles=16):
    """Match the C004 diagnostic grids; reject incomplete edge coverage up front."""
    w, h = image_size
    l, t, r, b = box
    values = [w, h, l, t, r, b, tile_size, stride, phase, max_tiles]
    if any(not isinstance(v, (int, np.integer)) for v in values):
        raise ValueError('Coordinates and grid settings must be integers')
    if not (0 <= l < r <= w and 0 <= t < b <= h):
        raise ValueError('ROI must be inside the working image')
    if not (0 < stride <= tile_size and 0 <= phase < stride and max_tiles > 0):
        raise ValueError('Invalid grid or tile budget')
    if (r-l)*(b-t) > 1_000_000:
        raise ValueError('Local diagnostic ROI exceeds 1M working pixels')

    def starts(length):
        if phase == 0:
            return [0] if length <= tile_size else sorted(set(range(0, length-tile_size+1, stride)) | {length-tile_size})
        return range(phase, length-tile_size+1, stride)

    xs = [x for x in starts(w) if x < r and x+tile_size > l]
    ys = [y for y in starts(h) if y < b and y+tile_size > t]
    windows = [(x, y, min(w, x+tile_size), min(h, y+tile_size)) for y in ys for x in xs]
    if len(windows) > max_tiles:
        raise ValueError('Tile budget exceeded; no inference performed')
    if not xs or not ys or min(xs) > l or min(ys) > t or max(xs)+tile_size < r or max(ys)+tile_size < b:
        raise ValueError('Shifted grid cannot fully cover this ROI; preserve original result')
    return windows


def fuse_roi(box, clipped_predictions):
    """Average (global overlap box, 4xHxW probabilities) in original tile order."""
    l, t, r, b = box
    sums = np.zeros((4, b-t, r-l), np.float32)
    counts = np.zeros((b-t, r-l), np.uint16)
    for overlap, probs in clipped_predictions:
        x, y, xx, yy = overlap
        if not (l <= x < xx <= r and t <= y < yy <= b):
            raise ValueError('Tile overlap must be inside ROI')
        probs = np.asarray(probs)
        if probs.shape != (4, yy-y, xx-x) or not np.isfinite(probs).all():
            raise ValueError('Expected finite four-class probabilities')
        if probs.min() < 0 or probs.max() > 1 or not np.allclose(probs.sum(0), 1, atol=1e-4):
            raise ValueError('Expected normalized class probabilities')
        sums[:, y-t:yy-t, x-l:xx-l] += probs
        counts[y-t:yy-t, x-l:xx-l] += 1
    if not counts.all():
        raise ValueError('ROI has uncovered pixels; preserve original result')
    probabilities = sums / counts[None]
    labels = probabilities.argmax(0).astype(np.uint8)
    return dict(probabilities=probabilities, labels=labels, mask=labels == 1, counts=counts)


def recheck_roi(working_rgb, box, predict_probabilities, phase=256, tile_size=1024, stride=512, max_tiles=16):
    """Generate one alternative ROI result; model preprocessing stays with caller.

    predict_probabilities(tile_rgb) must return 4xT xT softmax probabilities.
    Use torch.inference_mode() around this call for the original torch adapter.
    This function neither chooses a suspicious ROI nor judges candidate quality.
    """
    if working_rgb.ndim != 3 or working_rgb.shape[2] != 3 or working_rgb.dtype != np.uint8:
        raise ValueError('Expected original uint8 RGB working image')
    h, w = working_rgb.shape[:2]
    windows = plan_windows((w, h), box, tile_size, stride, phase, max_tiles)
    l, t, r, b = box

    def predictions():
        for x, y, xx, yy in windows:
            tile = np.full((tile_size, tile_size, 3), 255, np.uint8)
            tile[:yy-y, :xx-x] = working_rgb[y:yy, x:xx]
            probs = np.asarray(predict_probabilities(tile))
            if probs.shape != (4, tile_size, tile_size):
                raise ValueError('Model probability shape does not match configured tile')
            ll, tt, rr, bb = max(l, x), max(t, y), min(r, xx), min(b, yy)
            yield (ll, tt, rr, bb), probs[:, tt-y:bb-y, ll-x:rr-x]

    result = fuse_roi(box, predictions())
    result.update(box_working=list(box), phase=phase, windows=windows,
                  GT_read=False, acceptance='candidate_only', original_modified=False)
    return result
