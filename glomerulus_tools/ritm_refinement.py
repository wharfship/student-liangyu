"""RITM external-mask correction. This module never loads evaluation labels."""


def load_ritm(checkpoint, device='cuda:0', max_size=800):
    import torch
    from isegm.utils.serialization import load_model
    from isegm.inference.predictors.base import BasePredictor
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if isinstance(state, list):
        raise ValueError('This recipe requires a single registered RITM model')
    model = load_model(state['config'])
    model.load_state_dict(state['state_dict'], strict=True)
    if not model.with_prev_mask:
        raise ValueError('Checkpoint has no previous-mask input')
    model.to(device).eval().requires_grad_(False)
    predictor = BasePredictor(model, device=torch.device(device), max_size=max_size,
                              with_flip=True, zoom_in=None)
    return predictor, state['config']


def predict_ritm(predictor, image, coarse_mask, points):
    """points are (positive, y, x); callers must document their provenance."""
    import torch
    import numpy as np
    from isegm.inference.clicker import Clicker, Click
    predictor.set_input_image(image)
    clicker = Clicker()  # No GT is attached, even for externally supplied clicks.
    for positive, y, x in points:
        clicker.add_click(Click(is_positive=bool(positive), coords=(int(y), int(x))))
    previous = torch.as_tensor(np.asarray(coarse_mask, dtype=np.float32),
                               device=predictor.device)[None, None]
    with torch.inference_mode():
        probability = predictor.get_prediction(clicker, prev_mask=previous)
    if probability.shape != coarse_mask.shape:
        raise ValueError('RITM changed native geometry')
    return probability


def coarse_interior_point(coarse_mask):
    import cv2
    import numpy as np
    distance = cv2.distanceTransform(np.pad(coarse_mask.astype(np.uint8), 1),
                                     cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    y, x = np.unravel_index(distance.argmax(), distance.shape)
    return [(True, int(y), int(x))]
