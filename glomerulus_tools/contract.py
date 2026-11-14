"""Image-only input validation, metadata scale evidence, and compatibility checks."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET

FIELDS = {
    'task_id', 'dataset_id', 'species', 'stain', 'target', 'image', 'source_image',
    'native_box', 'downsample', 'per_tool_downsample', 'source_mpp', 'scale_evidence',
    'scope', 'source_mpp_for_crops', 'evaluation_shape', 'cohort', 'already_scaled_patch',
    'author_source_downsample', 'source_mpp_status', 'selection', 'input_form',
    'native_size', 'scale_basis',
}
ALIASES = {'H&E': 'HE', 'HЕ': 'HE', 'PAM': 'SIL', 'SILVER': 'SIL', 'MASSON': 'TRI'}


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def validate_task(task):
    unknown = set(task) - FIELDS
    if unknown:
        raise ValueError('Unknown/offline fields: ' + ','.join(sorted(unknown)))
    for key in ('task_id', 'image', 'species', 'stain', 'target'):
        if not isinstance(task.get(key), str) or not task[key].strip():
            raise ValueError('Missing online field: ' + key)
    result = dict(task)
    stain = result['stain'].strip().upper()
    result['stain'] = ALIASES.get(stain, stain)
    form = result.get('input_form', 'patch').lower()
    result['input_form'] = 'patch' if form == 'roi' else form
    if result['input_form'] not in ('patch', 'wsi'):
        raise ValueError('input_form must be patch/ROI or wsi')
    for key in ('source_mpp', 'source_mpp_for_crops'):
        if result.get(key) is not None and not positive(result[key]):
            raise ValueError('Invalid MPP: ' + key)
    if result.get('source_mpp') and result.get('source_mpp_for_crops'):
        if abs(result['source_mpp'] / result['source_mpp_for_crops'] - 1) > .05:
            raise ValueError('Conflicting source MPP fields')
    if not result.get('source_mpp') and result.get('source_mpp_for_crops'):
        result['source_mpp'] = result['source_mpp_for_crops']
    size = result.get('native_size')
    if size is not None and (not isinstance(size, (list, tuple)) or len(size) != 2 or
                             any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in size)):
        raise ValueError('native_size must be positive integer [width,height]')
    return result


def read_wsi_header(path):
    """Read metadata only; OME TIFFs need not be OpenSlide-readable."""
    try:
        import openslide
    except ImportError:
        openslide = None
    if openslide is not None:
        try:
            with openslide.OpenSlide(str(path)) as slide:
                props = slide.properties
                size = list(slide.dimensions)
                x = float(props.get('openslide.mpp-x', 0))
                y = float(props.get('openslide.mpp-y', 0))
            return dict(native_size=size, x=x, y=y, evidence='OpenSlide mpp-x/y metadata')
        except openslide.OpenSlideUnsupportedFormatError:
            pass
    import tifffile
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        size = [page.imagewidth, page.imagelength]
        xml = tif.ome_metadata
    x = y = 0.
    if xml:
        pixels = next((e for e in ET.fromstring(xml).iter() if e.tag.rsplit('}', 1)[-1] == 'Pixels'), None)
        if pixels is not None:
            if [int(pixels.get('SizeX')), int(pixels.get('SizeY'))] != size:
                raise ValueError('OME metadata dimensions differ from level0 image')
            units = {'µm': 1., 'μm': 1., 'um': 1., 'micrometer': 1., 'nm': .001, 'mm': 1000.}
            ux, uy = pixels.get('PhysicalSizeXUnit'), pixels.get('PhysicalSizeYUnit')
            if ux in units and uy in units:
                x = float(pixels.get('PhysicalSizeX', 0)) * units[ux]
                y = float(pixels.get('PhysicalSizeY', 0)) * units[uy]
    return dict(native_size=size, x=x, y=y, evidence='OME PhysicalSizeX/Y metadata with explicit units')


def prepare_task(task, require_scale=True):
    t = validate_task(task)
    if t['input_form'] == 'wsi':
        info = read_wsi_header(t['image'])
        x, y = info['x'], info['y']
        if positive(x) and positive(y) and abs(x-y)/x <= .02:
            if t.get('source_mpp') and abs(t['source_mpp']-x)/x > .05:
                raise ValueError('Task MPP conflicts with WSI header')
            t.update(source_mpp=x, source_mpp_status='measured_wsi_header',
                     scale_basis='measured_metadata', scale_evidence=info['evidence'])
        elif positive(x) or positive(y):
            raise ValueError('Anisotropic or incomplete WSI MPP; resample with a separately verified protocol')
        elif not t.get('source_mpp'):
            t.update(source_mpp=None, source_mpp_status='unresolved_wsi_header_mpp')
        size = info['native_size']
    else:
        from PIL import Image
        with Image.open(t['image']) as image:
            size = list(image.size)
    if t.get('native_size') and list(t['native_size']) != size:
        raise ValueError('native_size differs from actual input image')
    t['native_size'] = size
    if require_scale and (not positive(t.get('source_mpp')) or not (t.get('scale_evidence') or t.get('scale_basis'))):
        raise ValueError('Need source_mpp and scale_evidence/scale_basis; do not infer MPP from magnification')
    return t


def compatibility(card, task, research=False):
    reasons = []
    if not card.get('enabled', True): reasons.append('tool_disabled')
    if task.get('species') not in card.get('species', []): reasons.append('unsupported_species')
    if task.get('stain') not in card.get('stains', []): reasons.append('unsupported_stain')
    if task.get('input_form', 'patch') not in card.get('input_forms', ['patch']): reasons.append('unsupported_input_form')
    if task.get('target') != card.get('target'): reasons.append('target_mismatch')
    if not research:
        if not card.get('agent_enabled', True): reasons.append('research_only')
        if task.get('stain') not in card.get('agent_stains', card.get('stains', [])): reasons.append('exploratory_stain')
    mpp = task.get('source_mpp') or task.get('source_mpp_for_crops')
    if card.get('requires_mpp'):
        if not positive(mpp): reasons.append('unresolved_mpp')
        if not (task.get('scale_evidence') or task.get('scale_basis') or task.get('source_mpp_status')):
            reasons.append('missing_scale_evidence')
    if card.get('required_native_mpp') and (not positive(mpp) or abs(mpp/card['required_native_mpp']-1) > .001):
        reasons.append('unverified_native_grid')
    if card.get('allowed_dataset_ids') and task.get('dataset_id') not in card['allowed_dataset_ids']:
        reasons.append('unverified_cohort_grid')
    if card.get('id') == 'glo2_whole':
        if not positive(mpp) or not (.1 < mpp < .2 or .2 < mpp < .4 or .4 < mpp < .6):
            reasons.append('unsupported_glo_crop_mpp')
    if card.get('adapter')=='glo_matrix' and task.get('input_form','patch')=='patch':
        if not positive(mpp) or not (.1<mpp<.2 or .2<mpp<.4 or .4<mpp<.6):
            reasons.append('unsupported_glo_patch_crop_mpp')
    if task.get('stain') not in card.get('species_stains', {}).get(task.get('species'), card.get('stains', [])):
        reasons.append('unsupported_species_stain_pair')
    return reasons


def bind_tool_scale(card,task):
    """Fill legacy execution parameters from physical scale, without changing their graphs."""
    t=dict(task)
    mpp=t.get('source_mpp') or t.get('source_mpp_for_crops')
    if positive(mpp):
        t.setdefault('source_mpp_for_crops',mpp)
        key='omni_cap' if card['id']=='omni_full' else ('ki2021' if card['id'].startswith('ki2021_') else None)
        if key:
            expected=card['target_mpp']/mpp
            scales=dict(t.get('per_tool_downsample',{}))
            supplied=scales.get(key)
            if supplied is not None and (not positive(supplied) or abs(supplied/expected-1)>.05):
                raise ValueError('Legacy downsample conflicts with MPP for '+card['id'])
            scales.setdefault(key,expected)
            t['per_tool_downsample']=scales
    return t
