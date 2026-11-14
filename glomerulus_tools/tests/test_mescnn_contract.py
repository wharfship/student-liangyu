"""Prevent GT metadata and arbitrary unscaled crops entering the specialist batch."""
import copy
import pytest
from PIL import Image
from glomerulus_tools.mescnn import validate_batch

@pytest.fixture
def batch(tmp_path):
    image=tmp_path/'tile.png';Image.new('RGB',(32,32),'white').save(image)
    return dict(schema='mescnn_prepared_tile_batch_v1',cases=[dict(task_id='A001',image=str(image),species='human',
       stain='PAS',target='whole_glomerular_unit',input_form='patch',native_size=[32,32],source_mpp=1.,
       scale_basis='mescnn_metadata_10x_prepared_tile',scale_evidence='SVS metadata test')])

def test_accepts_prepared_image_only_batch(batch):
    validate_batch(batch)

@pytest.mark.parametrize('field',['GT','gt_id','source_class','delta_dice','error_label'])
def test_rejects_offline_decision_information(batch,field):
    batch['cases'][0][field]='offline'
    with pytest.raises(ValueError,match='Unknown/offline'):validate_batch(batch)

def test_rejects_arbitrary_native_crop(batch):
    batch['cases'][0]['scale_basis']='SVS_metadata'
    with pytest.raises(ValueError,match='prepared tile'):validate_batch(batch)

def test_rejects_duplicate_task_ids(batch):
    batch['cases'].append(copy.deepcopy(batch['cases'][0]))
    with pytest.raises(ValueError,match='Duplicate'):validate_batch(batch)
