"""Utility input contract protects offline labels and physical-size assumptions."""
import copy
import pytest
from glomerulus_tools.pathology_utilities import validate_batch

def sample():
 from pathlib import Path
 return dict(schema_version=1,samples=[dict(sample_id='A01',image=str(Path(__file__).resolve()),image_sha256='a'*64,source_mpp=.25,scale_evidence='measured header',stain='HE')])

def test_valid_utility_batch():
 task=sample();assert validate_batch(task,{'stains':['HE']}) is task

@pytest.mark.parametrize('mutation',[
 lambda t:t.update(GT='forbidden'),
 lambda t:t['samples'][0].update(target_dice=1),
 lambda t:t['samples'][0].update(source_mpp=0),
 lambda t:t['samples'][0].update(source_mpp=float('nan')),
 lambda t:t['samples'][0].update(scale_evidence=''),
 lambda t:t['samples'][0].update(sample_id='../escape'),
 lambda t:t['samples'][0].update(stain='PAS'),
 lambda t:t['samples'][0].update(image_sha256='unverified'),
 lambda t:t['samples'].append(copy.deepcopy(t['samples'][0])),
])
def test_reject_unsafe_or_offline_utility_inputs(mutation):
 task=sample();mutation(task)
 with pytest.raises(ValueError):validate_batch(task,{'stains':['HE']})
