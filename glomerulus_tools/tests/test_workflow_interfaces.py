"""Validate planned requests without letting pending tools become executable."""
import copy
import pytest
from glomerulus_tools.catalog import get_card
from glomerulus_tools.workflow_interfaces import describe, check_request


def request(card, operation):
    op = next(x for x in card['planned_operations'] if x['operation']==operation)
    values = {}
    for name, field in op['parameters']['properties'].items():
        values[name] = field['enum'][0] if 'enum' in field else (
            'example' if field['type']=='string' else 1)
    return dict(schema_version=1, operation=operation, arguments=values)


@pytest.mark.parametrize('tool_id', ['histoqc_wsi_quality_candidate','tiatoolbox_stain_workflow_candidate'])
def test_pending_describe_is_not_a_runtime_pass(tool_id):
    card=get_card(tool_id);desc=describe(card)
    assert not desc['enabled'] and not desc['agent_enabled']
    assert desc['interface_status']=='planned_not_executable'
    assert len(desc['operations'])==4
    for op in card['planned_operations']:
        result=check_request(request(card,op['operation']),card)
        assert result['contract_valid'] and not result['executable']
        assert not result['runtime_validated'] and not result['accuracy_validated']


@pytest.mark.parametrize('change', ['unknown','GT','missing_reference','bad_method','bool_mpp','nan_mpp'])
def test_reject_ambiguous_or_offline_inputs(change):
    card=get_card('tiatoolbox_stain_workflow_candidate')
    task=request(card,'normalize_stain')
    if change=='unknown':task['operation']='run_any_python'
    elif change=='GT':task['arguments']['GT']='truth.png'
    elif change=='missing_reference':del task['arguments']['reference_image']
    elif change=='bad_method':task['arguments']['method']='unregistered'
    elif change=='bool_mpp':task['arguments']['source_mpp']=True
    elif change=='nan_mpp':task['arguments']['source_mpp']=float('nan')
    with pytest.raises(ValueError):check_request(task,card)
