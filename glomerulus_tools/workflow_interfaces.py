"""Researcher-only descriptions of planned pathology operations.

The sole editable source is catalog.csv execution_json. A schema being valid
does not mean that a runtime adapter, image accuracy or Agent access is ready.
"""
import math


def describe(card):
    operations = card.get('planned_operations', [])
    if not operations:
        raise ValueError('No per-function protocol registered for this tool')
    return dict(tool_id=card['id'], status=card['status'],
                enabled=card['enabled'], agent_enabled=card['agent_enabled'],
                interface_status='planned_not_executable',
                source=card['open_source'],
                operations=[dict(op, function=dict(
                    name=card['id']+'__'+op['operation'],
                    description=op['description'], parameters=op['parameters']))
                    for op in operations])


def _validate(value, schema, label):
    kind = schema['type']
    if kind == 'object':
        if not isinstance(value, dict):
            raise ValueError(label+': expected object')
        if set(value)-set(schema['properties']):
            raise ValueError(label+': unknown fields; no GT/offline labels accepted')
        if set(schema['required'])-set(value):
            raise ValueError(label+': missing required fields')
        for key, item in value.items():
            _validate(item, schema['properties'][key], label+'.'+key)
        return
    if kind == 'string':
        valid = isinstance(value, str) and bool(value.strip())
    elif kind == 'integer':
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif kind == 'number':
        valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    elif kind == 'boolean':
        valid = isinstance(value, bool)
    else:
        raise ValueError('Unsupported planned schema type: '+kind)
    if not valid:
        raise ValueError(label+': invalid '+kind)
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError(label+': unsupported value')
    if 'minimum' in schema and value < schema['minimum']:
        raise ValueError(label+': below minimum')
    if 'exclusiveMinimum' in schema and value <= schema['exclusiveMinimum']:
        raise ValueError(label+': must be positive')
    if 'maximum' in schema and value > schema['maximum']:
        raise ValueError(label+': above maximum')


def check_request(task, card):
    if not isinstance(task, dict) or set(task) != {'schema_version', 'operation', 'arguments'}:
        raise ValueError('Require only schema_version, operation and arguments')
    if type(task['schema_version']) is not int or task['schema_version'] != 1:
        raise ValueError('Require schema_version=1')
    operation = next((op for op in card.get('planned_operations', [])
                      if op['operation'] == task['operation']), None)
    if operation is None:
        raise ValueError('Unknown operation')
    _validate(task['arguments'], operation['parameters'], 'arguments')
    return dict(tool_id=card['id'], operation=task['operation'],
                contract_valid=True, executable=False, agent_enabled=False,
                runtime_validated=False, accuracy_validated=False,
                reason='Selected pending tool; schema check only, no image execution or file-access check',
                source_api=operation['source_api'], expected_outputs=operation['expected_outputs'])
