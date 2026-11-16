"""Check the frozen export's bytes and key catalog states, using stdlib only."""
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root=ROOT):
    manifest = json.loads((root / 'provenance/export_manifest.json').read_text(encoding='utf-8-sig'))
    mismatches = []
    for name, expected in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or digest(path) != expected:
            mismatches.append(name)
    if mismatches:
        raise ValueError('Frozen export differs; check local deployment changes: ' + ', '.join(mismatches))
    catalog = root / 'glomerulus_tools/catalog.csv'
    registry = json.loads((root / 'glomerulus_tools/runtime_registry.json').read_text(encoding='utf-8-sig'))
    if not (digest(catalog) == registry['catalog_sha256'] == manifest['catalog_sha256']):
        raise ValueError('catalog/registry/manifest identity mismatch')
    with catalog.open(encoding='utf-8-sig', newline='') as stream:
        rows = {row['tool_id']: row for row in csv.DictReader(stream)}
    retired = [key for key in rows if key.startswith('mescnn_')]
    retired += ['ritm_hrnet32_external_mask_point1', 'nepath_segnext_mscant',
                'skinet_mest_main_candidate', 'histoqc_wsi_quality_candidate',
                'tiatoolbox_stain_workflow_candidate']
    for key in retired:
        if rows[key]['enabled'] != 'no' or rows[key]['agent_enabled'] != 'no':
            raise ValueError('Unexpected enabled state: ' + key)
    result = dict(status='passed', files=len(manifest['files']), catalog_rows=len(rows),
                  research_enabled=sum(row['enabled']=='yes' for row in rows.values()),
                  agent_enabled=sum(row['agent_enabled']=='yes' for row in rows.values()),
                  scope='bytes, catalog identity and key disabled states; no inference or accuracy validation')
    return result


if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False))
