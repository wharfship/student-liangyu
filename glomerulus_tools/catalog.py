"""Read the generated registry; catalog.csv is the sole editable tool table."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT / 'runtime_registry.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_registry():
    data = json.loads(REGISTRY.read_text(encoding='utf-8'))
    if data['catalog_sha256'] != sha(ROOT / 'catalog.csv'):
        raise ValueError('catalog.csv changed: run python -m glomerulus_tools.build before dispatch')
    return data


def get_card(tool_id):
    for card in load_registry()['tools']:
        if tool_id == card['id'] or tool_id in card.get('aliases', []):
            return card
    raise ValueError('Unknown tool ID: ' + tool_id)


def asset_status(card):
    keys = ('python', 'io_python', 'checkpoint', 'source_dir', 'config','matrix_adapter')
    paths = ([card[k] for k in keys if card.get(k)] + card.get('detector_checkpoints', []) +
             card.get('required_assets', []) + card.get('extra_paths', []))
    missing = [p for p in paths if not Path(p).exists()]
    return dict(available=bool(card.get('enabled')) and not missing,
                enabled=bool(card.get('enabled')), missing_assets=missing)
