from discover import fetch
from common import *
import concurrent.futures
QUERIES={
 'grandqc_readme':'https://raw.githubusercontent.com/cpath-ukk/grandqc/main/README.md',
 'grandqc_tree':'https://api.github.com/repos/cpath-ukk/grandqc/git/trees/main?recursive=1',
 'grandqc_releases':'https://api.github.com/repos/cpath-ukk/grandqc/releases',
 'instanseg_index':'https://raw.githubusercontent.com/instanseg/instanseg/main/instanseg/bioimageio_models/model-index.json',
 'instanseg_pyproject':'https://raw.githubusercontent.com/instanseg/instanseg/main/pyproject.toml',
 'instanseg_inference':'https://raw.githubusercontent.com/instanseg/instanseg/main/instanseg/inference_class.py',
 'phikon_api_mirror':'https://hf-mirror.com/api/models/owkin/phikon?blobs=true',
 'phikon_card_mirror':'https://hf-mirror.com/owkin/phikon/raw/main/README.md',
}
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(fetch,QUERIES.items()))
 save(HERE/'discovery_more.json',records)
