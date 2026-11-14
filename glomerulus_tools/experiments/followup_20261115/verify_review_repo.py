"""Verify and commit only the generated local snapshot, then test a fresh clone."""
import re,shutil,subprocess,sys,tempfile
from common import *

DEST=ROOT/'artifacts/student-liangyu'


def command(args,cwd):
 r=subprocess.run(args,cwd=cwd,text=True,encoding='utf8',capture_output=True)
 assert r.returncode==0,r.stderr+r.stdout
 return r.stdout


def verify_manifest(root):
 m=read(root/'provenance/export_manifest.json')
 for name,digest in m['files'].items():assert sha(root/name)==digest,name
 return len(m['files'])


def main():
 # Include this verifier and the final source identities in the generated copy.
 rel=Path('glomerulus_tools/experiments/followup_20261115')
 shutil.copy2(Path(__file__),DEST/rel/Path(__file__).name)
 source={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(HERE.glob('*.py'))}
 for p in [HERE/'README.md',REPORT/'结果汇报.md',ROOT/'glomerulus_tools/catalog.csv']:
  source[p.relative_to(ROOT).as_posix()]=sha(p)
 save(HERE/'final_source_manifest.json',dict(kind='final delivery source; not historic execution',files=source))
 shutil.copy2(HERE/'final_source_manifest.json',DEST/'results/followup_20261115/final_source_manifest.json')
 # Refresh complete byte manifest after adding verifier and generated receipts.
 names=set(command(['git','ls-files','-z'],DEST).split('\0'))
 names.update(command(['git','ls-files','--others','--exclude-standard','-z'],DEST).split('\0'));names.discard('')
 m=read(DEST/'provenance/export_manifest.json');names.discard('provenance/export_manifest.json')
 m['files']={n:sha(DEST/n) for n in sorted(names)};save(DEST/'provenance/export_manifest.json',m)
 assert m['catalog_sha256']==sha(DEST/'glomerulus_tools/catalog.csv')==sha(ROOT/'glomerulus_tools/catalog.csv')
 # Check export contents again after its final modification, without echoing secrets.
 patterns=[re.compile(r'sk-[A-Za-z0-9_-]{20,}'),re.compile(r'sk-ws\.'),re.compile(r'KB\d{5,}')]
 for name in names:
  p=DEST/name
  if p.suffix=='.xlsx':continue
  assert not any(q.search(p.read_text(encoding='utf-8-sig')) for q in patterns),name
 py_count=0
 for name in names:
  if name.endswith('.py'):compile((DEST/name).read_bytes(),name,'exec');py_count+=1
 tests=command([sys.executable,'-X','utf8','-m','pytest','glomerulus_tools/tests','-q'],DEST)
 assert '37 passed' in tests
 command([sys.executable,'-X','utf8','-m','glomerulus_tools','list'],DEST)
 checked=verify_manifest(DEST)
 command(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--check'],DEST)
 command(['git','add','--all'],DEST)
 command(['git','commit','-m','Record paired pyramid evaluation and follow-up evidence protocol'],DEST)
 commit=command(['git','rev-parse','HEAD'],DEST).strip()
 assert not command(['git','status','--porcelain'],DEST)
 clone=Path(tempfile.mkdtemp(prefix='student-liangyu-verify-20261115-',dir=ROOT/'artifacts'))
 command(['git','clone','--quiet','--no-hardlinks',str(DEST),str(clone)],ROOT)
 assert command(['git','rev-parse','HEAD'],clone).strip()==commit
 assert verify_manifest(clone)==checked
 clone_tests=command([sys.executable,'-X','utf8','-m','pytest','glomerulus_tools/tests','-q'],clone)
 assert '37 passed' in clone_tests
 command([sys.executable,'-X','utf8','-m','glomerulus_tools','list'],clone)
 assert not command(['git','status','--porcelain'],clone)
 assert not command(['git','remote'],DEST),'Review copy must have no remote'
 save(HERE/'repository_verification.json',dict(status='passed',packaging_commit=commit,files_SHA_checked=checked,
  python_sources_compiled=py_count,repository_tests=tests.strip(),clone_tests=clone_tests.strip(),
  clone_path=str(clone),catalog_identical_to_only_master=True,source_clean=True,clone_clean=True,
  packaging_commit_not_execution_commit=True,no_remote=True,no_push=True,no_feishu=True))
 print('Local packaging commit:',commit,'; fresh clone byte verification:',checked,'; both test runs:37 passed')


if __name__=='__main__':main()
