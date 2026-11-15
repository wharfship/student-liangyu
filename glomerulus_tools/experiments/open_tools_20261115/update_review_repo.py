"""Export only code and aggregates to the existing generated local Git review copy."""
import shutil,subprocess,sys,re,tempfile,json
from common import *
DEST=ROOT/'artifacts/student-liangyu'

def run(args,cwd=DEST):
 r=subprocess.run(args,cwd=cwd,capture_output=True,text=True,encoding='utf8');assert r.returncode==0,r.stderr+r.stdout;return r.stdout
def put(rel,text):
 p=DEST/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf8')
def copy(src,rel):
 p=DEST/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,p)

def main():
 assert not run(['git','status','--porcelain']),'Preserve unrelated changes'
 base=run(['git','rev-parse','HEAD']).strip()
 for p in (ROOT/'glomerulus_tools').glob('*.py'):copy(p,'glomerulus_tools/'+p.name)
 for p in (ROOT/'glomerulus_tools/tests').glob('*.py'):copy(p,'glomerulus_tools/tests/'+p.name)
 copy(ROOT/'glomerulus_tools/catalog.csv','glomerulus_tools/catalog.csv')
 for p in HERE.glob('*.py'):copy(p,'glomerulus_tools/experiments/open_tools_20261115/'+p.name)
 protocol=(HERE/'README.md').read_text(encoding='utf8')
 protocol=protocol.replace('[实际结果与直接测试](../../../已经整理好的内容/医院分割结果分析/开源工具扩展_20261115/结果报告.md)', '[脱敏结果](../../../docs/OPEN_TOOLS_20261115.md)')
 protocol=protocol.replace('[113组图册](../../../已经整理好的内容/医院分割结果分析/开源工具扩展_20261115/gallery.html)', '113组图册仅在授权内部目录')
 protocol+='\n注意：本导出不包含医院样例、原机器受保护的服务器手册或模型权重；一键远程客户端目前在原授权工作区验收。外部机器须先配置其自身服务器手册、路径和授权数据，不能从此导出免数据直接跑医院样例。\n'
 put('glomerulus_tools/experiments/open_tools_20261115/README.md',protocol)
 names=['acceptance_summary.json','fixed_fov_control_summary.json','feature_input_audit.json','evaluation_verification.json','delivery_verification.json',
  'weight_verification.json','cost_summary.json','catalog_finalization.json','smoke_complete.json','pilot_runs.json','retrieval_verification.json']
 for name in names:copy(HERE/name,'results/open_tools_20261115/'+name)
 copy(HERE/'execution_snapshot/manifest.json','results/open_tools_20261115/executed_manifest.json')
 copy(HERE/'requirements-executed-freeze.txt','environments/open_tools_historical_freeze.txt')
 put('docs/OPEN_TOOLS_20261115.md',PUBLIC)
 for rel in ['README.md','glomerulus_tools/README.md']:
  text=(DEST/rel).read_text(encoding='utf8').replace('37项测试','47项测试').replace('37项统一','47项统一')
  text=text.replace('34个enabled记录（包含同一模型的不同输入配方）','38个enabled记录（包含配方和非分割工具）').replace('34个历史enabled条目（含输入配方）','38个研究enabled条目（含配方与非分割工具）')
  link='docs/OPEN_TOOLS_20261115.md' if rel=='README.md' else '../docs/OPEN_TOOLS_20261115.md'
  text+='\n新增InstanSeg核分析、DINOv2表征、GrandQC组织/伪影4入口，226份正式CUDA输出，能力与许可边界见[新工具验收]('+link+')。唯一人工主表仍在原工作区，此处是生成审阅副本。\n'
  put(rel,text)
 for rel in ['docs/RESULTS.md','docs/DATA_AND_ASSETS.md','docs/REPRODUCIBILITY.md']:
  put(rel,(DEST/rel).read_text(encoding='utf8')+'\n新增工具的结果、模型取得、物理尺度与执行边界见[新工具验收](OPEN_TOOLS_20261115.md)。\n')
 run([sys.executable,'-X','utf8','-m','glomerulus_tools.build'])
 files=set(run(['git','ls-files','-z']).split('\0'));files.update(run(['git','ls-files','--others','--exclude-standard','-z']).split('\0'));files.discard('');files.discard('provenance/export_manifest.json')
 patterns=[re.compile(r'sk-[A-Za-z0-9_-]{20,}'),re.compile(r'sk-ws\.'),re.compile(r'KB\d{5,}')]
 for name in files:
  p=DEST/name;assert p.suffix.lower() not in {'.jpg','.jpeg','.png','.tif','.tiff','.pth','.pt','.zip','.tgz','.npy','.npz'},name
  if p.suffix=='.xlsx':continue
  text=p.read_text(encoding='utf-8-sig');assert not any(r.search(text) for r in patterns),name
  if p.suffix=='.py':compile(text,name,'exec')
 save(DEST/'provenance/export_manifest.json',dict(date='2026-11-15',kind='generated review snapshot',previous_packaging_commit=base,
  catalog_sha256=sha(ROOT/'glomerulus_tools/catalog.csv'),historical_run_commit=None,
  actual_execution='results/open_tools_20261115/executed_manifest.json',files={name:sha(DEST/name) for name in sorted(files)},
  exclusions=['credentials','AGENTS.md','hospital images','patient-level results','GT','model weights'],no_push=True,no_feishu=True))
 run(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--check'])
 run(['git','add','--all']);run(['git','commit','-m','Add accepted nuclei, feature and QC utilities with measured task boundaries'])
 commit=run(['git','rev-parse','HEAD']).strip();assert not run(['git','status','--porcelain']);assert not run(['git','remote'])
 clone=Path(tempfile.mkdtemp(prefix='student-liangyu-open-tools-verify-',dir=ROOT/'artifacts'))
 run(['git','clone','--quiet','--no-hardlinks',str(DEST),str(clone)],cwd=ROOT)
 manifest=read(clone/'provenance/export_manifest.json')
 for name,digest in manifest['files'].items():assert sha(clone/name)==digest,name
 tests=run([sys.executable,'-X','utf8','-m','pytest','glomerulus_tools/tests','-q'],cwd=clone);assert '47 passed' in tests
 assert not run(['git','status','--porcelain'],cwd=clone)
 save(HERE/'repository_verification.json',dict(status='passed',packaging_commit=commit,base=base,files_SHA_checked=len(files),clone=str(clone),clone_tests=tests.strip(),
  no_remote=True,no_push=True,no_feishu=True,patient_assets_exported=False,packaging_commit_not_execution_commit=True))
 print('Updated local review commit',commit,'; fresh clone',len(files),'file SHAs;47tests passed')

PUBLIC='''# 新开源能力：真实下载、CUDA推理与限定验收

2026-11-15新增三个家族、四个入口：InstanSeg明场细胞核、DINOv2表征、GrandQC组织/伪影。正式226份输出（首轮149＋固定视野复核77），4个smoke及本机客户端实际复测。没有训练或付费API，没有推送上传。元数据74条、38研究enabled、11 Agent enabled，不等于74个网络，默认SegFormer不变。

|能力|本轮结果|限制|
|---|---|---|
|InstanSeg核实例|TNBC8/4Slides前景Dice0.857220、组件代理F10.910232；另16医院局部完成|TNBC是已知训练来源，精确划分未知，非独立测试；医院无核GT，PAS4仅探索|
|DINOv2检索|77候选同染色跨来源号组同类首近邻73/77，RGB51/77；固定224um控制63/77，RGB42/77|无训练、无分类头；裁片敏感、队列已暴露；不是肾病分类准确率，更不是球真假验证|
|GrandQC组织|24HE局部组织mask与真实state_dict严格加载通过|无组织边界GT，10um粗格ROI适配，非完整WSI复现|
|GrandQC伪影|人工严重模糊12/12失焦比例增加>0.01，均值0→0.649134|无真实伪影GT；不能报真实敏感性/特异性；非商业许可|

InstanSeg与DINOv2为Apache-2.0；InstanSeg还须遵守来源数据许可。GrandQC为CC-BY-NC-SA-4.0，可作非商业研究工具，不当无限制OSI开源。引用：Weng et al., Nature Communications2024, doi:10.1038/s41467-024-54769-y，官方许可 https://github.com/cpath-ukk/grandqc/blob/main/LICENSE 。InstanSeg引用 doi:10.48550/arXiv.2408.15954；DINOv2 https://arxiv.org/abs/2304.07193 。

原图、GT、病例表和图册仅内部留存。此库包含新模块pathology_utilities.py、严格输入协议、10项新增合同检查、来源代码、环境/执行SHA和脱敏汇总。模型权重不附Git，但本机与实验室已取得并实际运行。

获取与放置（按catalog.execution_json设置对应机器绝对路径）：

- InstanSeg： https://github.com/instanseg/instanseg/releases/download/instanseg_models_v0.1.1/brightfield_nuclei.zip ，解压为assets/instanseg_model/instanseg.pt及rdf；模型SHA与作者rdf一致，详见results/open_tools_20261115/weight_verification.json。
- DINOv2： https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth ，保存assets/dinov2_vits14_pretrain.pth，源码固定作者commit见执行catalog/下载脚本；本地SHA固定、严格加载，不冒称有独立作者完整哈希校验。
- GrandQC组织： https://zenodo.org/records/14507273 ；伪影1.5um： https://zenodo.org/records/14041538 ，放assets/Tissue_Detection_MPP10.pth、GrandQC_MPP15.pth；与Zenodo MD5一致。原始许可/引用必保留。

当前已验收环境为124服务器私有tool_venv，复用已有torch2.5.1+cu121/torchvision0.20.1，numpy1.26.4/SMP0.3.1/timm0.4.12/OpenCV4.10/InstanSeg依赖；既有环境未改变。新机器需配置PyTorch/CUDA、固定源码/权重、catalog路径、build和smoke，不能把历史freeze当一键安装承诺。

统一推理入口：`python -m glomerulus_tools run --tool ID --task BATCH_NO_GT.json --output NEW_DIR --research`。协议见[实验README](../glomerulus_tools/experiments/open_tools_20261115/README.md)。输入只有图像/SHA、MPP/尺度依据、染色和匿名ID，无GT。核实例输出回原尺寸；QC保留10/1.5um工作格，坐标需按MPP映射；特征384维，无诊断概率。

原授权工作区的try_tool.py已实测默认features/qc及上传自定义核图流程，自动选择空闲GPU并取回本地。此导出不含医院样例及受保护服务器手册，外部机器须先配置自己的授权数据、服务器手册/路径，不能直接调用内部默认样例。search_features.py只读向量不读GT，原始检索只排除同一文件，论文式来源/染色过滤须显式另外执行。

本地审阅copy始终由原工作区唯一catalog生成；新打包commit不冒称运行commit，实际运行用execution_snapshot与record SHA追溯。当前工具尚未证明能改善SegFormer或让Agent更好，后续由研究者决定是否编排。
'''

if __name__=='__main__':main()
