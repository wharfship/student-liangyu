"""Update the generated local review copy using a narrow allowlist; never push."""
import re,shutil,subprocess,sys
from common import *

DEST=ROOT/'artifacts/student-liangyu'


def put(rel,text):
 p=DEST/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf8')


def copy(src,rel):
 p=DEST/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,p)


def main():
 assert (DEST/'.git').is_dir()
 status=subprocess.check_output(['git','status','--porcelain'],cwd=DEST,text=True)
 assert not status,'Generated review repository has changes; inspect before replacing'
 base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=DEST,text=True).strip()
 # Generated mirror of sole source catalog, never an independently maintained table.
 copy(ROOT/'glomerulus_tools/catalog.csv','glomerulus_tools/catalog.csv')
 for p in HERE.glob('*.py'):copy(p,'glomerulus_tools/experiments/followup_20261115/'+p.name)
 protocol=(HERE/'README.md').read_text(encoding='utf8')
 protocol=protocol.replace('[结果汇报](../../../已经整理好的内容/医院分割结果分析/专项后续_20261115/结果汇报.md)',
  '[公开汇总](../../../docs/FOLLOWUP_20261115.md)（医院内部图册不在本仓库）')
 put('glomerulus_tools/experiments/followup_20261115/README.md',protocol)
 safe_results=['paired_summary.json','paired_cost_and_compression.json','evaluation_verification.json',
  'pyramid_runs.json','pyramid_complete.json','decision_scoring_verification.json','delivery_verification.json',
  'executed_source_verification.json','legacy_object_summary.json','catalog_finalization.json',
  'observation_preparation.json','omission_gallery_verification.json']
 for name in safe_results:copy(HERE/name,'results/followup_20261115/'+name)
 # Keep actual execution SHA even though scored catalog status has since changed.
 copy(HERE/'execution_snapshot/manifest.json','results/followup_20261115/executed_manifest.json')
 assets=read(HERE/'asset_receipts.json')
 for a in assets:a['path']='assets/'+Path(a['path']).name
 put('provenance/skinet_asset_receipts.json',json.dumps(assets,ensure_ascii=False,indent=2))
 copy(HERE/'skinet_freeze.txt','environments/skinet_blocked_historical_freeze.txt')
 readme=(DEST/'README.md').read_text(encoding='utf8')
 readme=readme.replace('当前为2026-11-14本地整理快照','当前为2026-11-15本地整理快照')
 readme=readme.replace('32个enabled记录','34个enabled记录（包含同一模型的不同输入配方）')
 readme=readme.replace('不等于本仓库附带32个可用网络','不表示本仓库附带相应数量的独立可用网络')
 readme+='\n11月15日新增：MESCnn真实10×金字塔PNG/JPEG各80次仍无硬化净收益；旧SegNeXt有一例真实补漏但候选成本高；Skinet完整资产取得、TF导入受平台指令集阻塞；无GT观察资产和评分合同已备，未运行VLM。[后续结果与下一步](docs/FOLLOWUP_20261115.md)。\n'
 put('README.md',readme)
 toolreadme=(DEST/'glomerulus_tools/README.md').read_text(encoding='utf8').replace('32个历史enabled条目','34个历史enabled条目（含输入配方）')
 toolreadme=toolreadme.replace('不代表本快照附带32套源码、环境和权重','不代表本快照附带相应数量的独立网络源码、环境和权重')
 toolreadme=toolreadme.replace('## 登记表（自动生成）',
  '11月15日两个MESCnn输入配方已完成80+80实测，未改善硬化平均分割；Skinet完整资产取得但TF导入阻塞，仍禁用。详见[后续结果](../docs/FOLLOWUP_20261115.md)与实验源码。\n\n## 登记表（自动生成）')
 put('glomerulus_tools/README.md',toolreadme)
 put('docs/FOLLOWUP_20261115.md',PUBLIC_REPORT)
 for rel,extra in [('docs/RESULTS.md','本轮进一步验证的实际金字塔/JPEG结果与旧工具同目标对照'),
  ('docs/NEXT_EXPERIMENTS.md','下一步更新：人工确认遗漏；SegNeXt独有候选复核；Skinet平台兼容性验收；有效动作成立后再比较观察策略'),
  ('docs/REPRODUCIBILITY.md','本轮执行快照SHA与远程21文件一致；最终主表评分状态更新后SHA另记，不能替代实际运行快照')]:
  p=DEST/rel;put(rel,p.read_text(encoding='utf8')+'\n\n'+extra+'，见[11月15日后续](FOLLOWUP_20261115.md)。\n')
 p=DEST/'docs/DATA_AND_ASSETS.md'
 put('docs/DATA_AND_ASSETS.md',p.read_text(encoding='utf8')+'''

Skinet完整皮质与MEST主网络SavedModel来自 https://github.com/SkinetTeam/Skinet-MEST-C/releases/tag/v1.0 ，分别下载skinet_cortex.zip和skinet_mest_main.zip，放assets/并按作者方式解压（包含saved_model.pb/variables/label_map.pbtxt）。下载身份SHA在provenance/skinet_asset_receipts.json；发布页字节数核对一致，无作者另行哈希证据。此时完整权重已取得，不需重复下载。隔离TF2.15.1安装成功但服务器VM未暴露AVX，导入SIGILL，尚无模型结果；必须在兼容GPU平台先验收，不能把安装freeze当已验证可移植环境。训练MPP/染色/成员未知。\n''')
 subprocess.run([sys.executable,'-X','utf8','-m','glomerulus_tools.build'],cwd=DEST,check=True)
 # Scan bytes of all prospective tracked text, without printing matches.
 names=set(subprocess.check_output(['git','ls-files','-z'],cwd=DEST).decode().split('\0'))
 names.update(subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z'],cwd=DEST).decode().split('\0'));names.discard('')
 patterns=[re.compile(r'sk-[A-Za-z0-9_-]{20,}'),re.compile(r'sk-ws\.'),re.compile(r'KB\d{5,}')]
 flagged=[]
 for name in names:
  p=DEST/name
  assert p.suffix.lower() not in {'.jpg','.jpeg','.png','.tif','.tiff','.pth','.pt','.npz','.zip','.tgz'},name
  if p.suffix=='.xlsx':continue
  text=p.read_text(encoding='utf-8-sig')
  if any(r.search(text) for r in patterns):flagged.append(name)
 assert not flagged,'Export content check failed in '+str(flagged)
 manifest='provenance/export_manifest.json';names.discard(manifest)
 files={name:sha(DEST/name) for name in sorted(names)}
 put(manifest,json.dumps(dict(date='2026-11-15',kind='generated local review snapshot',previous_packaging_commit=base,
  historical_run_commit=None,actual_execution='results/followup_20261115/executed_manifest.json',
  catalog_sha256=sha(ROOT/'glomerulus_tools/catalog.csv'),files=files,
  exclusions=['hospital images','patient-level results','GT','model weights','credentials','AGENTS.md'],
  no_push=True,no_feishu=True),ensure_ascii=False,indent=2))
 save(HERE/'review_repository_update.json',dict(path=str(DEST),base_commit=base,files=len(files)+1,
  secret_pattern_matches=0,pushed=False,feishu_sent=False))
 print('Local generated review copy updated:',len(files)+1,'files; no patient images or row-level results')


PUBLIC_REPORT='''# 11月15日后续实验：实际结果与边界

新增两个输入配方、160次真实推理，**未找到硬化净收益**。保持SegFormer基线，尚未证明动态Agent有效。本页为内部全队列报告的脱敏汇总，病例图册和逐行结果不导出。

|配方|硬化40目标Dice|普通40目标Dice|硬化改善/恶化>0.01|
|---|---:|---:|---:|
|SegFormer保留|0.842106|0.914157|—|
|MESCnn原page0 AREA→10×|0.676153|0.740630|6/20|
|MESCnn实际10×金字塔PNG|0.676892|0.717101|7/20|
|MESCnn实际10×金字塔JPEG95|0.676851|0.717119|6/20|

两新配方同原权重/4096原生窗/score0.5、无面积过滤。80/80通过作者最近10×倍率检查。两个配方都未救回3个完全遗漏，硬化匹配30/40、普通31/40（SegFormer37/40和40/40）。普通是已检出保护对照，不能估计普通总体漏检。硬化HE35/PAS5，仅开发暴露数据；未复现完整WSI tiling/NMS，仍有取整/亚像素映射差异。

目标指标按离线GT选最佳实例评价，不能当在线选择或完整WSI成绩。三个无预测目标由GT定位参考窗，不能证明部署时会发现它们。分组区间/原命令/执行SHA/输出验收见results/followup_20261115，严禁将打包commit冒称运行commit。

旧工具按原任务/实例ID复用：SegNeXt在同40硬化匹配36，救回1例完全遗漏、丢失2例原匹配，普通匹配38/40。该例预测来自旧图像全WSI推理。医院40图独有65候选仅1个GT匹配、64未匹配，追加策略FP风险高；未匹配不是病理确定非球，仍须复核漏标。Glo19/40、Omni CAP2/40、unanimity1/40，未救回三个完全遗漏。不能据一例宣称稳定补救或Agent收益。

Skinet完整皮质/主网络SavedModel取得，label3=nsg/4=pac；TF2.15.1独立安装成功，导入原生模块SIGILL−4。VM未暴露AVX/AVX2/FMA，模型尚未加载；这是平台阻塞，无分割精度结果。训练MPP/染色/成员未知，仍禁用。资产获取/SHA见docs/DATA_AND_ASSETS.md与provenance；没有权重随仓库分发。

内部40例人工复核证据和工作表已备，病理判断未代填。77个已定位目标生成539张具名JPEG观察资产，GT映射只用于离线评价。决策合同和keep/accept复现通过；**未运行VLM，没有动态观察收益数据**。仅2个硬化RITM提案改善>0.01、4个恶化，不能以此急于大批付费调用。VISTA原文尚未核清，不宣称方法复现。

下一步先确认真遗漏与GT，再复核SegNeXt独有候选，并在兼容GPU环境验收Skinet完整链。若无有效现成补救动作，考虑按患者独立划分的专用适配/可靠纠错点。动作成立后比较SegFormer、有效专网、固定规则和同预算Agent；固定单视图/固定多视图/动态观察需共享动作与预算，记录误伤、FP、复核量和真实传输成本。当前80例不得再叫独立测试。

160次推理进程墙时390.59秒，裁图78.43秒、观察资产125.17秒，环境安装/下载另计。付费VLM/API请求0次，无训练、无上传/飞书。统一工具37项测试通过；HTML/JPEG/SHA/决策合同验收是工程正确性，不替代精度或临床验证。元数据70行/34 enabled/11 Agent，包含配方/组件/暂停项，不等于70个网络。

本轮代码与协议在glomerulus_tools/experiments/followup_20261115。内含历史路径，需授权数据与原冻结结果才可复算。该本地仓库仍是原工作区唯一catalog的生成审阅副本，避免双处人工维护。
'''


if __name__=='__main__':main()
