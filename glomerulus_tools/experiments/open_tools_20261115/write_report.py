"""Generate results, source protocol and ready-to-run instructions from receipts."""
from common import *
import csv,hashlib,numpy as np

def main():
 summary=read(HERE/'acceptance_summary.json');ctrl=read(HERE/'fixed_fov_control_summary.json');runs=read(HERE/'pilot_runs.json')
 wall=sum(r['seconds'] for r in runs);nuc=summary['nuclei_TNBC'];raw=next(s for s in summary['retrieval'] if s['model']=='DINOv2' and s['stain']=='ALL');fixed=next(s for s in ctrl['summaries'] if s['model']=='DINOv2' and s['stain']=='ALL')
 verification=[]
 for meta,name in [('grandqc_zenodo_tissue','Tissue_Detection_MPP10.pth'),('grandqc_zenodo_artifacts','GrandQC_MPP15.pth')]:
  source=next(x for x in read(HERE/'sources'/(meta+'.txt'))['files'] if x['key']==name);p=HERE/'assets'/name
  assert source['checksum']=='md5:'+hashlib.md5(p.read_bytes()).hexdigest()
  verification.append(dict(file=name,url=source['links']['self'],bytes=p.stat().st_size,sha256=sha(p),author_MD5=source['checksum'],author_MD5_matched=True))
 p=HERE/'assets/instanseg_model/instanseg.pt';assert sha(p)=='0c724da169d507d8dd2e206ad3675a0b1566824eee6e20ad9a0ba916311d9dd0'
 verification.append(dict(file='instanseg_model/instanseg.pt',sha256=sha(p),author_rdf_SHA_matched=True))
 p=HERE/'assets/dinov2_vits14_pretrain.pth';verification.append(dict(file=p.name,url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth',bytes=p.stat().st_size,sha256=sha(p),independent_upstream_SHA_verified=False,strict_state_load=True))
 save(HERE/'weight_verification.json',verification)
 report=f'''# 新开源工具已下载、配置并实测：2026-11-15

这次取得了**3个工具家族、4个可调用入口**，完成226份正式推理输出（149项首轮＋77项固定视野复核），另有4个smoke和本地一键客户端复测。全部实际使用CUDA，非随机张量测试，也没有用旧工具改名凑数。

**建议优先试InstanSeg和DINOv2；GrandQC可作为有非商业限制的质控候选。** “能用”已有运行和产物证据；“好用”限于下面实测功能与数据。这里没有承诺它们会改善肾小球分割，也不把任何输出当成已经验证的临床结论。

[全部113组实际输出图册](gallery.html) · [工具协议与源码](../../../glomerulus_tools/experiments/open_tools_20261115/README.md)。本轮未上传仓库、未发飞书、未调用付费API。

## 结果和推荐用途

|工具与直接测试别名|真正提供的能力|本轮实际验收|目前判断|
|---|---|---|---|
|InstanSeg `nuclei`|细胞核实例、数量、核面积、原图坐标实例图|TNBC8/4切片平均核前景Dice **{nuc['mean_binary_dice']:.4f}**，组件代理F1 **{nuc['mean_instance_f1']:.4f}**；另16医院局部全部运行|可优先作为核分析工具试用；本地无核GT，不能给医院精度保证|
|DINOv2 ViT-S/14 `features`|384维图像表征、相似局部检索，无分类头|77候选跨来源组、同染色检索同类首近邻**73/77**，RGB基线51/77；固定224μm视野复核**63/77**，RGB42/77|有检索价值；对视野敏感，不能将94.8%写成硬化分类准确率|
|GrandQC tissue `tissue`|粗尺度组织/背景掩膜，供质量分析门控|24HE局部（原始12＋人工模糊12），真实模型严格加载，全部有合法输出|工程已验收；组织边界真实精度未测|
|GrandQC artifact `qc`|折叠、暗点/异物、笔迹、气泡/边缘、失焦及背景/组织的类图|人工严重模糊12/12失焦比例增加>0.01，均值0→0.6491；完整任务模型与组织链均实际运行|人工失焦响应通过；真实伪影识别精度还需标注验证|

InstanSeg与DINOv2为Apache-2.0；InstanSeg模型包还列明来源数据许可义务。GrandQC为CC-BY-NC-SA-4.0，**属于可供本次非商业研究使用的公开源码/权重，不是无限制的OSI开源许可**，应单独考虑。所有来源与许可在唯一catalog和源码许可文件中登记。

## 1. InstanSeg：这次最完整的任务精度验收

下载作者`brightfield_nuclei` v0.1.1完整TorchScript模型；权重SHA与作者包内rdf逐字匹配。作者方法支持明场细胞核，模型工作尺度0.5μm/px；输入必须提供真实或有依据的MPP，wrapper按尺度缩放，预测回原图尺寸。本轮固定作者wrapper源码、百分位归一化及CUDA，保留uint32实例TIFF。实例输出不是肾小球mask；核数和面积也不是硬化判定规则。

公开TNBC验收在看输出前固定：按名称排序取前4个Slide目录、每目录前2张，共8图；635个GT连通组件、581个预测核、IoU≥0.5一对一匹配559个。平均前景Dice0.857220，平均组件代理F1=0.910232。原GT为二值标注，按作者loader做连通组件，接触核未保证独立ID，因此称“组件代理F1”。逐图指标：[TNBC8验收表](TNBC8_细胞核验收_EVAL_ONLY.csv)。

**TNBC是模型明确使用过的训练来源，未取得精确字段划分，因此这只是功能/质量验收，不能作为独立测试。** 医院16局部中HE12、PAS4（探索），均非空，检出数6–283；医院没有细胞核GT，图册用于人工查看，不制造Dice。PAS的有效性不能从HE外推，仍待确认。

官方随包test-input与reference prediction的前景一致率99.99237%，实例标签并非完全逐字节相同，差5个前景像素；参考输出本身是作者模型预测，不是专家GT。单图smoke与正式首样本输出则字节完全一致。没有隐瞒硬件/归一化引起的微小差别。

## 2. DINOv2：作为特征/检索工具有实测价值

取得Meta公开ViT-S/14预训练权重和固定官方源码，384维CLS，短边Resize256、CenterCrop224、ImageNet归一化；state_dict严格加载，无新分类头、无训练、无文本提示。它是自然图像通用自监督模型，**不是病理专用模型**。本轮只验证它能否提供有用的形态检索表征。

77个已定位目标由现有预测mask的包围框加20%边缘裁图，没有GT图层；标签、原实例与来源号映射仅用于离线验收。原图像素SHA无重复，查询的候选库排除同来源号组，只在相同染色中找最近邻。队列是旧40硬化＋40已检出普通保护目标中可定位的77例，完全漏掉的三例不在其中，全部属于已暴露开发数据。

原裁片同类首近邻73/77（94.81%），HE67/68、PAS6/9，Top5同类比例87.01%；RGB直方图基线51/77。由于结果较强，追加**预先固定224μm原生方形视野**的77次控制，避免只凭包围框裁片比例下结论。固定视野同类首近邻63/77（81.82%），HE57/68、PAS6/9，Top5比例78.70%，RGB42/77。原结果和复核结果都保留，不择优隐藏。

这支持“图像表征可用于相似病例检索”的初步结论，也显示上下文/裁片尺度对性能影响明显。**不是球/非球分类任务，也不是独立硬化分类测试。** 目标来自GT知情队列，原类别未本轮病理复诊；病例数小、来源相关、PAS仅9例。精确患者身份与训练成员重叠未知，没有给出临床分类概率。检索的cosine只是相似度，不是诊断置信度。

[原裁片逐例检索](77候选_跨来源组同染色检索_EVAL_ONLY.csv) · [固定224μm复核](固定224微米_同染色跨组检索_EVAL_ONLY.csv)。图册展示全部查询和前三近邻。`search_features.py`只读向量、不读GT；原始索引检索只排除同一文件，若用于研究复现实验还须按来源/染色限制候选库，不能混淆两种协议。

## 3. GrandQC：组织和图像质量工具

下载官方组织与1.5μm/px伪影模型，两个文件均与Zenodo MD5一致，并保存完整SHA256。组织为Unet++ EfficientNet-b0任务state_dict、严格加载；伪影是作者完整模型。保留组织10μm/px→伪影1.5μm/px的门控链、作者组织JPEG80编码/OpenCV读取顺序以及伪影RGB/ImageNet预处理。

当前封装是**有限ROI协议**：按物理尺寸取整重采样，白色补齐512，组织标签最近邻门控；没有冒称执行作者完整WSI脚本。作者WSI偏移/边界拼接不同，不能将局部结果当完整WSI复现。输出保留工作MPP，图像坐标换算为`原图坐标=工作格坐标×工作MPP/源MPP`，取整误差另留；不会直接覆盖原图或SegFormer。

12个HE原始局部与每张σ=8μm严重高斯模糊副本共24项。原图失焦类均值0、模糊后0.649134，12/12增加超过0.01。此合成扰动很强，**只说明工具有预期响应，不能据此报告真实伪影敏感性/特异性或自动剔除阈值**。没有真实伪影/组织标注；各类图、组织mask与原始图均在本地。未测试PAS。

原始类编号：1正常组织、2折叠、3暗点/异物、4笔迹、5气泡/边缘、6失焦、7背景。未知0保留单独统计；本轮24图该比例均0。不能把正常硬化组织、胶原或色深自动称作伪影。

引用：Weng et al., *GrandQC: a comprehensive solution to quality control problem in digital pathology*, Nature Communications, 2024，doi:10.1038/s41467-024-54769-y。许可见官方LICENSE，非商业研究使用，集成时保留引用与许可声明。

## 你现在怎样直接测试

已配置的是实验室124服务器GPU环境，使用已有SSH密钥；本机不要求装PyTorch或CUDA。以下从项目根目录PowerShell运行，使用已存在的真实样例，每次新建输出目录并取回本机：

```powershell
python -X utf8 glomerulus_tools/experiments/open_tools_20261115/try_tool.py --tool nuclei
python -X utf8 glomerulus_tools/experiments/open_tools_20261115/try_tool.py --tool features
python -X utf8 glomerulus_tools/experiments/open_tools_20261115/try_tool.py --tool tissue
python -X utf8 glomerulus_tools/experiments/open_tools_20261115/try_tool.py --tool qc
```

测试自己的局部图（保持原图不覆盖）：

```powershell
python -X utf8 glomerulus_tools/experiments/open_tools_20261115/try_tool.py --tool nuclei --image "C:/绝对路径/patch.png" --mpp 0.25 --scale-evidence "请替换为该图真实元数据或标定依据" --stain HE
```

0.25仅示例，必须用自己图像的真实MPP；QC只接受HE，本轮核工具PAS仍探索。上限单边8192，核工具建议先512–1024px；整张WSI请先按合法图像坐标裁局部。脚本完整读取指定服务器手册、检查当前空闲GPU、显式禁用CPU回退、核对输入和权重SHA，通过统一`python -m glomerulus_tools run`调用**实际已测执行快照**。无空闲GPU会停止，不占用其他任务。图像只发送到指定实验室服务器，不发送VLM或第三方API。

结果下载到`glomerulus_tools/experiments/open_tools_20261115/manual_tests/manual_.../`，含输入副本、任务JSON、执行收据与`results/`。细胞核看`instances.tif`/`overlay.jpg`，特征看`embedding.npy`/`embedding_l2.npy`，质控看`tissue_classes.png`/`artifact_classes.png`/`overlay.jpg`，每例均有`record.json`。模型结果是数据，尚不允许Agent自动修改原分割。

已保存的特征可直接作相似检索，不再次调用GPU：

```powershell
python -X utf8 glomerulus_tools/experiments/open_tools_20261115/search_features.py --query glomerulus_tools/experiments/open_tools_20261115/results/features77/P001/embedding_l2.npy --index glomerulus_tools/experiments/open_tools_20261115/results/features77 --top-k 5
```

## 下载、环境、协议与版本

- 模型真实文件：`glomerulus_tools/experiments/open_tools_20261115/assets/`，服务器同相对目录已放置；无需再下载。源码固定commit压缩包及解压目录都在`assets/source/`。原始作者入口和身份见`download_receipts.json`、`weight_verification.json`、各API元数据档案。
- 权重：InstanSeg约15MB、DINOv2约88MB、GrandQC组织26.6MB/伪影25.4MB。下载不等于验收，本轮已逐一执行实际推理。
- 执行环境：独立`tool_venv`，复用已有sam的torch2.5.1+cu121/torchvision0.20.1，不修改旧环境；私有numpy1.26.4、SMP0.3.1、timm0.4.12、InstanSeg0.1.2依赖、OpenCV4.10。初次安装发现继承的OpenCV4.12与numpy版本约束冲突，已在私有环境覆盖为4.10；原日志保留，未隐去。完整freeze和来源在实验目录；继承环境含其他无关包，freeze不是最小便携安装清单。
- 实际执行代码：`execution_snapshot/manifest.json`及`retrieval_verification.json`，取回时与远程逐文件SHA一致；记录每项输入/权重/适配器SHA、MPP、设备、时间和显存。最终catalog更新验证结论后SHA变化，保留执行时原快照，不伪造运行commit。
- 首轮5组进程墙时合计{wall:.2f}秒，含模型加载和输出；固定视野控制、4次smoke和客户端复测另计。GPU4任务已退出，未修改其他用户环境或进程。无训练、无付费VLM/API。
- 唯一主表新增4行，当前74条（含配方/组件）、38研究enabled、11 Agent enabled；Agent池和默认SegFormer不变。工具输入严格只接受图像、SHA、MPP、尺度依据和染色等协议字段，拒绝GT和离线分数。47项测试通过，图册/客户端/产物验证见收据。

## 没列入可用工具的候选

Hibou官方模型需要账户授权，不绕过门槛；Kaiko源码为非商业许可且两次官方checkpoint下载连接重置，未取得任务权重，**不算本轮可用工具**；Phikon本来已调研，HF直连超时，镜像元信息可读，但未借机重新当成新的专用分类器。没有为凑工具数量再次重跑MESCnn、SAMRefiner、RITM或MORI。

下一步由你决定如何使用这些能力。当前建议：InstanSeg做核数量/形态证据，DINOv2做类似区域查找和参考样例检索，GrandQC做输入质量提示。它们是否值得进入分割Agent、如何验收修改，仍需独立证明，不能由这份工具验收直接推出。
'''
 (REPORT/'结果报告.md').write_text(report,encoding='utf8')
 (REPORT/'README.md').write_text('# 新开源工具验收\n\n[实际结果与一键测试命令](结果报告.md) · [全部113组原图/输出](gallery.html)。InstanSeg/DINOv2为Apache工具；GrandQC为非商业研究许可。全部权重已下载并在GPU实跑，能力和数据边界分别报告。\n',encoding='utf8')
 protocol='''# 新工具运行协议与实验记录

本轮用户要求继续调研任意可用、好用的开源能力，实际下载、跑通、配置可直接测试的协议。没有上传、飞书或付费API调用。工具方法信息仍统一在glomerulus_tools/README.md，唯一人工主表catalog.csv。

内部报告：[实际结果与直接测试](../../../已经整理好的内容/医院分割结果分析/开源工具扩展_20261115/结果报告.md)；[113组图册](../../../已经整理好的内容/医院分割结果分析/开源工具扩展_20261115/gallery.html)。

## 输入协议v1

统一入口为`python -m glomerulus_tools run/check --tool TOOL_ID --task ... --output NEW_DIR --research`，适配器`pathology_utility`。四个ID为`instanseg_brightfield_nuclei_v011`、`dinov2_vits14_features`、`grandqc_tissue_mpp10`、`grandqc_artifact_mpp15`。均研究enabled、Agent禁用。语义是细胞核/特征/QC，不是whole_glomerular_unit；拒绝用分割mask结果冒充工具任务。

```json
{
  "schema_version": 1,
  "samples": [{
    "sample_id": "EXAMPLE01",
    "image": "/ABS/authorized_patch.png",
    "image_sha256": "REPLACE_WITH_REAL_64_HEX_SHA256",
    "source_mpp": 0.25,
    "scale_evidence": "REPLACE_WITH_ACTUAL_METADATA_OR_CALIBRATION;0.25_IS_EXAMPLE",
    "stain": "HE"
  }]
}
```

严格只接受这些字段，未知字段（含GT、标签、质量分数）拒绝；ID唯一无路径穿越、有限正MPP、绝对图像路径和64hex SHA必需。输入图像RGB，原图/GT/mask原格式保留；图册和任何VLM发送副本JPEG，当前无VLM调用。每例权重SHA/输入SHA、设备、尺寸/MPP、秒数、显存记录；输出目录必须新建，失败不覆盖旧结果。

本机直接用`try_tool.py --tool nuclei/features/tissue/qc`，默认真实已测样例；自定义局部图加`--image --mpp --scale-evidence --stain`。用户界面实际调用服务器的execution_snapshot，避免更新主表影响已验收配置；每次实时选择有>12GB可用显存的卡，无GPU则报错。程序读服务器手册、用现有SSH key，无凭据写入命令。自定义输入上传到该实验manual_inputs，结果取回manual_tests新目录；仅实验室传输，无第三方API。

服务器原生示例（先完整读指定服务器手册、实时检查GPU，再设CUDA_VISIBLE_DEVICES，编号不得照搬）：

```bash
cd /home/chenliangyu/project/new_agent/glomerulus_tools/experiments/open_tools_20261115/execution_snapshot
/home/chenliangyu/project/new_agent/glomerulus_tools/experiments/open_tools_20261115/tool_venv/bin/python -m glomerulus_tools check --tool instanseg_brightfield_nuclei_v011 --task ../nuclei_remote_NO_GT.json --research
# GPU环境变量由实时检查决定；输出必须是从未存在的新目录
/home/chenliangyu/project/new_agent/glomerulus_tools/experiments/open_tools_20261115/tool_venv/bin/python -m glomerulus_tools run --tool instanseg_brightfield_nuclei_v011 --task ../nuclei_remote_NO_GT.json --output ../my_new_run --research
```

特征检索`search_features.py`只读384维L2向量、不读GT。实际跨来源同染色验收使用独立离线来源映射排除同组；简单查询脚本只排除同文件，若要复现论文式实验需同样过滤。cosine不是概率，当前无分类头。

## 输出和坐标

InstanSeg：`instances.tif` uint32实例ID、原图同尺寸，背景0；`overlay.jpg`、record中的核数/面积。作者模型0.5um/px，wrapper依据输入MPP缩放后回原网格；面积乘原MPP平方。此核数不是肾小球细胞类型或硬化分类。

DINOv2：`embedding.npy`与`embedding_l2.npy`均384维，固定作者ViT-S/14；Resize短边256/CenterCrop224、ImageNet归一化，无分类头。MPP记入元数据，预训练物理尺度未知，输入FOV对检索有影响。

GrandQC：组织类图10um/px，0组织/1背景；artifact类图1.5um/px，1组织/2折叠/3暗点异物/4笔迹/5气泡边缘/6失焦/7背景，未知0单独保留。原图坐标换算x_source=x_output*MPP_output/MPP_source，固定物理网格舍入有误差。ROI白补512，保留作者JPEG80/OpenCV组织通道与伪影RGB预处理，标签门控用最近邻；非完整WSI offsets/余数拼接复现。GrandQC引用与CC-BY-NC-SA许可声明在主模块和README中保留。

## 实际小范围验收

首轮5批149项：核医院16（HE12/PAS4探索）、TNBC8/4Slides（已知训练来源，不是独立测试）、DINO77（37硬化/40已检出普通、均暴露开发）、组织24/伪影24（HE12原图＋人工严重模糊12）。各工具先smoke exit0才扩批；四个首样本smoke/正式输出字节一致。InstanSeg随包reference prediction前景一致99.99237%，不是专家GT。

检索结果较强后追加固定224um视野77项：无重复原图RGB SHA；排除同来源组、仅同染色检索。原73/77，固定视野63/77，RGB51/77和42/77。保留视野敏感性、不夸成硬化诊断。没有训练模型，没有在GT上挑seed/阈值。所有指标和限制见acceptance_summary/fixed_fov_control_summary及内部CSV。

## 来源/环境/复现

`assets/`本机和服务器同相对目录均有真实权重；`assets/source/`是固定上游commit源码。InstanSeg torchscript SHA与rdf一致；GrandQC两权重与Zenodo MD5一致；DINOv2官方直链文件本地SHA固定，但未宣称有独立的作者完整哈希核对。weight_verification.json留档。

torch2.5.1+cu121/torchvision0.20.1来自已有sam环境，私有tool_venv以system-site-packages复用但不修改原环境。私有numpy1.26.4/scipy1.13.1/SMP0.3.1/timm0.4.12/scikit-image0.24/matplotlib3.9.4/fastremap1.17.7/einops0.8.1/OpenCV4.10.0.84。InstanSeg固定作者源码覆盖PyPI wrapper，TorchScript v0.1.1。DINO无需xFormers，实际走CUDA原生attention；不是CPU回退。初装冲突与私有OpenCV修复日志均保留。

新机器可建Python3.10隔离环境，安装相同CUDA torch/torchvision及这些任务依赖，按官方来源取得固定源码/权重，再在唯一catalog的execution_json设置本机绝对路径、build并smoke。当前已验收的是124服务器，不能保证未经测试的新主机一键等价；freeze包含继承无关包，不能直接当最小requirements。

实际执行以execution_snapshot/manifest、每项record、真实argv/exit、环境freeze追溯，不伪造Git commit。最终catalog写入精度结论后与运行时SHA不同，原快照保留。旧数据/权重只读，不修改原冻结实验。

## 文件用途

discover/assets_research/download_*：来源查证与下载收据；prepare_pilot/prepare_acceptance：冻结图像输入，GT映射单独EVAL_ONLY；register：唯一主表一次性登记（已执行勿重复）；stage/launch_smoke/launch_pilot：实际执行；retrieve/evaluate/evaluate_control：取回与离线评分；make_gallery/write_report：展示；try_tool/search_features：研究者直接测试；finalize_catalog：把实际状态写回主表。

没有通过验收的候选：Hibou受官方门槛；Kaiko非商业许可且权重下载失败，不算可用工具。Phikon已有调研不算新增。当前工具是否值得进入Agent由研究者后续确定，本轮只证明工具功能及限定数据上的表现。
'''
 (HERE/'README.md').write_text(protocol,encoding='utf8')
 save(HERE/'cost_summary.json',dict(first5_process_wall_seconds=wall,first5_formal_outputs=149,fixed_FOV_outputs=77,formal_total=226,paid_API_calls=0,training=False,assistant_billing_not_observable=True))
 print('Report/protocol and weight identities written')

if __name__=='__main__':main()
