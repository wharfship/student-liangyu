# 新工具运行协议与实验记录

本轮用户要求继续调研任意可用、好用的开源能力，实际下载、跑通、配置可直接测试的协议。没有上传、飞书或付费API调用。工具方法信息仍统一在glomerulus_tools/README.md，唯一人工主表catalog.csv。

内部报告：[脱敏结果](../../../docs/OPEN_TOOLS_20261115.md)；113组图册仅在授权内部目录。

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

注意：本导出不包含医院样例、原机器受保护的服务器手册或模型权重；一键远程客户端目前在原授权工作区验收。外部机器须先配置其自身服务器手册、路径和授权数据，不能从此导出免数据直接跑医院样例。
