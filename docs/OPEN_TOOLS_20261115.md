# 新开源能力：真实下载、CUDA推理与限定验收

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
