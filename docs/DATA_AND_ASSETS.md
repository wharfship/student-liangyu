# 数据、权重与环境

医院原始WSI、实例GT、病例映射和预测不提供公开下载，须按机构授权取得。放置于本机`data/`或外部只读目录；不要提交Git。原GT仅离线评分，推理任务不允许GT字段。VLM仅发送JPEG副本，原图与掩膜保留原格式。

以下SHA是实际已测试的身份。网络/训练划分版本不同不能用同一成绩：

|工具|获取入口与放置|SHA256|
|---|---|---|
|SegFormer-B5|https://huggingface.co/NePathTK/tma-segmentation （作者 https://github.com/Nicolik/NePathTK ）；将对应任务checkpoint放`research/nepathtk_tools_20261105/weights/segformer_b5.pth`，包装器按models.json校验|1fb44517a8d0136d97a71c0ea76bd6ab9db6597444e8b2e4da568d9cd705e53b|
|MESCnn internal-validation Cascade Mask R50|https://huggingface.co/MESCnn/MESCnn/blob/a38cc99aa5fea432743fdeaf5f3f4113b2057aa4/detection/logs/cascade_mask_rcnn_R_50_FPN_1x/internal-validation/output/model_final.pth ；放`assets/mescnn_cascade_r50_internal_model_final.pth`|007181ac33b5a4a0e6f13d5d6dfef9ee06a45fa0e8d33494ff7c885f98d812d4|
|SAM ViT-B|https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth ；放`assets/sam_vit_b_01ec64.pth`|ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912|
|RITM HRNet32|https://github.com/supervisely-ecosystem/ritm-interactive-segmentation/releases/download/v0.1/coco_lvis_h32_itermask.pth ；放`assets/coco_lvis_h32_itermask.pth`|cc4d55e44c23c6fbed7f2ed3d9dc1fb7f4f08d8cd1642d4f38824c89f3dc44f1|

MESCnn源码固定 https://github.com/Nicolik/MESCnn/tree/284e83781ae8e83a1362d896c2ea11fa95f82a9e ，Detectron2固定247691a1c292654867deb35cabb27f18d75a1363。作者HF直连曾失败，镜像文件与官方LFS的字节/哈希一致，收据在provenance。实际环境Python3.10、torch2.5.1+cu121/torchvision0.20.1，Detectron2 C++以FORCE_CUDA=0 MAX_JOBS=2隔离构建，模型仍在GPU执行。复建参考本轮setup_specialist_remote.py和environments/mescnn_historical_freeze.txt；该脚本内有历史服务器路径，不能在别的机器无修改照跑。

SegFormer使用mmseg/mmcv/mmengine依赖，按environments/segformer_historical_freeze.txt匹配；配置含初始化下载URL，但包装器清空backbone.init_cfg并严格加载任务权重，无需另下载预训练骨干。输入RGB原0..255按配置归一化，源MPP→1.0μm/px工作格、1024窗/512步长；工作格不是已核实训练MPP。

SAMRefiner使用 https://github.com/linyq2117/SAMRefiner/tree/4bb7f95738c7a2e805f189210fb09e02a2197557 中修改版segment-anything、FastGeodis及依赖。需保留extra_paths中的作者源码、segment-anything和memory_overlay/utils.py，不能换成另一份同名SAM包。已随仓库提供overlay及哈希/许可，其他依赖从作者来源复建。

RITM使用provenance/ritm_source_receipt.json内固定源码文件SHA。原归档入口为master而非固定Git commit，所以应按文件SHA核对，不能声称移动分支天然可重现。

复制configs/execution_paths.example.json为本地配置，填实际绝对路径，使用scripts/configure_assets.py写回catalog.execution_json并build。SAM/RITM还需配置其python/source_dir/checkpoint/extra_paths和RITM source_receipt。该助手不会改checkpoint_sha256、网络配方或Agent开放状态。模型未完整移植前`list`中的missing_assets为真实限制。

公开NEPTUNE/HuBMAP等历史数据来源以catalog evidence追溯，本快照未附历史数据清单，不声称可直接复算其分数。训练成员和训练/测试重叠未知处保持未知。


Skinet完整皮质与MEST主网络SavedModel来自 https://github.com/SkinetTeam/Skinet-MEST-C/releases/tag/v1.0 ，分别下载skinet_cortex.zip和skinet_mest_main.zip，放assets/并按作者方式解压（包含saved_model.pb/variables/label_map.pbtxt）。下载身份SHA在provenance/skinet_asset_receipts.json；发布页字节数核对一致，无作者另行哈希证据。此时完整权重已取得，不需重复下载。隔离TF2.15.1安装成功但服务器VM未暴露AVX，导入SIGILL，尚无模型结果；必须在兼容GPU平台先验收，不能把安装freeze当已验证可移植环境。训练MPP/染色/成员未知。
