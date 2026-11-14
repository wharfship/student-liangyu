# student-liangyu

面向强分割模型残余错误的、由证据驱动的动态肾小球分割Agent研究。

**当前为2026-11-14本地整理快照，尚未上传。** SegFormer-B5为强基线；首先核实硬化球漏检、范围错误与组织混淆，证明补救工具有用，再比较专用方案、固定规则和动态Agent。现有新工具尚无硬化净收益，不能据工程跑通宣称Agent有效。

## 环境与最小运行

建议Python3.10。元数据和合同测试不需要GPU：

```bash
python -m venv .venv
# Linux: source .venv/bin/activate；Windows: .venv/Scripts/Activate.ps1
python -m pip install -r requirements-core.txt
python -m glomerulus_tools.build
python -m glomerulus_tools list
python -m pytest glomerulus_tools/tests -q
```

以上验证登记、输入边界和代码接口，**不进行模型分割**。推理需另建匹配CUDA环境、取得任务权重和合法数据，并按[资产与环境说明](docs/DATA_AND_ASSETS.md)设置路径；原服务器绝对路径不是通用安装路径。

```bash
python scripts/configure_assets.py --overrides configs/execution_paths.local.json
python -m glomerulus_tools check --tool nepath_segformer_b5 --task configs/my_patch_NO_GT.json --research
# 实时检查空闲GPU，再设置CUDA_VISIBLE_DEVICES；Linux示例：
CUDA_VISIBLE_DEVICES=0 python -m glomerulus_tools run --tool nepath_segformer_b5 --task configs/my_patch_NO_GT.json --output outputs/new_run --research
```

示例MPP须替换为实际输入的元数据/标定证据。Windows用`$env:CUDA_VISIBLE_DEVICES="0"`。研究调用不自动授权Agent采用结果；默认禁止CPU回退。

## 内容与状态

- `glomerulus_tools/`：统一入口、适配器、唯一catalog及生成表、37项测试；工具信息均从这里维护。
- `research/nepathtk_tools_20261105/`：冻结baseline包装器与配置；旧research为实验档案，不在此另建工具表。
- `glomerulus_tools/experiments/`：本轮完整源脚本与必要旧评价依赖。运行前显式修改数据路径；历史批次文件未外发，不能免数据直接复算。
- `configs/`、`environments/`：配置示例与实际环境冻结收据；`scripts/`只调整机器资产路径，不改变配方身份。
- `results/`：分组结果、协议、验证和源码SHA；不含病例逐行结果或医院图册。
- `provenance/`：来源commit、下载校验与导出文件清单；`third_party/`：许可说明。

32个enabled记录不等于本仓库附带32个可用网络。首批资产整理覆盖SegFormer、MESCnn、SAMRefiner和RITM；其他历史适配器可能依赖尚未移植的旧ops模块。缺失会拒绝执行，不能把可执行状态等同于精度。

[当前结果与负结果](docs/RESULTS.md) · [下一阶段实验](docs/NEXT_EXPERIMENTS.md) · [数据/权重放置](docs/DATA_AND_ASSETS.md) · [版本对应与局限](docs/REPRODUCIBILITY.md)

尚未完成：完整可移植模型环境自动安装、医院数据外发授权、全部旧Agent依赖打包、可靠专用补救动作、无GT验收、动态观察消融和独立患者留出测试。VISTA仅作为主动观察设计参考，未宣称复现。原创代码公开许可待确定，第三方许可各自适用。
