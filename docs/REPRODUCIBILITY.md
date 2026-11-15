# 版本对应与复现边界

原工作区和远程工程没有Git历史。本地新仓库的初始提交仅代表整理快照，**不能写成历史运行使用该提交**。

本轮每组结果通过以下信息对应到配方与代码：catalog tool_id/权重SHA、protocol.json、固定上游commit/文件SHA、实际执行snapshot、运行命令/环境freeze、输出record、离线评价脚本。外发分组结果由本次内部产物只读导出；results/final_source_manifest.json保留整理后源SHA，provenance/export_manifest.json对应此次导出文件。

实际执行snapshot与整理后源码清单不同含义：前者证明运行用了什么；后者证明交付包含什么。内部执行路径/任务记录仍在原实验目录，没有杜撰过去的commit号。未来运行应记录当前`git rev-parse HEAD`、dirty diff/源SHA、配置SHA、任务manifest SHA、checkpoint SHA、环境和输出位置。

元数据与37项测试可直接复跑。完整模型复算仍需取得授权数据、复建隔离环境和路径；准备脚本中的历史路径尚未全部参数化。新机器GPU精度尚未独立验收，不能宣称一键复现医院结果。

包中catalog是唯一源表的生成冻结副本，原字段未手工再整理为另一个主表。用户确定后续主工作仓库后，只保留一处人工维护位置。本地configure_assets只改变执行机器路径，配方变化应另登记并验证。

医院图片、病例号、GT/预测掩膜、凭据及AGENTS.md均未导出。Git ignore是补充措施，不能替代导出白名单和内容检查。没有执行远程push、建站或飞书发送。


本轮执行快照SHA与远程21文件一致；最终主表评分状态更新后SHA另记，不能替代实际运行快照，见[11月15日后续](FOLLOWUP_20261115.md)。

新增工具的结果、模型取得、物理尺度与执行边界见[新工具验收](OPEN_TOOLS_20261115.md)。
