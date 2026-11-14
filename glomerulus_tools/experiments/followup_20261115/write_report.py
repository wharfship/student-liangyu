"""Build an internal review report from frozen metrics, without inference."""
from common import *


def main():
 ss=read(HERE/'paired_summary.json');rr=rows(REPORT/'金字塔配对逐目标_EVAL_ONLY.csv')
 full=[s for s in ss if s['stain']=='ALL'];table=[]
 for mode in ['PNG','JPG']:
  a=next(s for s in full if s['mode']==mode and s['cohort']=='sclerosis')
  b=next(s for s in full if s['mode']==mode and s['cohort']=='normal_control')
  table.append(f"|真实金字塔{mode}|{a['new']:.6f}|{a['delta_vs_baseline']:+.6f}|{a['improved001']}/{a['worsened001']}|{b['new']:.6f}|{b['delta_vs_baseline']:+.6f}|{b['improved001']}/{b['worsened001']}|")
 runs=read(HERE/'pyramid_runs.json');wall=sum(r['end']-r['start'] for r in runs)
 numeric=[]
 for group in ['sclerosis','normal_control']:
  for mode in ['PNG','JPG']:
   subset=[r for r in rr if r['cohort']==group and r['mode']==mode]
   numeric.append(dict(cohort=group,mode=mode,mean_JPEG_vs_PNG=sum(float(r.get('delta_JPEG_vs_PNG') or 0) for r in subset)/len(subset) if mode=='JPG' else None,
     inference_seconds=sum(float(r['seconds']) for r in subset)))
 save(HERE/'paired_cost_and_compression.json',dict(process_wall_seconds=wall,rows=numeric,paid_API_requests=0))
 report=f'''# 11月15日专项后续：供明天判断的结果

本轮完成了两套输入配方的160次真实推理、旧工具同目标对照、40例人工复核材料，以及77例无GT观察资产和离线决策评分器。**目前仍没有可替代SegFormer的硬化专用方案，也没有证明动态Agent优于固定流程。** 最值得保留的线索是旧SegNeXt能救回一个完全漏球，但其独有候选的误检负担很重。

这份报告是内部研究材料；病例图册、逐目标表和人工工作表均不进入公开仓库。无需先读历史对话。原始图像、GT、已有预测均保留。

## 先看这几处

1. [三例完全遗漏与旧工具](三例遗漏与旧工具.html)：先看S040，再看共同遗漏S019/S033。
2. [全部40例人工复核](人工复核.html)：按原SegFormer Dice升序，原生局部、当前预测、GT和FP/FN并排。[待填写工作表](硬化40_人工复核工作表_EVAL_ONLY.csv)已留出组织混淆、标注和病理确认字段，未代填诊断。
3. [MESCnn配对全80例](金字塔配对.html)：完整展示成功与失败；可筛硬化、普通和病例编号。
4. [77例无GT观察样例](无GT观察样例.html)：可检查不同观察是否提供了有用信息。本轮没有让VLM判定，没有Agent准确率。
5. [本地student-liangyu](../../../artifacts/student-liangyu/README.md)：整理副本，未上传、未发飞书。

## 1. 输入链是否解释MESCnn的落差？本轮不支持

在原40硬化＋40普通保护目标上，保持同一权重、4096原生像素参考窗、阈值0.5、不加面积过滤。新增实际OpenSlide最近10×金字塔层的PNG配方，再以OpenCV默认JPEG95编码检验作者的JPEG输入链。80/80通过作者10%倍率容差；未用GT选推理实例或修改参数。

|配方|硬化目标Dice|相对SegFormer|硬化改善/恶化>0.01|普通目标Dice|相对SegFormer|普通改善/恶化>0.01|
|---|---:|---:|---:|---:|---:|---:|
|SegFormer保留|0.842106|0|—|0.914157|0|—|
|原page0 AREA→10× MESCnn|0.676153|−0.165953|6/20|0.740630|−0.173528|7/16|
{chr(10).join(table)}

PNG与JPEG相对旧MESCnn的硬化均值分别只变动+0.000738和+0.000698；按来源号组自助法的配对区间均跨0。JPEG相对PNG的硬化均值差−0.00004050、普通+0.00001786。两套新配方都没有救回S019/S033/S040，硬化IoU≥0.5匹配仍为30/40，SegFormer为37/40；普通为31/40，SegFormer为40/40。以新结果全面替换会分别损失7个和9个已有匹配。

这次控制实验排除了“此次最近10×层/JPEG适配即可挽回明显精度落差”的解释。**它没有复现作者完整WSI流程**：没有整图tiling、重叠拼接与NMS。实际金字塔下采样比略偏离整数，按作者舍入规则取1024/2048边长，再映回固定4096窗，仍有亚像素/取整差异。结论限于登记的参考窗配方，不能写成整个MESCnn方法无效。

主指标为原GT目标对应的离线最佳IoU预测实例Dice，非在线实例选择；参考窗里整窗Dice、TP/FP变化已附逐目标表，重叠窗不可相加成WSI成绩。三个无预测目标的参考窗由GT定中心，**只检验有无补救能力，不证明部署时找得到它们**。

分层区间见实验`paired_summary.json`；[160行逐目标评分](金字塔配对逐目标_EVAL_ONLY.csv)。HE35/PAS5，PAS仅5个目标，不能作可靠染色泛化结论。

## 2. 旧工具是否能救这40个目标？SegNeXt有一个真实线索

按原任务ID和实例ID复用冻结全图推理，不重新切图跑模型。硬化40例中，SegNeXt匹配36，救回SegFormer漏掉的1例，同时丢掉SegFormer已匹配的2例；普通40由40降为38。Glo19/40、Omni CAP2/40、Omni unanimity1/40，均未救回三个完全遗漏。KI/Tokyo只支持这里5个PAS目标，不混作40例覆盖。

**S040：**原SegFormer与目标GT零交集；旧SegNeXt覆盖0.816088，最佳实例IoU0.796924。该预测来自原始全图图像推理，并非本轮GT定位后新跑。图册仅为离线观察选用了GT中心窗口。它说明当前兼容工具中至少有一处可补救的真遗漏，但一例不能建立稳定收益。

**代价：**旧医院40图SegNeXt独有候选共65个，仅1个匹配GT，64个未匹配；未匹配候选仍可能受漏标影响，未经复核不能全当病理确定的非球。S040所在切片两个独有候选中有一个真匹配。若直接把全部独有候选加入，会引入很大FP风险。S019与S033仍未由这些可兼容工具形成匹配，不能靠“换模型”自动解决。

这条线索可用来研究**拒绝大多数无效候选，同时保留真实补漏**。目前只能叫候选来源，不能称为硬化专网或已验证调度策略。[同目标旧结果](旧工具同40硬化与保护对照_EVAL_ONLY.csv)。

## 3. Skinet开源工具核查：任务资产齐了，平台尚不能执行

已从作者v1.0取得皮质和MEST主网络两个完整SavedModel，均含变量和label_map；主网络3=nsg、4=pac（全局硬化类）。此前缺权重的问题已推进。获取地址、字节数和本地SHA在`asset_receipts.json`；发布页字节数一致，**未找到作者另行公布的哈希**，本地哈希用于后续身份核对。

独立安装TensorFlow2.15.1成功，导入立即SIGILL（退出−4），在导入原生模块时尚未加载模型。服务器虚拟机没有暴露AVX/AVX2/FMA，关闭oneDNN仍失败；证据支持CPU指令集兼容性阻塞。没有切到CPU运行，也没有改其他环境。须换可暴露必要CPU指令集的兼容GPU环境，或由管理员处理虚拟机配置，之后先验证GPU和模型输出再谈精度。**本轮没有Skinet分割结果，不能记为模型精度差。**

源码确认该方法包含皮质门控、主网络切块与后处理；未跳过链路声称复现。训练MPP、精确染色与成员仍未知，研究与Agent调用保持禁用。Jiang作者Drive权重链接单次20秒超时，未取得checkpoint；不据此断言项目不可用。MORI按既定安排未运行。

## 4. 主动观察：繁琐的资产和验收已备好，实证尚未完成

复用已有RITM提案，对77个已定位目标生成539张具名JPEG：局部原图、预测支持、周边上下文、原生中心和当前/提案轮廓差异。显示ID打乱为V编号，GT和分组映射单独留在EVAL_ONLY文件；渲染不读GT。**只有图像字节和允许的描述能发送给模型，内部路径不能发送**，路径会暴露来源编号。

评分器只接受`keep/accept/manual_review`、有限观察请求和理由，强制一例一条、观察上限、合法视图，不接收GT字段。`keep`严格复现零变化；`accept`严格复现旧77例RITM结果。人工复核决定保留原掩膜，另计复核量。目标变化、正确接受、有害接受和观察次数均可导出；这只是评分接口验收。

当前RITM硬化提案只有2例改善>0.01、4例恶化>0.01，信息量有限；不能为了证明观察有效而无选择地付费调用。该77例来自GT知情且已暴露的开发队列，不含3例完全漏球。当前没有合法的无GT错误发现实验、固定观察/动态观察对照或新Agent闭环成绩。

VISTA尚未核实到具体原文。两次免费元数据查询遭HTTP429，本地按文件名未定位到对应论文，均不代表论文不存在。本轮采用的是导师提出的“按疑问补充原图、上下文和差异证据”设计，没有冒称复现某篇VISTA。

## 5. 明天建议如何判断

**优先做病例确认，再决定补救动作。** 建议先看S019/S033/S040及S039等低分例，确认组织、目标边界、扫描质量和邻近血管/纤维组织混淆。工作表的病理问题待研究者/医生填写；本轮没有凭图像编造病理结论。当前已检出硬化37例平均Dice0.910385，接近已检出普通对照0.914157；全40均值主要受完全遗漏影响，但普通对照选取已检出目标，不能据此推断两类整体性能相同。

|优先级|下一步|通过条件与停止条件|
|---|---|---|
|P0|复核3个完全遗漏和低分例，区分真失败与标注/图像问题|确认清单与可评价边界；保留原结果，不擅改GT|
|P1|复核SegNeXt65个独有候选，测试现成专业身份工具的可保留/可拒绝性|同候选报告真补漏、未匹配FP、普通球误伤；一例真补漏只作开发证据，不能独立证明Agent|
|P1|在兼容GPU环境验收Skinet完整链|先导入/GPU/加载/尺度/坐标smoke，再固定配方做40+40；未通过smoke不扩批|
|P2|若现成工具仍无足够收益，考虑硬化定位/分割适配或取得可靠独立纠错点|必须建立患者分组训练/开发/测试清单，先验证动作本身；禁止训练后再回当前40例当盲测|
|P3|有效动作成立后做观察与调度对照|固定观察、固定多视野、动态按疑问观察，在相同动作和图像/Token预算下比较；先冻结协议|

最后完整系统比较仍为A SegFormer、B有效专用方案、C强固定规则、D同预算Agent。规则和Agent使用同一候选池、补救动作、兼容性检查与计算上限。测试按患者/来源隔离，当前80例全部视作开发暴露；每次报告全图Dice、实例一对一检出、FP、普通球误伤、复核量、GPU/墙时、工具调用和VLM成本。GT知情最佳可达分析与实际无GT策略分开，不把局部窗均值写成全图成绩。

**本轮决定建议：**保留SegFormer强基线；不全面换MESCnn、不扩大重复通用修补器实验；将SegNeXt单例互补与Skinet环境验收作为两条有依据的后续线索。论文主张暂定为待验证的研究问题，不能提前写成“Agent已解决硬化分割”。

## 6. 完成情况、成本与复现

- 新MESCnn输入链160/160完成，两个进程exit0；进程墙时合计{wall:.2f}秒（含加载/输出，不含之前裁图和环境安装）。裁图78.43秒，观察资产125.17秒；Skinet下载和依赖安装另有日志，未并入网络推理成本。
- 40例复核160张JPEG、80例输入配对320张JPEG、三例遗漏6张JPEG和539张具名观察JPEG已生成；均保留完整队列。
- 统一入口37项测试及本轮HTML/图片/SHA/决策合同验收见实验收据；工程检查不证明临床准确性。
- 付费VLM/API请求0次，无训练。本轮不承诺可读取助手平台实际账单；未另外消耗所提供的Qwen付费API。用户20元上限不被解释为需要花满。
- 代码在`glomerulus_tools/experiments/followup_20261115/`。实际执行快照、主表SHA、输入/权重SHA、环境、命令、exit code、输出record和评价脚本可追溯。新本地Git提交只代表整理版本，不冒称历史运行commit。
- 主表新增的是同一MESCnn网络的两个**输入配方**，不是两个新网络；Agent开放池和默认SegFormer未改变。无上传、无飞书发送，无写入知识库或组会文档。

本报告只读复用了原实验，新增补跑保留在新目录。人工复核尚未完成、Skinet环境阻塞和动态Agent未验证均明确留作待解决项。
'''
 (REPORT/'结果汇报.md').write_text(report,encoding='utf8')
 (REPORT/'README.md').write_text('# 本轮审阅入口\n\n先看[结果汇报](结果汇报.md)，再看[三例遗漏](三例遗漏与旧工具.html)、[40例人工复核](人工复核.html)与[80例输入配对](金字塔配对.html)。[无GT观察资产](无GT观察样例.html)尚无Agent结果。全部为内部证据，未上传。\n',encoding='utf8')
 print('Report written; process wall seconds',wall)


if __name__=='__main__':main()
