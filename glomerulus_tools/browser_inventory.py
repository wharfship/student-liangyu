"""Render the existing gallery's tool inventory from the sole editable catalog.

Does not run tools or alter their authorization. Historical experiment rows and
all other gallery tabs remain intact. Works with file:// without fetch().
"""
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
GALLERY = ROOT / '已经整理好的内容/医院分割结果分析/尝试过的方案/20261112_RITM新方法实测/gallery.html'
TEMPLATE = ROOT / 'glomerulus_tools/experiments/all_methods_review_20261112/page.html'
OUTPUT = ROOT / 'glomerulus_tools/experiments/browser_inventory_20261115'
GROUPS = {
    'segmentation': ('肾小球定位与完整分割', 'SegFormer基线、专业分割网络、冻结配方与硬化候选。'),
    'repair': ('局部轮廓修复', 'SAM系、CascadePSP、交互纠错及尚缺权重的修复方法。'),
    'identity': ('对象真假与病变证据', '分类器、检测器、病变头；不把类别／框当分割掩膜。'),
    'features': ('细胞核与图像检索', '核实例、形态测量、通用表征；无任务头不等于会诊断。'),
    'quality': ('图像质控与病理预处理', '质量、组织背景、染色处理与切块工作流。'),
    'observation': ('观察与Agent工作流', '候选裁图、视觉取证和已有闭环；工程与准确性分别验收。'),
    'mouse': ('鼠肾与其他域', '与人肾分开记录，不能混用队列分母。'),
}
STATES = {
    'research': '研究入口启用', 'selected': '已选待部署', 'smoke': '单图跑通·仍禁用',
    'retired': '已停用／弃用', 'candidate': '候选／受阻', 'reference': '冻结配方／内部组件',
}
READINESS = {'ran':'已取得并批量测试', 'smoke':'已取得·单图跑通',
             'prepared':'源码已备·待跑图', 'blocked':'获取／运行受阻',
             'loaded':'仅加载·未跑真图', 'research':'仅调研·未测试', 'api':'已做真实API测试'}
AUDITS = {'scoped':'本次运行检查完成', 'partial':'部分检查完成', 'pending':'尚未完成检查'}


def esc(text):
    return html.escape(str(text), quote=True)


def href(path):
    target = ROOT / path
    try:
        return Path(os.path.relpath(target, GALLERY.parent)).as_posix()
    except ValueError:  # Existing historical evidence can live on D:.
        return target.as_uri()


def link(path, label):
    return f'<a href="{esc(href(path))}" target="_blank" rel="noopener">{esc(label)}</a>'


def group(row):
    if row.get('display_review'):
        return json.loads(row['display_review'])['purpose']
    tool = row['tool_id']
    adapter = json.loads(row['execution_json']).get('adapter')
    if tool in {'capybara_kpis', 'likiseg333_light'}:
        return 'mouse'
    if tool.startswith(('grandqc_', 'histoqc_', 'tiatoolbox_', 'deepflash2_')):
        return 'quality'
    if tool.startswith(('candidate_crop', 'visual_reader')):
        return 'observation'
    if tool.startswith(('instanseg_', 'dinov2_', 'hovernet_')) or 'backbone' in tool or tool in {'conch_identity_candidate', 'musk_identity_candidate'}:
        return 'features'
    if adapter == 'mask_refinement' or tool.startswith(('gloreflex_', 'samrefiner_', 'segrefiner_', 'crm_')):
        return 'repair'
    if adapter in {'candidate_classifier', 'object_evidence'} or tool in {'glo_gs_head_component', 'glo_circlenet5_component', 'gncnn_disease_classifier', 'mescnn_lesion_classifier', 'juang2024_classifier_candidate'}:
        return 'identity'
    return 'segmentation'


def state(row):
    status = row['status']
    if any(word in status for word in ['retired_by_user', 'abandoned_by_user', 'excluded_by_user', 'paused_by_user']):
        return 'retired'
    if status == 'selected_by_user_pending_deployment_no_agent':
        return 'selected'
    if status == 'local_cpu_smoke_passed_accuracy_unvalidated':
        return 'smoke'
    if row['enabled'] == 'yes':
        return 'research'
    if status in {'component_not_standalone_tool', 'frozen_recipe_reference_only'}:
        return 'reference'
    return 'candidate'


def evidence_links(row):
    result = [link('glomerulus_tools/README.md', '统一说明／调用协议')]
    seen = set()
    for token in re.split(r'\s*;\s*|\s*；\s*', row['evidence']):
        token = token.strip()
        if token.startswith('glomerulus_tools/'):
            path = ROOT / token
        elif token.startswith(('experiments/', 'literature/')):
            path = ROOT / 'glomerulus_tools' / token
        else:
            path = ROOT / token
        if path.is_file() and path not in seen:
            seen.add(path)
            result.append(link(path, path.name))
    for url in re.findall(r'https?://[^\s"\],;；]+', row['evidence']):
        if url not in seen:
            seen.add(url)
            result.append(f'<a href="{esc(url)}" target="_blank" rel="noopener">{esc(url)}</a>')
    return ' · '.join(result)


def tool_row(row):
    bucket, purpose = state(row), group(row)
    card = json.loads(row['execution_json'])
    operations = card.get('planned_operations', {})
    if isinstance(operations, list):
        operations = [x.get('name', x.get('operation', '')) if isinstance(x, dict) else str(x) for x in operations]
    operation_text = '；预备功能：' + ' / '.join(operations) if operations else ''
    default = ' · 人肾默认' if row['default_for'] == 'human' else ''
    note = row['researcher_note']
    details = f'<p><strong>登记状态：</strong>{esc(row["status"])}</p>' + ''.join(f'<p><strong>{esc(label)}：</strong>{esc(row[key])}</p>' for key, label in [
        ('verification', '实际验收'), ('results', '已记录结果／原队列分母'),
        ('training_sources', '训练来源'), ('test_overlap', '重叠风险'),
        ('scale', '尺度'), ('limitations', '限制'), ('code_license', '源码许可'), ('weight_license', '权重许可')])
    if note:
        details += f'<p><strong>研究备注（含历史决定，以最新状态为准）：</strong>{esc(note)}</p>'
    summary = row['results']
    # Keep long original results in the expandable receipt; don't truncate the only copy.
    short = summary if len(summary) <= 145 else summary[:145] + '…'
    search = ' '.join([row['tool_id'], row['name'], row['family'], row['target'], row['status'], summary, note, operation_text]).lower()
    return f'''<tr class="inventory-recipe" data-tool-id="{esc(row['tool_id'])}">
<td><strong>{esc(row['name'])}</strong>{esc(default)}<br><code>{esc(row['tool_id'])}</code><br><small>{esc(row['species'].replace('|', ' / '))} · {esc(row['stains'].replace('|', ' / '))}</small></td>
<td><span class="inventory-badge state-{bucket}">{STATES[bucket]}</span><br>研究入口：{'启用' if row['enabled']=='yes' else '禁用'}<br>Agent：{'已开放' if row['agent_enabled']=='yes' else '未开放'}</td>
<td>{esc(row['target'])}{esc(operation_text)}<p class="inventory-summary">{esc(short)}</p><details><summary>展开结果、限制与证据</summary>{details}<p>{evidence_links(row)}</p></details></td></tr>'''


def families(rows):
    result = []
    seen = set()
    for row in rows:
        if not row.get('display_name'):
            continue
        family = row['display_family']
        assert family not in seen, family
        seen.add(family)
        members = [r for r in rows if r.get('display_family') == family]
        review = json.loads(row['display_review'])
        assert review['readiness'] in READINESS and review['audit'] in AUDITS
        result.append(dict(row=row, members=members, review=review))
    assert sum(len(f['members']) for f in result) == len(rows), 'Every catalog ID needs exactly one display family'
    result.sort(key=lambda f: (f['row'].get('default_for')!='human',
                               all(state(r)=='retired' for r in f['members'])))
    return result


def family_card(family):
    row, members, review = family['row'], family['members'], family['review']
    research = any(r['enabled']=='yes' for r in members)
    agent = any(r['agent_enabled']=='yes' for r in members)
    retired = all(state(r)=='retired' for r in members)
    selected = any(state(r)=='selected' for r in members)
    visibility = 'focus' if review['readiness'] in {'ran','smoke','api'} or selected else 'candidate'
    search = ' '.join([row['display_name'],row['display_conclusion']]+[r['name']+' '+r['tool_id']+' '+r['results'] for r in members]).lower()
    status = '已停用' if retired else '已选待用' if selected else '当前保留'
    label = f'<span class="inventory-badge state-{"retired" if retired else "selected" if selected else "reference"}">{status}</span>'
    top_links = evidence_links(row)
    primary = f'<span class="inventory-badge state-selected">{READINESS[review["readiness"]]}</span><span class="inventory-badge state-reference">{AUDITS[review["audit"]]}</span>'
    return f'''<article class="inventory-card" data-family="{esc(row['display_family'])}" data-purpose="{review['purpose']}" data-scope="{visibility}" data-readiness="{review['readiness']}" data-audit="{review['audit']}" data-research="{'yes' if research else 'no'}" data-agent="{'yes' if agent else 'no'}" data-retired="{'yes' if retired else 'no'}" data-search="{esc(search)}">
<div class="inventory-card-title"><h3>{esc(row['display_name'])}</h3>{label}</div><p class="inventory-verdict">{esc(row['display_conclusion'])}</p>
<div class="inventory-badges">{primary}</div><p class="inventory-access">研究入口：<strong>{'可调用' if research else '未启用'}</strong>　Agent：<strong>{'已开放部分能力' if agent else '未开放'}</strong></p>
<details class="inventory-card-detail"><summary>展开结果、测试范围与配置</summary><p class="mini">“运行检查完成”仅表示本次任务的输入／输出与执行记录已核对，不等于完整系统、独立精度或所有功能已验收。获取与测试也不替代许可，原许可限制见各配置详情。</p><p>{top_links}</p><p>该工具包含{len(members)}项已登记配方／组件，研究与Agent开关以具体项为准：</p><div class="table-wrap"><table><thead><tr><th>具体配方</th><th>执行状态</th><th>结果、来源与许可</th></tr></thead><tbody>{''.join(tool_row(r) for r in members)}</tbody></table></div></details></article>'''


def recent_records():
    """Dated experiment reading notes, not another tool status table."""
    hospital = '已经整理好的内容/医院分割结果分析/'
    tools = 'glomerulus_tools/experiments/'
    return [
        ('新辅助工具：核、检索、质控',
         'InstanSeg、DINOv2、GrandQC三家族四入口已真实CUDA推理226项。核验收TNBC8 Dice0.8572，但属于已知训练来源；DINO原裁片73/77、固定224μm视野63/77是检索结果，不是硬化诊断准确率；QC人工模糊有响应，真实伪影精度未验。GrandQC含非商业许可。',
         [(hospital+'开源工具扩展_20261115/结果报告.md', '实测报告'), (hospital+'开源工具扩展_20261115/gallery.html', '113组全部输出'), (tools+'open_tools_20261115/README.md', '源码、环境与复测命令')]),
        ('三个辅助工具能否互补／联合调用',
         '同224μm视野新增125项输出。DINO检索63/77、核形态42/77、固定融合64/77，改善7／损失6、区间跨0，未证明融合收益。强模糊使核总数1679→1、检索10/12→7/12；QC提示10/12。宽上下文补提示是事后线索，非Agent成绩。联合命令已跑通，分别返回质量、核测量、特征，未修改分割。',
         [(hospital+'工具权威性与互补_20261115/结果报告.md', '权威性与互补报告'), (hospital+'工具权威性与互补_20261115/gallery.html', '89组全部对照'), (tools+'tool_evidence_20261115/README.md', '联合命令与协议')]),
        ('HistoQC／TIAToolbox：已选入待使用，暂不给Agent',
         '两者有官方源码并已下载，许可分别BSD-3-Clause-Clear、BSD-3-Clause。HistoQC预备基本统计、颜色、模糊、整条QC；TIA预备组织mask、染色矩阵、归一化、切块，共8项函数协议。HistoQC尚未图像执行，TIA导入缺umap。describe可看参数，check只校验结构、exit2且executable=false；run拒绝，不能写成已跑通。',
         [(tools+'histoqc_tia_interfaces_20261115/README.md', '源码与8项协议验收'), ('glomerulus_tools/README.md', '统一功能说明')]),
        ('硬化专项与MESCnn后续：负结果也留档',
         '全40硬化＋40普通保护目标，SAMRefiner、RITM均无硬化净收益；MESCnn原输入80次、实际金字塔PNG／JPEG各80次仍未建立净改善，三例GT定位遗漏参考均未救回。77例JPEG观察资产已备，未运行该批付费VLM；VISTA仍只作主动观察参考，未称复现。',
         [(hospital+'硬化专项验证_20261114/结果报告.md', '40＋40专项'), (hospital+'专项后续_20261115/结果汇报.md', '金字塔配对与成本'), (hospital+'专项后续_20261115/三例遗漏与旧工具.html', '三例遗漏')]),
        ('最新工具去留决定',
         'RITM按用户取舍停用，优先SAM，不宣称已做全场景等价证明；MESCnn五项维持弃用；SegNeXt移出可用池、停止作为补漏方向。旧80目标配对SegNeXt曾救回1例／丢失2例，淘汰不抹去这一历史事实。上述历史图册继续可看。',
         [(tools+'tool_retirement_20261115/README.md', '决定与禁用验收'), ('docs/交接workflow的文件/2026-11-15_工具淘汰与Skinet期刊复审.md', '最新交接')]),
        ('Skinet：期刊核实、本机跑通、单图范围异常',
         '当前MEST-C为NDT2023，早期为CJASN2022，期刊值得保留。旧VM缺AVX、TF导入SIGILL；本机独立Python3.10／TF2.15.1已完成皮质和主网各1张1024窗CPU推理，约16／19秒（加载另计）。主网把大片组织判为非硬化球，范围明显过大；无正式精度结果，仍禁用，先核尺度与皮质裁切，不扩队列。',
         [(tools+'tool_retirement_20261115/README.md', '期刊、运行收据及命令'), (tools+'tool_retirement_20261115/local_smoke/skinet_mest_main/preview.jpg', '实际主网JPEG输出')]),
        ('输入与切块动作：C004／C003及GT修正',
         'C004平移网格恢复截断上半部，已有13窗诊断和概率回放，未证明自动找错／接受；C003最高层有白块、20×／10×层保留组织，已恢复替代输入但未重跑分割。C037／C038经用户确认修正GT偏移，当前指标已重评，原预测未重跑。这些不归为新独立分割网络。',
         [('glomerulus_tools/agent_capabilities/README.md', '动作代码与缺口'), (hospital+'尝试过的方案/20261112_RITM新方法实测/C003金字塔层异常与输入恢复.md', 'C003核查'), ('docs/交接workflow的文件/2026-11-15_C037_C038源GT修正完成.md', 'GT修正收据')]),
        ('导师目标、Agent证据与仓库交付边界',
         '主线仍为强SegFormer残余错误的证据驱动动态分割Agent；需先证明有效动作，再比较单工具、专用动作、固定路由与动态调度，观察普通球误伤和成本。旧视觉Agent／真实闭环已经试过，尚未证明优于同预算强规则。student-liangyu已有本地审阅版本；用户当前要求暂不上传，未发飞书，不把本地仓库写成已公开。',
         [('docs/交接workflow的文件/2026-11-13_导师任务方案_硬化肾小球与工具复用.md', '导师方案与四组比较'), ('docs/交接workflow的文件/2026-11-14_硬化专项验证与student-liangyu本地仓库交接.md', '本地仓库交接'), ('docs/交接workflow的文件/2026-11-11_已做尝试与工具复用_防重复交接.md', '旧Agent实验与完整历史')]),
        ('其他已查候选：没有结果的也注明停止点',
         'HoVer-Net已核MedIA2019论文与作者代码，尚无本地任务运行；QuPath是已有病理分析平台，不充作新分割网络。Hibou需账户授权，Kaiko公开checkpoint下载两次连接重置，均未取得可验收输出；Phikon早已调研，不重新记为新工具。SegRefiner／CRM／FocalClick缺已取得任务权重，未实跑。Saltfish已按用户决定移出主表，不绕过决定重新登记；详见旧交接。',
         [(hospital+'开源工具扩展_20261115/结果报告.md', '下载与未列入可用的候选'), (hospital+'工具权威性与互补_20261115/结果报告.md', '正式来源核查'), ('docs/交接workflow的文件/2026-11-11_已做尝试与工具复用_防重复交接.md', '旧候选与Saltfish记录')]),
    ]


def legacy_groups(old):
    """Keep the original 16 experiment rows verbatim, grouped by purpose."""
    table = old.split('<tbody>', 1)[1].split('</tbody>', 1)[0]
    rows = re.findall(r'<tr>.*?</tr>', table, flags=re.S)
    assert len(rows) == 16, len(rows)
    buckets = [
        ('完整分割、补漏与跨域', [0, 1, 15]),
        ('分类筛查、删FP与检测证据', [2, 3, 4, 5]),
        ('轮廓修复与几何操作', [6, 7, 8, 9, 10]),
        ('调度、取证、选择与预处理', [11, 12, 13, 14]),
    ]
    result = []
    for title, indexes in buckets:
        result.append(f'<details class="inventory-group"><summary>{title} · {len(indexes)}项历史尝试</summary><div class="table-wrap"><table><thead><tr><th>尝试方向</th><th>当时的实际结论（当前开关见上方清单）</th><th>证据入口</th></tr></thead><tbody>'+''.join(rows[i] for i in indexes)+'</tbody></table></div></details>')
    return '\n'.join(result)


CSS = '''
.inventory-toolbar{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:18px 0}.inventory-toolbar input{padding:8px 12px;min-width:220px;flex:1;border:1px solid #b9c5d4;border-radius:7px}.inventory-toolbar select{max-width:100%}.inventory-links{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0}.inventory-links a{padding:6px 10px;border:1px solid #dbe2e9;border-radius:6px;background:#f5f8fb;text-decoration:none}.inventory-group{background:white;border:1px solid #dbe2e9;border-radius:9px;margin:12px 0;padding:14px}.inventory-group>summary{font-size:17px}.inventory-group>p{color:#596775}.inventory-group table{table-layout:fixed}.inventory-group th:nth-child(1){width:29%}.inventory-group th:nth-child(2){width:23%}.inventory-group td{overflow-wrap:anywhere}.inventory-tool code{font-size:12px;word-break:break-all}.inventory-tool details{margin-top:8px}.inventory-tool details p{font-size:13px}.inventory-summary{color:#3f5062}.inventory-badge{font-size:13px;font-weight:600;padding:2px 7px;border-radius:4px;display:inline-block}.state-research{background:#e5f4ed;color:#126445}.state-retired{background:#f1e9ea;color:#8c3b49}.state-candidate,.state-reference{background:#eef0f4;color:#536171}.state-selected,.state-smoke{background:#fff0cc;color:#775316}.inventory-records{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px}.inventory-records article{margin:0;padding:15px}.inventory-records h3{margin-top:0}.inventory-records p{font-size:14px}.inventory-tool[hidden],.inventory-group[hidden]{display:none!important}#inventory-count{font-variant-numeric:tabular-nums}#inventory-empty{padding:20px;color:#596775}@media(max-width:700px){.inventory-records{grid-template-columns:1fr}.inventory-toolbar input{min-width:160px}.inventory-group{padding:10px}.inventory-group table{min-width:780px}}
'''
CSS += '''
.inventory-cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.inventory-card{margin:0;padding:18px;border:1px solid #d6e0ec;border-radius:10px;min-width:0}.inventory-card-title{display:flex;justify-content:space-between;align-items:center;gap:8px}.inventory-card-title h3{font-size:21px;margin:0}.inventory-verdict{font-size:16px;font-weight:600;line-height:1.7;margin:10px 0 13px}.inventory-badges{display:flex;flex-wrap:wrap;gap:6px}.inventory-access{font-size:13px;color:#596775;margin:9px 0}.inventory-card-detail{border-top:1px solid #e5eaf0;padding-top:10px}.inventory-card-detail summary{color:#145aa0;font-size:14px}.inventory-card-detail .table-wrap{margin-top:10px}.inventory-card table{min-width:730px;table-layout:fixed}.inventory-card th:nth-child(1){width:28%}.inventory-card th:nth-child(2){width:22%}.inventory-card code{overflow-wrap:anywhere;font-size:12px}.inventory-card p{overflow-wrap:anywhere}.inventory-card[hidden]{display:none!important}.inventory-toolbar{gap:10px}.inventory-filters{background:#f5f8fc;border-radius:8px;padding:10px 14px;margin:10px 0}.inventory-filters .inventory-toolbar{margin:8px 0}.inventory-note{font-size:13px;color:#596775}.inventory-group>summary{margin-bottom:12px}.inventory-group{border:0;padding:0;margin:24px 0}.inventory-scope-label{font-size:14px}@media(max-width:800px){.inventory-cards{grid-template-columns:1fr}.inventory-card{padding:14px}.inventory-verdict{font-size:15px}}
#inventory-tools{scroll-margin-top:65px}
'''

SCRIPT = '''
(()=>{const root=document.getElementById('tab-attempts');if(!root)return;
const search=document.getElementById('inventory-search');const fields=['purpose','scope','readiness','audit','research','agent','retired'];
const controls=fields.map(k=>[k,document.getElementById('inventory-'+k)]);
const rows=Array.from(root.querySelectorAll('.inventory-card')),groups=Array.from(root.querySelectorAll('[data-inventory-group]'));
function filter(){const q=search.value.trim().toLowerCase();let shown=0;for(const row of rows){const ok=(!q||row.dataset.search.includes(q))&&controls.every(([k,e])=>!e.value||row.dataset[k]===e.value);row.hidden=!ok;if(ok)shown++;}
for(const group of groups){const n=Array.from(group.querySelectorAll('.inventory-card')).filter(r=>!r.hidden).length;group.hidden=!n;group.querySelector('.inventory-visible-count').textContent=n;if(n)group.open=true;}
document.getElementById('inventory-count').textContent='当前显示 '+shown+' 个工具';document.getElementById('inventory-empty').hidden=shown!==0;}
for(const element of [search,...controls.map(([,e])=>e)])element.addEventListener('input',filter);
document.getElementById('inventory-reset').addEventListener('click',()=>{search.value='';for(const [k,e] of controls)e.value=k==='scope'?'focus':'';filter();});
for(const button of root.querySelectorAll('[data-inventory-expand]'))button.addEventListener('click',()=>{for(const g of groups)if(!g.hidden)g.open=button.dataset.inventoryExpand==='yes';});
filter();})();
'''


def render(rows, old):
    digest = hashlib.sha256((ROOT/'glomerulus_tools/catalog.csv').read_bytes()).hexdigest()
    tools = families(rows)
    recent = ''.join('<article><h3>'+esc(title)+'</h3><p>'+esc(text)+'</p><p>'+' · '.join(link(path, label) for path, label in links)+'</p></article>' for title, text, links in recent_records())
    def select(key, label, options, selected=''):
        return f'<label>{label} <select id="inventory-{key}">'+''.join(f'<option value="{esc(k)}" {"selected" if k==selected else ""}>{esc(v)}</option>' for k,v in options.items())+'</select></label>'
    intro = f'''<section class="tab" id="tab-attempts" hidden>
<div class="box"><h2>已经尝试工具</h2><p><strong>SegFormer继续做基线。</strong>RITM停用、MESCnn弃用、SegNeXt不再推进补漏；HistoQC与TIAToolbox已选待用。Skinet单图跑通但效果待核。</p><p class="inventory-note">每个工具先看效果结论；染色、提示方式和具体版本收在详情。获取与测试、运行检查、研究入口、Agent开放分别记录。</p>
<div class="inventory-links"><a href="#inventory-tools">看工具和结论</a><a href="#inventory-recent">最近实验补记</a><a href="#inventory-legacy">旧实验归档</a><a href="#inventory-history">早期Agent实验</a></div></div>
<div class="box" id="inventory-tools"><h2>工具与当前结论</h2>
<div class="inventory-toolbar"><input id="inventory-search" type="search" placeholder="搜索工具名或功能，如 SAM、染色、核…" aria-label="搜索工具与功能">{select('purpose','用途',{'':'全部用途',**{k:v[0] for k,v in GROUPS.items()}})}{select('scope','范围',{'focus':'已测试或已选用','':'包含仅调研候选'},'focus')}<button id="inventory-reset">重置</button></div>
<details class="inventory-filters"><summary>按测试、检查或开放状态筛选</summary><div class="inventory-toolbar">{select('readiness','获取与测试',{'':'全部',**READINESS})}{select('audit','运行检查',{'':'全部',**AUDITS})}{select('research','研究入口',{'':'全部','yes':'可调用','no':'未启用'})}{select('agent','Agent',{'':'全部','yes':'已开放','no':'未开放'})}{select('retired','去留',{'':'全部','yes':'已停用','no':'当前保留／待用'})}</div><p class="inventory-note">“运行检查完成”只针对已测范围，精度与误伤看工具结论；源码和权重的许可限制在详情。</p></details>
<p id="inventory-count" aria-live="polite"></p><p id="inventory-empty" hidden>没有匹配工具。可将“范围”改为包含仅调研候选，或调整筛选。</p>'''
    groups = []
    for key, (title, description) in GROUPS.items():
        members = [f for f in tools if f['review']['purpose'] == key]
        groups.append(f'<details class="inventory-group" data-inventory-group="{key}" open><summary>{title} · <span class="inventory-visible-count">{len(members)}</span>个工具</summary><div class="inventory-cards">'+''.join(family_card(f) for f in members)+'</div></details>')
    # Retain the original history and coverage DOM IDs used by existing scripts.
    tail = old[old.index('<div class="box"><h2>更早的完整实验史与实际数值'):]
    tail = tail.replace('<div class="box"><h2>更早的完整实验史与实际数值', '<div class="box" id="inventory-history"><h2>更早的完整实验史与实际数值', 1)
    recent_panel = f'<div class="box" id="inventory-recent"><h2>最近聊天补记与结果入口</h2><p>以下{len(recent_records())}项是近期实验／决定摘要。数字保留原队列和任务，不能跨任务相加或排成一张分割榜。</p><details><summary>展开最近实验与决定</summary><div class="inventory-records">{recent}</div></details></div>'
    provenance=f'<details class="inventory-note"><summary>记录来源与维护</summary><p>按常用名称归并展示{len(tools)}组，内部{len(rows)}条配置／候选均保留。由唯一{link("glomerulus_tools/catalog.csv","工具主表")}生成；{link("glomerulus_tools/README.md","方法与调用说明")}。来源SHA256：<code id="inventory-source-sha">{digest}</code>。本页不代表完成所有未执行实验。</p></details>'
    return intro+'\n'.join(groups)+provenance+'</div>'+recent_panel+'<div class="box" id="inventory-legacy"><h2>原有实验按用途归档</h2><p>以下保留原16项摘要的文字与证据链接；属于当时结果，当前去留以上方结论为准。</p>'+legacy_groups(old)+'</div>'+tail


def update_page(path, section, old):
    page = path.read_text(encoding='utf-8')
    match = re.search(r'<section class="tab" id="tab-attempts".*?</section>', page, re.S)
    assert match, path
    # Another secretary task maintains this research note in the same tab.
    # It is not part of the generated tool inventory; retain it verbatim.
    extra = re.search(r'<div class="box" id="segformer-review-summary-20261116">.*?</div>', match.group(), re.S)
    if extra:
        note = '<details class="box" id="inventory-review-notes"><summary>人工复核材料与说明</summary>'+extra.group()+'</details>'
        section = section.replace('<div class="box" id="inventory-recent">', note+'<div class="box" id="inventory-recent">', 1)
    page = page[:match.start()] + section + page[match.end():]
    page = page.replace('data-tab="attempts">已尝试方案总结', 'data-tab="attempts">已经尝试工具')
    style = '<style id="inventory-style">'+CSS+'</style>'
    script = '<script id="inventory-script">'+SCRIPT+'</script>'
    if '<style id="inventory-style">' in page:
        page = re.sub(r'<style id="inventory-style">.*?</style>', lambda _: style, page, flags=re.S)
        page = re.sub(r'<script id="inventory-script">.*?</script>', lambda _: script, page, flags=re.S)
    else:
        page = page.replace('</head>', style+'</head>', 1)
        page = page.replace('</body>', script+'</body>', 1)
    path.write_text(page, encoding='utf-8')


def build_inventory(draft_path=None):
    output = Path(draft_path).parent if draft_path else OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    target = Path(draft_path) if draft_path else GALLERY
    before = GALLERY.read_text(encoding='utf-8')
    if draft_path:
        # Keep relative assets on their original base when previewing in outputs/.
        before = before.replace('<head>', '<head><base href="'+GALLERY.parent.as_uri()+'/">', 1)
        target.write_text(before, encoding='utf-8')
    archive = OUTPUT/'original_attempts_section.html'
    if not archive.exists():
        section = re.search(r'<section class="tab" id="tab-attempts".*?</section>', before, re.S).group()
        archive.write_text(section, encoding='utf-8')
    old = archive.read_text(encoding='utf-8')
    with (ROOT/'glomerulus_tools/catalog.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    section = render(rows, old)
    # A read-only reading snapshot; no tool execution or catalog mutation.
    update_page(target, section, old)
    if not draft_path and TEMPLATE.exists():
        update_page(TEMPLATE, section, old)
    after = target.read_text(encoding='utf-8')
    # All payloads, images, metrics, P0/P1 review and capabilities outside this tab are preserved.
    def rest(text):
        text = re.sub(r'<section class="tab" id="tab-attempts".*?</section>', '', text, flags=re.S)
        text = re.sub(r'<style id="inventory-style">.*?</style>', '', text, flags=re.S)
        text = re.sub(r'<script id="inventory-script">.*?</script>', '', text, flags=re.S)
        return text.replace('data-tab="attempts">已经尝试工具', 'data-tab="attempts">已尝试方案总结')
    assert rest(before) == rest(after), 'Unrelated gallery content changed'
    extra = re.search(r'<div class="box" id="segformer-review-summary-20261116">.*?</div>', before, re.S)
    if extra:
        assert after.count(extra.group()) == 1, 'The secretary review note must remain intact'
    receipt = dict(status='passed', catalog_sha256=hashlib.sha256((ROOT/'glomerulus_tools/catalog.csv').read_bytes()).hexdigest(),
                   tool_rows=len(rows), display_tools=len(families(rows)), purpose_counts={k:sum(f['review']['purpose']==k for f in families(rows)) for k in GROUPS},
                   state_counts={k:sum(state(r)==k for r in rows) for k in STATES}, recent_records=len(recent_records()),
                   original_experiment_rows=16, all_other_gallery_content_preserved=True,
                   template_updated=not draft_path and TEMPLATE.exists(), draft_only=bool(draft_path), model_inference=False, agent_access_changed=False,
                   sources_scope='visible conversation and existing local handoffs, not inaccessible chat archives')
    (output/'generation_verification.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    return receipt


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft',type=Path,help='Generate a preview only; do not modify shared pages')
    print(json.dumps(build_inventory(parser.parse_args().draft), ensure_ascii=False))
