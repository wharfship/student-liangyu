"""Generate runtime JSON and paper-facing tables from the ONE editable CSV."""
import csv
import json
from pathlib import Path
from .catalog import ROOT, sha


def build(update_browser=True):
    with (ROOT/'catalog.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    cards, ids, aliases = [], set(), set()
    required = ['tool_id', 'name', 'architecture', 'training_sources', 'training_access',
                'test_overlap', 'stains', 'scale', 'preprocessing', 'verification', 'evidence', 'execution_json']
    for row in rows:
        for key in required:
            if not row.get(key): raise ValueError(f"{row.get('tool_id')}: missing {key}")
        if row['tool_id'] in ids: raise ValueError('Duplicate tool ID')
        ids.add(row['tool_id'])
        card = json.loads(row['execution_json'])
        card.update(id=row['tool_id'], name=row['name'], protocol=row['recipe'], status=row['status'],
                    enabled=row['enabled']=='yes', agent_enabled=row['agent_enabled']=='yes',
                    species=row['species'].split('|'), stains=row['stains'].split('|'),
                    agent_stains=row['agent_stains'].split('|') if row['agent_stains'] else [],
                    input_forms=row['input_forms'].split('|'), target=row['target'],
                    scale_status=row['scale'], limitations=[row['limitations']])
        card.update(training_sources=row['training_sources'],training_data_access=row['training_access'],
                    training_overlap_status=row['test_overlap'],code_license=row['code_license'],weight_license=row['weight_license'])
        for alias in card.get('aliases', []):
            if alias in aliases: raise ValueError('Duplicate alias')
            aliases.add(alias)
        if card['enabled'] and card.get('adapter') not in ('legacy', 'nepath', 'qupath', 'likiseg','mescnn','glo_matrix','candidate_view','visual_reader','candidate_classifier','object_evidence','mask_refinement','pathology_utility'):
            raise ValueError('Enabled tool has no implemented adapter: '+card['id'])
        required_assets=('python','source_dir') if card.get('adapter') in ('candidate_view','visual_reader') else ('python','checkpoint','source_dir')
        if card['enabled'] and not all(card.get(k) for k in required_assets):
            raise ValueError('Enabled tool missing assets: '+card['id'])
        if card['agent_enabled'] and not card['enabled']: raise ValueError('Agent tool must be enabled')
        cards.append(card)
    if ids & aliases: raise ValueError('Alias conflicts with canonical ID')
    for card in cards:
        localizer = card.get('default_localizer')
        if localizer and localizer not in ids: raise ValueError('Unknown default localizer')
    defaults={}
    for row in rows:
        if row.get('default_for'):
            species=row['default_for']
            if species in defaults:raise ValueError('Duplicate default tool for '+species)
            if row['agent_enabled']!='yes':raise ValueError('Default must be an enabled Agent tool')
            defaults[species]=row['tool_id']
    data = dict(schema_version=1, catalog_sha256=sha(ROOT/'catalog.csv'), tools=cards,
                default_tools=defaults,
                maintenance='Edit catalog.csv only; generated registry and tables must not be hand-edited')
    (ROOT/'runtime_registry.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    import xlsxwriter
    workbook = xlsxwriter.Workbook(ROOT/'工具总表.xlsx')
    sheet = workbook.add_worksheet('工具总表')
    headers = [k for k in rows[0] if k not in ('execution_json','researcher_note')]
    expanded=['checkpoint','checkpoint_sha256','python','config','adapter']
    headers+=expanded
    headers+=['researcher_note']
    display_rows=[]
    for row,card in zip(rows,cards):
        display_rows.append(dict(row,**{k:card.get(k,'') for k in expanded}))
    wrap=workbook.add_format({'text_wrap':True,'valign':'top'})
    labels=dict(tool_id='工具ID',name='名称',family='来源家族',architecture='网络架构',species='物种',
        stains='兼容/声明染色',agent_stains='Agent开放染色',target='任务目标',input_forms='输入格式',
        enabled='允许执行',agent_enabled='开放Agent',status='当前状态',training_sources='训练来源',
        training_access='训练数据公开性',test_overlap='与测试来源重叠风险',training_manifest='训练成员清单',
        scale='源尺度与工作尺度',scale_evidence='尺度依据',preprocessing='预处理',dependencies='定位依赖',
        recipe='推理配方',verification='真实验收情况',results='指标及样本分母',evidence='证据入口',
        code_license='源码许可',weight_license='权重许可',limitations='限制与不确定性',default_for='默认推荐物种',
        checkpoint='权重路径',checkpoint_sha256='权重SHA256',python='执行环境Python',config='模型配置',adapter='适配器',
        researcher_note='给研究者的备注（不进入AI输入）')
    labels.update(display_family='浏览器工具归组',display_name='常用工具名',
                  display_conclusion='效果结论与当前用途',display_review='获取测试与检查范围（JSON）')
    sheet.add_table(0,0,len(rows),len(headers)-1,{'name':'GlomerulusTools','style':'Table Style Medium 2',
        'columns':[{'header':labels.get(k,k),'format':wrap} for k in headers],
        'data':[[row[k] for k in headers] for row in display_rows]})
    sheet.freeze_panes(1,3)
    sheet.set_column(0,len(headers)-1,36,wrap)
    for i in range(1,len(rows)+1):sheet.set_row(i,90)
    workbook.close()
    overview=['|工具ID|网络/配方|物种·染色|执行/Agent|尺度|验收与效果|', '|---|---|---|---|---|---|']
    for row in rows:
        cells=[row['tool_id'],row['name'],row['species']+' · '+row['stains'].replace('|','/'),
               row['enabled']+'/'+row['agent_enabled'],row['scale'],row['verification']+'；'+row['results']]
        overview.append('|'+ '|'.join(c.replace('|','/').replace('\n',' ') for c in cells)+'|')
    path=ROOT/'README.md'; doc=path.read_text(encoding='utf-8')
    start,end='<!-- TABLE_START -->','<!-- TABLE_END -->'
    before,tail=doc.split(start,1);_,after=tail.split(end,1)
    path.write_text(before+start+'\n'+'\n'.join(overview)+'\n'+end+after,encoding='utf-8')
    from .browser_inventory import GALLERY, build_inventory
    if update_browser and GALLERY.is_file():
        build_inventory()
    print(json.dumps(dict(status='PASSED',rows=len(rows),enabled=sum(c['enabled'] for c in cards),
                          agent_enabled=sum(c['agent_enabled'] for c in cards)),ensure_ascii=False))


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-browser',action='store_true',help='Build catalog derivatives without writing a gallery held by another task')
    build(update_browser=not parser.parse_args().skip_browser)
