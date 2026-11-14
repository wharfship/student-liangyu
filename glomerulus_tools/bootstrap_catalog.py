"""One-time migration of existing inventories. Never rerun over an existing main CSV."""
import csv
import json
from pathlib import Path
from .catalog import ROOT, sha

WORKSPACE=ROOT.parent
PROJECT=Path('D:/project/renal_tool_agent')
REMOTE='/home/chenliangyu/project/new_agent'
PYTHON='/home/chenliangyu/miniconda3/envs/'


def main():
    if (ROOT/'catalog.csv').exists():raise FileExistsError('Main CSV already exists; edit its rows instead')
    old_path=PROJECT/'protocols/EXECUTABLE_TOOL_PROTOCOLS_20261031.json'
    old=json.loads(old_path.read_text(encoding='utf-8'))
    audit_path=WORKSPACE/'research/strong_glomerulus_networks_20261105/training_and_input_audit.csv'
    with audit_path.open(encoding='utf-8-sig',newline='') as stream:audit=list(csv.DictReader(stream))
    rows=[]
    def row(tool_id,name,a,**kw):
        base=dict(tool_id=tool_id,name=name,family=name.split()[0],architecture=a['architecture'],
          species='human',stains='HE|PAS',agent_stains='',target='whole_glomerular_unit',input_forms='patch',
          enabled='no',agent_enabled='no',status='pending',training_sources=a['training_sources'],
          training_access=a['training_data_access'],test_overlap=a['test_overlap'],training_manifest='未获得精确checkpoint成员表',
          scale=a['physical_scale'],scale_evidence='见作者来源与冻结协议；未知不猜',preprocessing='待核',
          dependencies='none',recipe=a['conditioning'],verification=a['verification_status'],results='未取得可归属该配方的本地成绩',
          evidence=a['evidence'],code_license='未核',weight_license='未核',limitations=a.get('limitations') or '患者级重叠未排除',execution_json='{}')
        base.update(kw);rows.append(base);return base
    mapping={'omni_full':16,'ki2021_he_unit_batchstats':17,'ki2021_pas_unit_batchstats':18,
             'ki2021_sil_unit_batchstats':19,'ki2021_tri_unit_batchstats':20,'tokyo_espnet_pas':24,
             'capybara_kpis':9,'glo2_whole':21}
    for card in old['tools']:
        i=mapping[card['id']];execution=dict(card,adapter='legacy')
        # Retain legacy dataset/grid restrictions verbatim; no claim of arbitrary WSI support.
        r=row(card['id'],card['id'],audit[i],species='|'.join(card['species']),stains='|'.join(card['stains']),
              agent_stains='|'.join(card['stains']),input_forms='|'.join(card.get('input_forms',['patch'])),
              enabled='yes',agent_enabled='no' if card['id'].startswith('ki2021') else 'yes',
              status=card['status'],scale=card.get('scale_status') or audit[i]['physical_scale'],
              preprocessing=card.get('preprocessing') or card['protocol'],recipe=card['protocol'],
              dependencies='|'.join(card.get('dependencies',[])) or 'none',
              evidence='D:/project/renal_tool_agent/protocols/EXECUTABLE_TOOL_PROTOCOLS_20261031.json ; '+audit[i]['evidence'],
              execution_json=json.dumps(execution,ensure_ascii=False))
        if card['id']=='glo2_whole':
            r.update(results='Zenodo阳性HE50/PAS24 Dice0.7787/0.8113；HuBMAP8 0.8394；医院HE34/PAS6 0.6363/0.7582；医院硬化19/40',
                     scale='检测4.040058449 μm/px；分割按源MPP开区间裁2048/1024/512→512',
                     preprocessing='定位BGR作者归一化；分割RGB/255 task10 scale0；保留原融合规则')
        elif card['id']=='capybara_kpis':
            r.update(species='mouse',results='PAS四病组各3patch Dice0.9651/0.8887/0.9469/0.9374；2组织ROI0.9658/0.9546；非全WSI',
                     scale='沿用KPIs队列工作格；2048外窗→模型768；未支持任意MPP自动适配')
        elif card['id']=='omni_full':
            r.update(results='旧Omni有多个推理配方，分数必须按recipe对齐；不把原CAP矩阵分数自动归给此full入口')
    nepath=json.loads((WORKSPACE/'research/nepathtk_tools_20261105/models.json').read_text(encoding='utf-8'))['models']
    scores={'unet':'.8854','fpn':'.8834','deeplabv3plus':'.8824','mask2former_swint':'.8855'}
    for i,model in enumerate(nepath):
        mid=model['id'];source=REMOTE+'/nepathtk_tools_20261105';cp=source+'/'+model['checkpoint']
        execution=dict(adapter='nepath',model_id=mid,aliases=[mid],python=PYTHON+model['recommended_environment']+'/bin/python',
                       source_dir=source,checkpoint=cp,checkpoint_sha256=model['sha256'],patch=1024,stride=512,target_mpp=1.,
                       requires_mpp=True,timeout_seconds=10800,extra_paths=[REMOTE+('/strong_glomerulus_networks_20261105/deps' if model['backend']=='mmseg' else '/tool_expansion_20261105/deps')])
        if model['config']:
            execution.update(config=source+'/'+model['config'],config_sha256=sha(WORKSPACE/'research/nepathtk_tools_20261105'/model['config']))
        result='同8ROI逐图平均Dice'+scores[mid] if mid in scores else (
            'Zenodo78(含空GT)/HuBMAP8/医院40 Dice0.8866/0.8566/0.8694；NEPTUNE8/20银3/三3/HE1/PAS1 .9574/.9584/.9432/.7867；医院硬化37/40' if mid=='segformer_b5' else
            'Zenodo78(含空GT)/HuBMAP8/医院40 Dice0.8897/0.8469/0.8573；医院硬化36/40；NMF推理随机初始化')
        row('nepath_'+mid,'NePath '+model['architecture'],audit[i],family='NePathTK',stains='HE|PAS|SIL|TRI',agent_stains='HE|PAS',
            input_forms='patch|wsi',enabled='yes',agent_enabled='yes',status='full126_verified' if mid in ('segformer_b5','segnext_mscant') else 'pilot8_verified_WSI_wrapper_not_accuracy_validated',
            scale='源MPP→1.0 μm/px工作格；1024窗/512步长；非实测训练MPP',scale_evidence='作者d4/10×语义；冻结项目适配协议，非物理训练标定',
            preprocessing=json.dumps(model['preprocessing'],ensure_ascii=False),recipe='d4 AAG四类；概率均值→argmax→class1；无TTA；原坐标nearest',
            results=result,evidence='research/nepathtk_tools_20261105/verification.json ; research/full_networks_20261105/全量结果.md ; research/nepath_neptune_colors_20261106/README.md ; '+audit[i]['evidence'],
            code_license=model['code_license'],weight_license=model['weight_license'],
            limitations=audit[i]['limitations']+'；SIL/TRI小样本，仅研究调用；其余四架构WSI精度未验收',execution_json=json.dumps(execution,ensure_ascii=False))
    expansion=WORKSPACE/'research/glomerulus_tool_expansion_20261105';remote_exp=REMOTE+'/tool_expansion_20261105'
    qupath_cp='vendor/Code_to_Michael/checkpoint/best_metric_model_segmentation2d_array.pth'
    liki_folder='vendor/LiKiSeg/Dataset333_MECA_HE_SR_glu_light/nnUNetTrainer__nnUNetPlans__2d'
    for i,tool,adapter,checkpoint,pre in [(14,'qupath_monai','qupath',qupath_cp,'2048窗→512 nearest；ScaleIntensity；空间转置还原；sigmoid概率均值≥0.5'),
                                        (15,'likiseg333_light','likiseg',liki_folder+'/fold_0/checkpoint_final.pth','512滑窗；Gaussian；per-image ZScore；fold0；无镜像TTA；class2为球')]:
        execution=dict(adapter=adapter,python=PYTHON+'omni_seg/bin/python',source_dir=remote_exp,
                       checkpoint=remote_exp+'/'+checkpoint,checkpoint_sha256=sha(expansion/checkpoint),
                       requires_mpp=True,required_native_mpp=.5,extra_paths=[remote_exp+('/deps' if adapter=='qupath' else '/deps_nnunet')])
        if adapter=='likiseg':execution['model_folder']=remote_exp+'/'+liki_folder
        row(tool,tool,audit[i],species='human' if adapter=='qupath' else 'human|mouse',stains='HE|PAS' if adapter=='qupath' else 'HE|PAS|MECA32|SIRIUSRED',
            enabled='yes',agent_enabled='no',status='research_only_pilot8_scale_unknown',
            scale='仅保持已测0.5原生工作格；训练MPP未知；不适配未知倍率',preprocessing=pre,recipe=pre,
            results='同8人ROI Dice'+('0.5505' if adapter=='qupath' else '0.7293；鼠域未测'),
            evidence='research/glomerulus_tool_expansion_20261105/调研与试跑报告.md ; '+audit[i]['evidence'],
            code_license='MIT' if adapter=='qupath' else 'GPL-3.0',weight_license='未单独声明' if adapter=='qupath' else 'CC-BY-4.0',execution_json=json.dumps(execution,ensure_ascii=False))
    # Preserve researched candidates and component identities without pretending they can execute.
    pending=[(6,'hats_efficientsam'),(7,'prpseg'),(8,'mori_seg_paused'),(10,'saltfish_kpis'),(11,'hula_unpaired'),
             (12,'mescnn_cascade_r50'),(13,'mescnn_mask_r101'),(22,'glo_gs_head_component'),
             (23,'glo_circlenet5_component'),(25,'sarder_multicompartment')]
    for i,tool in pending:
        r=row(tool,tool,audit[i],species='mouse' if i==10 else 'human',stains='PAS' if i==10 else '待核',
              status='paused_by_user' if i==8 else ('component_not_standalone_tool' if i in (22,23) else 'not_inference_accepted'),
              preprocessing=audit[i]['conditioning'])
        if i==8:r.update(results='Zenodo78/医院40 Dice0.5854/0.4702；已按用户决定暂停，不追加运行',evidence='research/full_networks_20261105/全量结果.md ; '+r['evidence'])
    # Explicitly keep frozen experimental recipes apart from the current executable implementation.
    for tool,name,recipe,result,evidence in [
        ('omni_cap_matrix_original','OmniCAP 原大表配方','原大表variant及portable_mask_path冻结配方','Zenodo阳性HE50/PAS24 .6693/.8226；HuBMAP8 .5216；医院HE34/PAS6 .3739/.3408','D:/project/renal_tool_agent/all_dataset_dice_20261102/全数据集_逐图Dice_RESEARCH_ONLY.csv'),
        ('omni_outer_unanimity_frozen','Omni 外层全票冻结配方','外层全票；仅复用冻结结果，未作为通用新任务执行器','医院HE34/PAS6 .6145/.6454；HuBMAP8 .7766','research/omni_outer_unanimity_20261103/pipeline_state.json')]:
        row(tool,name,audit[16],stains='HE|PAS|SIL|TRI',status='frozen_recipe_reference_only',recipe=recipe,
            preprocessing='按冻结源码与variant；不与当前omni_full混用分数',results=result,evidence=evidence)
    with (ROOT/'catalog.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    migration=ROOT/'migration';migration.mkdir(exist_ok=True)
    (migration/'catalog_sources.json').write_text(json.dumps(dict(legacy=str(old_path),legacy_sha256=sha(old_path),
         audit=str(audit_path),audit_sha256=sha(audit_path),rows=len(rows),sole_future_source='../catalog.csv'),ensure_ascii=False,indent=2),encoding='utf-8')
    print('Migrated',len(rows),'rows')


if __name__=='__main__':main()
