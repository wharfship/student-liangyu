"""One metadata-only interface; models execute in their registered isolated environments."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .catalog import ROOT, asset_status, get_card, load_registry
from .contract import compatibility, prepare_task


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['list','describe','check','run','crop','review','classify','assess','refine','multiclass'])
    parser.add_argument('--tool');parser.add_argument('--task',type=Path)
    parser.add_argument('--output',type=Path);parser.add_argument('--research',action='store_true')
    parser.add_argument('--research-stain-override',choices=['HE'],help='Explicit out-of-domain Glo5 research test; does not change registered support')
    parser.add_argument('--candidates',type=Path,help='Predicted candidate mask for a dependent tool; never GT')
    args=parser.parse_args()
    if args.research_stain_override and not (args.research and args.action in ('classify','check') and args.tool=='glo5_pas_identity'):
        parser.error('Stain override is only allowed for Glo5 research classification/check')
    if args.action=='list':
        cards=load_registry()['tools']
        print(json.dumps([dict(tool_id=c['id'],status=c['status'],agent_enabled=c['agent_enabled'],
                               species=c['species'],stains=c['stains'],input_forms=c['input_forms'],
                               **asset_status(c)) for c in cards],ensure_ascii=False,indent=2));return
    if args.action=='describe':
        if not args.tool or not args.research:parser.error('Function protocols are researcher-only; need --tool and --research')
        from .workflow_interfaces import describe
        print(json.dumps(describe(get_card(args.tool)),ensure_ascii=False,indent=2));return
    if not args.tool or not args.task:parser.error('Need --tool and --task')
    card=get_card(args.tool)
    if card.get('adapter')=='planned_workflow':
        if args.action!='check' or not args.research:
            parser.error('Pending tool has no executable adapter; only check --research is available; Agent access disabled')
        if args.candidates:parser.error('Pending workflow does not accept candidate masks')
        from .workflow_interfaces import check_request
        result=check_request(json.loads(args.task.read_text(encoding='utf-8-sig')),card)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        raise SystemExit(2)  # Protocol valid is not a runnable-tool pass.
    if card.get('adapter')=='pathology_utility':
        from .pathology_utilities import validate_batch
        if args.action not in ('run','check') or not args.research or not card['enabled']:
            parser.error('Pathology utility requires enabled tool, run/check and --research')
        if args.candidates:parser.error('Image-only utility does not accept masks')
        task=json.loads(args.task.read_text(encoding='utf-8-sig'));validate_batch(task,card)
        assets=asset_status(card)
        if args.action=='check':
            print(json.dumps(dict(tool_id=card['id'],compatible=True,**assets),ensure_ascii=False,indent=2))
            if not assets['available']:raise SystemExit(2)
            return
        if not assets['available']:raise ValueError('Missing utility assets: '+str(assets))
        if not args.output:parser.error('Need --output')
        if args.output.exists():raise ValueError('Output directory must be new')
        subprocess.run([card['python'],'-u','-m','glomerulus_tools.pathology_utilities','--tool',card['id'],
            '--task',str(args.task.resolve()),'--output',str(args.output.resolve())],cwd=ROOT.parent,check=True,timeout=7200)
        return
    if args.action=='run' and card.get('adapter')=='mescnn':
        from .mescnn import validate_batch
        if not args.research or not card['enabled']:parser.error('MESCnn prepared tiles require enabled research tool')
        task=json.loads(args.task.read_text(encoding='utf-8-sig'));validate_batch(task)
        if not args.output:parser.error('Need --output')
        if not asset_status(card)['available']:raise ValueError('Missing MESCnn assets')
        subprocess.run([card['python'],'-u','-m','glomerulus_tools.mescnn','--tool',card['id'],
            '--task',str(args.task.resolve()),'--output',str(args.output.resolve())],cwd=ROOT.parent,check=True,timeout=7200)
        return
    if args.action=='multiclass':
        from .multiclass import validate_batch
        if not args.research or not card.get('multiclass_export'):
            parser.error('Semantic-class export requires registered support and --research')
        if args.candidates:parser.error('Semantic-class export does not use candidate masks')
        task=json.loads(args.task.read_text(encoding='utf-8-sig'));validate_batch(task,card)
        if not asset_status(card)['available']:raise ValueError('Missing registered model assets')
        if not args.output:parser.error('Need --output')
        subprocess.run([card['python'],'-u','-m','glomerulus_tools.multiclass','--tool',card['id'],
                        '--task',str(args.task.resolve()),'--output',str(args.output.resolve())],
                       cwd=ROOT.parent,check=True,timeout=card.get('timeout_seconds',10800)*len(task['cases']))
        return
    if args.research_stain_override:
        card=dict(card,stains=card['stains']+[args.research_stain_override])
    if args.action=='refine' or (args.action=='check' and card.get('adapter')=='mask_refinement'):
        from .mask_refinement import validate_task
        if card.get('adapter')!='mask_refinement' or not card['enabled'] or not args.research:
            parser.error('Mask refinement requires a registered enabled ID and --research')
        if args.candidates:parser.error('Refinement uses its own predicted-mask task schema')
        task=json.loads(args.task.read_text(encoding='utf-8-sig'));validate_task(task,card)
        assets=asset_status(card)
        if args.action=='check':
            print(json.dumps(dict(tool_id=card['id'],compatible=True,**assets),ensure_ascii=False,indent=2))
            if not assets['available']:raise SystemExit(2)
            return
        if not assets['available']:raise ValueError('Missing refinement assets: '+str(assets))
        if not args.output:parser.error('Need --output')
        if args.output.exists():raise ValueError('Output directory must be new')
        subprocess.run([card['python'],'-u','-m','glomerulus_tools.mask_refinement','--tool',card['id'],
                        '--task',str(args.task.resolve()),'--output',str(args.output.resolve())],
                       cwd=ROOT.parent,check=True,timeout=card.get('timeout_seconds',10800))
        return
    if args.action=='assess' or (args.action=='check' and card.get('adapter')=='object_evidence'):
        from .classification import validate_task
        if card.get('adapter')!='object_evidence' or not card['enabled'] or not args.research:
            parser.error('Object evidence requires a registered enabled ID and --research')
        if args.candidates:parser.error('Object evidence uses predicted boxes in its task schema')
        task=json.loads(args.task.read_text(encoding='utf-8-sig'));validate_task(task,card)
        assets=asset_status(card)
        if args.action=='check':
            print(json.dumps(dict(tool_id=card['id'],compatible=True,**assets),ensure_ascii=False,indent=2))
            if not assets['available']:raise SystemExit(2)
            return
        if not assets['available']:raise ValueError('Missing object evidence assets: '+str(assets))
        if not args.output:parser.error('Need --output')
        if args.output.exists():raise ValueError('Output directory must be new')
        subprocess.run([card['python'],'-u','-m','glomerulus_tools.object_evidence','--tool',card['id'],
                        '--task',str(args.task.resolve()),'--output',str(args.output.resolve())],
                       cwd=ROOT.parent,check=True,timeout=card.get('timeout_seconds',3600))
        return
    if args.action=='classify' or (args.action=='check' and card.get('adapter')=='candidate_classifier'):
        from .classification import validate_task
        if card.get('adapter')!='candidate_classifier' or not card['enabled'] or not args.research:
            parser.error('Candidate classifiers require a registered enabled ID and --research')
        if args.candidates:parser.error('Classification uses predicted boxes in its own task schema')
        task=json.loads(args.task.read_text(encoding='utf-8-sig'))
        validate_task(task,card)
        assets=asset_status(card)
        if args.action=='check':
            print(json.dumps(dict(tool_id=card['id'],compatible=True,**assets),ensure_ascii=False,indent=2))
            if not assets['available']:raise SystemExit(2)
            return
        if not assets['available']:raise ValueError('Missing classifier assets: '+str(assets))
        if not args.output:parser.error('Need --output')
        if args.output.exists():raise ValueError('Output directory must be new')
        command=[card['python'],'-u','-m','glomerulus_tools.classification','--tool',card['id'],
                 '--task',str(args.task.resolve()),'--output',str(args.output.resolve())]
        if args.research_stain_override:
            command+=['--research-stain-override',args.research_stain_override]
        subprocess.run(command,cwd=ROOT.parent,check=True,timeout=card.get('timeout_seconds',3600))
        return
    if args.action in ('crop','review'):
        from .candidate_views import CandidateViews, dump
        from .vlm_jpeg import read_candidate_jpeg
        expected='candidate_crop_v1' if args.action=='crop' else 'visual_reader_v1'
        if card['id']!=expected or not card['enabled'] or not args.research:
            parser.error('Candidate utilities require their registered ID and --research')
        if not args.output:parser.error('Need --output')
        task=json.loads(args.task.read_text(encoding='utf-8-sig'))
        if args.action=='review':
            result=read_candidate_jpeg(task,args.output)
        else:
            args.output.mkdir(parents=True,exist_ok=False)
            views=CandidateViews(task,args.output)
            _,result=views.render('initial')
            dump(args.output/'task_NO_GT.json',task)
        print(json.dumps(result,ensure_ascii=False,indent=2));return
    task=prepare_task(json.loads(args.task.read_text(encoding='utf-8-sig')))
    reasons=compatibility(card,task,research=args.research)
    assets=asset_status(card)
    check=dict(tool_id=card['id'],compatible=not reasons,reasons=reasons,task=task,**assets)
    if args.action=='check':
        print(json.dumps(check,ensure_ascii=False,indent=2))
        if reasons or not assets['available']:raise SystemExit(2)
        return
    if reasons or not assets['available']:raise ValueError(json.dumps(check,ensure_ascii=False))
    if not args.output:parser.error('Need --output')
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):raise ValueError('Set CUDA_VISIBLE_DEVICES after checking current GPU availability')
    if card.get('dependencies') and not args.candidates:
        raise ValueError('This CLI requires a predicted --candidates mask; Agent batch interface can execute the registered localizer')
    if args.candidates and not card.get('dependencies'):raise ValueError('Tool does not accept a candidate mask')
    args.output.mkdir(parents=True,exist_ok=False)
    task_file=args.output/'online_task.json';task_file.write_text(json.dumps(task,ensure_ascii=False,indent=2),encoding='utf-8')
    command=[card['python'],'-u','-m','ops.tool_protocol_runner','--tool',card['id'],'--task-json',str(task_file),
             '--output',str(args.output)]
    if args.research:command+=['--research']
    if args.candidates:command+=['--candidates',str(args.candidates)]
    environment=os.environ.copy();environment.update(card.get('environment',{}))
    with (args.output/'inference.log').open('w',encoding='utf-8') as log:
        try:
            result=subprocess.run(command,cwd=ROOT.parent,env=environment,stdout=log,stderr=subprocess.STDOUT,
                                  timeout=card.get('timeout_seconds',10800),check=False)
        except subprocess.TimeoutExpired:
            (args.output/'failure.json').write_text(json.dumps(dict(status='timeout',gt_read=False)),encoding='utf-8')
            raise
    receipt=args.output/'inference_record.json'
    if result.returncode or not receipt.exists():
        (args.output/'failure.json').write_text(json.dumps(dict(status='execution_failed',exit_code=result.returncode,
                    log=str(args.output/'inference.log'),gt_read=False)),encoding='utf-8')
        raise RuntimeError('Worker failed; see '+str(args.output/'inference.log'))
    record=json.loads(receipt.read_text(encoding='utf-8'))
    if not record.get('valid'):raise RuntimeError('Worker did not return a valid result')
    print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
