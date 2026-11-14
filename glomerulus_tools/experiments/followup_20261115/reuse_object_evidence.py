"""Join existing frozen original-ID results; never run another legacy model."""
from collections import Counter,defaultdict
from common import *

def main():
 cohort=read(OLD/'cohort_EVAL_ONLY.json');out=[];summary=[]
 for o in cohort:
  case=read(ROOT/'research/segformer_complementarity_20261106/cases'/(o['task_id']+'.json'))
  before=next(g for g in case['baseline_objects'] if str(g['instance_id'])==o['instance_id'])
  for tool in case['tools']:
   if tool['compatibility']!='accepted_frozen_run' and tool['model'] in ['ki2021','tokyo_omni']:continue
   if o['stain']=='HE' and tool['model'] in ['ki2021','tokyo_omni']:continue
   target=next((g for g in tool['objects'] if str(g['instance_id'])==o['instance_id']),None)
   if target is None:continue
   out.append(dict(case_id=o['case_id'],cohort=o['cohort'],group_id=o['group_id'],task_id=o['task_id'],instance_id=o['instance_id'],
    model=tool['model'],stain=o['stain'],original_source_class=o['source_class'],baseline_matched05=before['matched_0.5'],
    other_matched05=target['matched_0.5'],baseline_true_omission=before['true_omission'],other_true_omission=target['true_omission'],
    baseline_coverage=before['pixel_coverage'],other_coverage=target['pixel_coverage'],baseline_best_iou=before['best_component_iou'],
    other_best_iou=target['best_component_iou'],evaluation_only=True,inference_reused=True))
 for model in sorted({r['model'] for r in out}):
  for group in ['sclerosis','normal_control']:
   rr=[r for r in out if r['model']==model and r['cohort']==group]
   summary.append(dict(model=model,cohort=group,n=len(rr),baseline_matched=sum(r['baseline_matched05'] for r in rr),
     other_matched=sum(r['other_matched05'] for r in rr),rescued=sum(not r['baseline_matched05'] and r['other_matched05'] for r in rr),
     harmed=sum(r['baseline_matched05'] and not r['other_matched05'] for r in rr),
     true_omission_rescued=sum(r['baseline_true_omission'] and r['other_matched05'] for r in rr)))
 csvout(REPORT/'旧工具同40硬化与保护对照_EVAL_ONLY.csv',out);save(HERE/'legacy_object_summary.json',summary)
 omitted={r['case_id'] for r in out if r['cohort']=='sclerosis' and r['baseline_true_omission']}
 save(HERE/'three_omissions_legacy.json',[r for r in out if r['case_id'] in omitted])
 print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
