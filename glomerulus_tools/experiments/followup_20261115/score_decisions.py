"""Offline only: score recorded observation decisions without exposing GT to policy."""
import argparse,collections
from common import *

def validate(decisions,packages):
 index={p['candidate_id']:p for p in packages}
 if not isinstance(decisions,list) or len(decisions)!=len(index):raise ValueError('Require one decision per candidate')
 seen=set()
 for d in decisions:
  if not isinstance(d,dict):raise ValueError('Decision must be an object')
  if set(d)!={'candidate_id','decision','views_requested','reason'}:raise ValueError('Only explicit decision fields accepted')
  cid=d['candidate_id']
  if cid not in index or cid in seen:raise ValueError('Unknown/duplicate candidate')
  seen.add(cid)
  if d['decision'] not in ['keep','accept','manual_review']:raise ValueError('Invalid decision')
  if not isinstance(d['reason'],str) or not d['reason'].strip():raise ValueError('Require a nonempty textual reason')
  views=d['views_requested']
  if not isinstance(views,list) or len(views)!=len(set(views)) or len(views)>3:raise ValueError('Observation budget violation')
  if set(views)-set(index[cid]['allowed_view_names']):raise ValueError('Unknown view')
 return decisions

def score(decisions,output):
 packages=read(HERE/'observation_packages_NO_GT.json');validate(decisions,packages)
 links={r['view_id']:r for r in read(HERE/'observation_linkage_EVAL_ONLY.json')}
 old={r['case_id']:r for r in rows(ROOT/'已经整理好的内容/医院分割结果分析/硬化专项验证_20261114/专项逐对象结果_EVAL_ONLY.csv') if r['method']=='ritm_hrnet32_external_mask_point1'}
 evaluated=[]
 for d in decisions:
  link=links[d['candidate_id']];r=old[link['case_id']];gain=float(r['delta_target_dice']);accepted=d['decision']=='accept'
  evaluated.append(dict(view_id=d['candidate_id'],cohort=link['cohort'],group_id=link['group_id'],decision=d['decision'],
    target_baseline=float(r['target_dice_before']),target_after=float(r['target_dice_after']) if accepted else float(r['target_dice_before']),
    realized_delta=gain if accepted else 0.,proposal_delta=gain,views=len(d['views_requested']),
    beneficial=gain>.01,harmful=gain<-.01,accepted=accepted,
    local_delta=float(r['delta_local_dice']) if accepted else 0.,
    note='manual_review keeps original mask; not a global WSI policy score'))
 summary=[]
 for group in ['ALL','sclerosis','normal_control']:
  rr=[r for r in evaluated if group=='ALL' or r['cohort']==group];n=len(rr)
  accepted=[r for r in rr if r['accepted']]
  summary.append(dict(cohort=group,n=n,accepted=len(accepted),review=sum(r['decision']=='manual_review' for r in rr),
    mean_target_delta=sum(r['realized_delta'] for r in rr)/n,
    accepted_beneficial=sum(r['beneficial'] for r in accepted),accepted_harmful=sum(r['harmful'] for r in accepted),
    beneficial_proposals=sum(r['beneficial'] for r in rr),harmful_proposals=sum(r['harmful'] for r in rr),
    accepted_beneficial_precision=sum(r['beneficial'] for r in accepted)/len(accepted) if accepted else None,
    harmful_acceptance_rate=sum(r['harmful'] and r['accepted'] for r in rr)/sum(r['harmful'] for r in rr) if any(r['harmful'] for r in rr) else None,
    requested_view_images=sum(r['views'] for r in rr),cohort_scope='77 located targets; excludes3 complete omissions; exposed development'))
 save(output,dict(summary=summary,per_candidate=evaluated,not_Agent_result_without_actual_decisions=True,GT_offline_only=True))
 return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--decisions',type=Path);p.add_argument('--output',type=Path);p.add_argument('--verify-baselines',action='store_true');a=p.parse_args()
 if a.verify_baselines:
  packages=read(HERE/'observation_packages_NO_GT.json');records=[]
  for name in ['keep','accept']:
   ds=[dict(candidate_id=p['candidate_id'],decision=name,views_requested=[],reason='fixed reproducibility reference') for p in packages]
   output=HERE/('fixed_'+name+'_offline_score.json');summary=score(ds,output)
   if name=='keep':assert all(r['mean_target_delta']==0 for r in summary)
   else:
    raw=rows(ROOT/'已经整理好的内容/医院分割结果分析/硬化专项验证_20261114/专项逐对象结果_EVAL_ONLY.csv')
    for s in summary:
     rr=[r for r in raw if r['method']=='ritm_hrnet32_external_mask_point1' and r['refinement_available']=='True' and (s['cohort']=='ALL' or r['cohort']==s['cohort'])]
     assert abs(s['mean_target_delta']-sum(float(r['delta_target_dice']) for r in rr)/len(rr))<1e-12
   records.append(dict(policy=name,summary=summary))
  save(HERE/'decision_scoring_verification.json',dict(status='passed',fixed_references=records,actual_VLM_or_Agent_run=False))
  print('keep reproduces zero delta; accept exactly reproduces old frozen77 results')
 else:
  if a.decisions is None or a.output is None:p.error('Need --decisions and --output')
  print(score(read(a.decisions),a.output))

if __name__=='__main__':main()
