"""Restricted candidate reader: request real views or submit an identity reading."""
import base64
import json
import os
from pathlib import Path
import re
import time

import requests

from .candidate_views import CandidateViews, VIEWS, dump
from .catalog import sha

MODEL = 'qwen3.8-flash'
COMMON = '''You are the dedicated visual reader of ONE kidney candidate, not the global controller.
Judge the candidate tissue marked by a THIN YELLOW SUPPORT OUTLINE, not whether any glomerulus exists anywhere in the rectangle. Raw image then outline image always form a pair. The outline is a predicted proposal, NOT a correct annotation. No true label is available. Do not infer identity from tool agreement or the proposal itself.
You cannot choose other candidates, invoke segmentation, edit masks, approve a replacement, or manage the global budget. Identity and boundary quality are different questions. This task is identity only; do not recommend editing from identity alone. Never assume the candidate already contains a glomerulus.
Return only JSON: either {"action":"request_view","view":"one allowed name","reason":"brief specific unresolved visual question"} or {"action":"submit_reading","identity":"glomerulus|non_glomerulus|uncertain","p_glomerulus":0.0,"adequacy":"adequate|limited|insufficient","differential":"glomerulus|tubule|vessel|mixed|other|unknown","evidence":"visible architecture and contradictory evidence; short","uncertainty":"what is unresolved; short"}.
Use glomerulus only for p>=0.8, non_glomerulus only for p<=0.2; otherwise uncertain. Limited visibility or an unresolved identity requires uncertain; insufficient visibility also requires uncertain. These are uncalibrated subjective scores, not validated probabilities. No extra fields.
'''
PATHOLOGY = '''Pathology knowledge, adapted from the user's PAS verifier/repair prompts and audited for HE:
1. A renal corpuscle comprises a glomerular capillary tuft and, when visible, Bowman space and capsule. Seek coherent capillary/mesangial tuft architecture and its relation to surrounding tissue. A single lumen lined by a continuous tubular epithelial ring suggests a tubule; repeated neighboring tubular profiles must not be merged into an imagined tuft. Vessel walls and lumina are another differential. Explain the actual visible evidence for this candidate.
2. Roundness, dense nuclei, purple color, a compact mask, a detection box, or a white halo alone do not establish glomerular identity. Transverse/oblique tubules can mimic round corpuscles. PAS also stains tubular and vascular basement membranes: a purple rim is not specific.
3. In PAS, basement-membrane and mesangial matrix contrast can help interpret structure. In HE, reason from architecture, nuclear/cytoplasmic organization and lumen patterns; do not require a PAS-purple basement membrane or apply PAS color rules.
4. Sclerosis, collapse, injury, cutting plane and limited field may obscure the tuft, Bowman space or capsule. A closed capsule nuclear ring, open Bowman space and healthy appearance are NOT required to be a glomerulus. Their absence is not proof of non-glomerulus. If morphology is equivocal, use uncertain and state why; do not diagnose disease from this task.
5. Whole-corpuscle versus tuft annotation is dataset-dependent. Lack of a proposed capsule rim is not a negative identity criterion. Nearby tubules/interstitium should not be accepted merely to complete a round boundary. For a mixed proposal, judge whether glomerular tissue is actually represented in the marked support and state contamination separately.
6. Native pixels, context and complementary views can resolve a tubule/tuft ambiguity; interpolation cannot create tissue evidence. Inspect raw tissue before proposal geometry. If the target is clipped or too small, request an appropriate available view or return uncertain. A numerical MPP supplied as a working assumption is not measured scale. Do not use unsupported size cutoffs.
'''


def get_key():
    key = os.environ.get('DASHSCOPE_API_KEY')
    if key:
        return key
    text = (Path(__file__).resolve().parents[1]/'AGENTS.md').read_text(encoding='utf8')
    match = re.search(r'DASHSCOPE_API_KEY\s*=\s*["\']([^"\']+)', text)
    if not match:
        raise RuntimeError('DASHSCOPE_API_KEY unavailable')
    return match.group(1)


def api_call(messages):
    body = dict(model=MODEL, messages=messages, temperature=0, max_tokens=1000,
                enable_thinking=False, response_format={'type': 'json_object'})
    start = time.monotonic()
    response = requests.post('https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
                             headers={'Authorization': 'Bearer '+get_key()}, json=body, timeout=150)
    if response.status_code != 200:
        raise RuntimeError('Provider HTTP '+str(response.status_code))
    result = response.json()
    return dict(raw=result, seconds=time.monotonic()-start)


def validate(card, allow_request):
    if card.get('action') == 'request_view':
        if not allow_request or set(card) != {'action', 'view', 'reason'} or card['view'] not in VIEWS:
            raise ValueError('Illegal view request')
        if not isinstance(card['reason'], str) or not card['reason'].strip():
            raise ValueError('View request needs a question')
        return
    fields = {'action', 'identity', 'p_glomerulus', 'adequacy', 'differential', 'evidence', 'uncertainty'}
    if set(card) != fields or card['action'] != 'submit_reading':
        raise ValueError('Invalid reading schema')
    p = card['p_glomerulus']
    if type(p) not in (int, float) or not 0 <= p <= 1:
        raise ValueError('Invalid score')
    if card['identity'] not in ('glomerulus', 'non_glomerulus', 'uncertain'):
        raise ValueError('Invalid identity')
    if card['adequacy'] not in ('adequate', 'limited', 'insufficient'):
        raise ValueError('Invalid adequacy')
    if card['differential'] not in ('glomerulus', 'tubule', 'vessel', 'mixed', 'other', 'unknown'):
        raise ValueError('Invalid differential')
    if card['identity'] == 'glomerulus' and (p < .8 or card['adequacy'] != 'adequate'):
        raise ValueError('Incoherent positive reading')
    if card['identity'] == 'non_glomerulus' and (p > .2 or card['adequacy'] != 'adequate'):
        raise ValueError('Incoherent negative reading')
    if any(not isinstance(card[k], str) or len(card[k]) > 2000 for k in ('evidence', 'uncertainty')):
        raise ValueError('Invalid evidence text')


def image_message(text, paths):
    content = [dict(type='text', text=text)]
    for p in paths:
        encoded = base64.b64encode(Path(p).read_bytes()).decode()
        content.append(dict(type='image_url', image_url={'url': 'data:image/png;base64,'+encoded}))
    return dict(role='user', content=content)


def read_candidate(task, output, variant='autonomous', caller=api_call):
    if variant not in ('generic_static', 'pathology_static', 'autonomous'):
        raise ValueError('Unknown variant')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    dump(output/'task_NO_GT.json', task)
    views = CandidateViews(task, output/'views')
    paths, receipt = views.render('initial')
    system = COMMON + (PATHOLOGY if variant != 'generic_static' else '')
    dump(output/'prompt_NO_GT.json', dict(system=system, variant=variant, model=MODEL))
    messages = [dict(role='system', content=system)]
    history = []
    calls = []
    attempted = 0
    final = None
    limit = 3 if variant == 'autonomous' else 1
    next_text = ('Candidate '+task['candidate_id']+'; stain='+task['stain']+'. Scale basis='+task['scale_basis']+
                 '. Each pair shows the same field: raw followed by exact candidate support. View receipt='+json.dumps(receipt))
    for step in range(limit):
        available = [v for v in VIEWS if v not in views.receipts] if step < limit-1 else []
        instruction = ('Allowed request_view names='+json.dumps(available)+'. Remaining API calls including this one='+str(limit-step)+
                       '. Request only if it can answer a specific visual question. ' if available else 'No further view requests allowed. Submit a reading now. ')
        messages.append(image_message(instruction+next_text, paths))
        dump(output/f'call_{step+1}_request_NO_GT.json', dict(messages=messages))
        try:
            attempted += 1
            response = caller(messages)
            dump(output/f'call_{step+1}_response.json', response)
            calls.append(dict(seconds=response['seconds'], usage=response['raw'].get('usage', {})))
            text = response['raw']['choices'][0]['message']['content']
            card = json.loads(text)
            validate(card, bool(available))
            history.append(card)
            messages.append(dict(role='assistant', content=text))
            if card['action'] == 'submit_reading':
                final = card
                break
            paths, receipt = views.render(card['view'])
            next_text = 'View tool result: '+json.dumps(receipt)+'. Read all views of the SAME candidate together.'
        except Exception as exc:
            dump(output/'failure.json', dict(step=step+1, type=type(exc).__name__, error=str(exc)))
            break
    valid = final is not None
    handoff = dict(candidate_id=task['candidate_id'], status='read' if valid else 'failed',
                   identity=final['identity'] if valid else 'uncertain',
                   p_glomerulus=final['p_glomerulus'] if valid else .5,
                   adequacy=final['adequacy'] if valid else 'insufficient',
                   differential=final['differential'] if valid else 'unknown',
                   review_required=not valid or final['identity']=='uncertain',
                   boundary_assessed=False, edit_authorized=False, api_calls=attempted,
                   completed_api_responses=len(calls),
                   view_requests=len(views.receipts)-1)
    dump(output/'specialist_reading.json', dict(final=final, history=history, costs=calls, gt_read=False))
    dump(output/'controller_handoff_NO_GT.json', handoff)
    return handoff
