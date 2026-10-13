"""Operational evidence selection and original-action gating over revised state status."""
from __future__ import annotations
from typing import Any, Mapping
import re
from .cascade_propagation_v3 import scope_overlap

QUERY_SCHEMA={'type':'object','additionalProperties':False,'properties':{
 'intent':{'type':'string','enum':['CURRENT_QUERY','HISTORICAL_QUERY','ACTION_REQUEST']},
 'requested_ids':{'type':'array','items':{'type':'string'}},
 'premise_ids':{'type':'array','items':{'type':'string'}},
 'action_ids':{'type':'array','items':{'type':'string'}},
 'reason':{'type':'string','maxLength':160}},
 'required':['intent','requested_ids','premise_ids','action_ids','reason']}
INTENT_SCHEMA={'type':'object','additionalProperties':False,'properties':{
 'intent':{'type':'string','enum':['CURRENT_QUERY','HISTORICAL_QUERY','ACTION_REQUEST']},
 'reason':{'type':'string','maxLength':120}},'required':['intent','reason']}
INTENT_SYSTEM='''From query wording choose CURRENT_QUERY|HISTORICAL_QUERY|ACTION_REQUEST; future use of old plans is not historical.
INPUT: query. OUTPUT: intent/reason.'''
QUERY_SYSTEM='''Select supplied requested_ids (asked about), premise_ids (presupposed CURRENT), action_ids (requested actions); respect intent without guessing.
INPUT: query, intent, typed/scoped nodes with aliases/evidence.
OUTPUT: requested_ids, premise_ids, action_ids, reason <100 characters.
Code checks dependency/status/scope/action eligibility; no gate prompt.'''
ANSWER_SYSTEM='''Use CURRENT evidence within applicability_scopes; historical evidence only for historical queries. Correct stale premises; obey original_action_execution_allowed; no blocked execution/invented resources.
CURRENT inputs and verified_recomputation_results govern values/units. Sufficient inputs: COMPUTATION; otherwise REASSESSMENT_REQUIRED/null. Reassess downweighted claims; missing confirmation is not unavailability.
INPUT: query, runtime_gate, verified_recomputation_results, evidence_aliases.
OUTPUT: answer, action_decision NO_ACTION|EXECUTE|REVISE_PLAN|BLOCK|CLARIFY, CURRENT used_evidence_refs, <=5 replanned_actions; respect runtime restrictions.
REVISE_PLAN proposes grounded alternatives; BLOCK only blocks; CLARIFY requests inputs. No execution.
Recomputation: COMPUTABLE|INSUFFICIENT_INPUT, expression/input_refs/reason; restricted decimal arithmetic.
Return only the supplied answer schema fields; recomputation is handled by a separate call and runtime.'''
ANSWER_SCHEMA={'type':'object','additionalProperties':False,'properties':{
 'answer':{'type':'string','minLength':1,'maxLength':1800},
 'action_decision':{'type':'string','enum':['NO_ACTION','EXECUTE','REVISE_PLAN','BLOCK','CLARIFY']},
 'used_evidence_refs':{'type':'array','items':{'type':'string'}},
 'replanned_actions':{'type':'array','maxItems':5,'items':{'type':'string','maxLength':240}}},
 'required':['answer','action_decision','used_evidence_refs','replanned_actions']}



def literal_query_scope(query_text: str) -> dict:
    """Only explicit ISO dates/times; unknown relative times remain unresolved."""
    pattern = r'(?<![\w-])\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})?)?(?![\w-])'
    dates = re.findall(pattern, query_text)
    if len(dates) not in (1, 2):
        return {}
    if re.search(r'\b(?:before|after|since|until|not on)\b', query_text, re.I):
        return {}
    if len(dates) == 2 and not re.search(r'\b(?:from|between)\b', query_text, re.I):
        return {}
    return {'start': dates[0], 'end': dates[-1], 'source': 'literal query span'}


def gate(nodes:list[Mapping[str,Any]],cascade:Mapping[str,Any],query:Mapping[str,Any],dependencies:list[Mapping[str,Any]]|None=None,query_text:str="")->dict:
 by_id={n['state_id']:n for n in nodes};statuses=cascade['statuses']
 if query.get('intent') not in {'CURRENT_QUERY','HISTORICAL_QUERY','ACTION_REQUEST'}:raise ValueError('unknown query intent')
 for field in ('requested_ids','premise_ids','action_ids'):
  ids=query.get(field)
  if not isinstance(ids,list) or len(ids)!=len(set(ids)) or any(i not in by_id for i in ids):raise ValueError('invalid query references: '+field)
 if any(by_id[i]['node_type']!='ACTION' for i in query['action_ids']):raise ValueError('action references must identify actual action nodes')
 relevant=set(query['requested_ids']+query['premise_ids']+query['action_ids'])
 # Required evidence follows the persisted prerequisite -> dependent direction.
 while True:
  expanded=relevant|{e['prerequisite'] for e in dependencies or []
                     if e['dependent'] in relevant and e['final_decision'] in {'STRICT_DEPENDENCY','WEAK_DEPENDENCY'}}
  if expanded==relevant:break
  relevant=expanded
 downweighted=set(cascade['downweighted_state_ids']);needs_recompute=set(cascade['needs_recompute'])
 query_scope=literal_query_scope(query_text)
 conditional={x['dependent'] for x in cascade['scoped_impacts']
              if x['effect']!='KEEP' and scope_overlap(x['scope_bounds'],query_scope)!='DISJOINT'}
 conditional|={x['state_id'] for x in cascade['scoped_overrides']
               if scope_overlap(x['scope_bounds'],query_scope)!='DISJOINT'}
 # A status question can mention both an archived action and a pending action.
 # Only live requested actions provide a target scope. A prerequisite can be
 # supplied for that target when every scoped override is provably disjoint,
 # without making the prerequisite globally/unconditionally usable.
 live_targets=[by_id[i] for i in dict.fromkeys(query['requested_ids']+query['action_ids'])
               if by_id[i]['node_type']=='ACTION' and statuses[i]=='CURRENT'
               and by_id[i].get('event_frame',{}).get('action_status') in {'PENDING','IN_PROGRESS'}]
 scoped_usable={}
 if query['intent']=='CURRENT_QUERY' and not query_scope and live_targets:
  target_scopes=[n.get('temporal_scope',{}) for n in live_targets]
  for root_id in conditional:
   bounds=[x['scope_bounds'] for x in cascade['scoped_overrides'] if x['state_id']==root_id]
   descendants={root_id}
   while True:
    expanded=descendants|{e['dependent'] for e in dependencies or []
                         if e['prerequisite'] in descendants and e['final_decision'] in {'STRICT_DEPENDENCY','WEAK_DEPENDENCY'}}
    if expanded==descendants:break
    descendants=expanded
   if (root_id not in query['requested_ids'] and bounds
       and all(n['state_id'] in descendants for n in live_targets) and all(
       scope_overlap(bound,target)=='DISJOINT' for bound in bounds for target in target_scopes)):
    scoped_usable[root_id]=target_scopes
 conditional-=scoped_usable.keys()
 # Applicability is separate from lifecycle status. Preserve dated CURRENT
 # assertions as context when their validity scope is disjoint from every
 # explicit target scope. Unknown overlap stays unresolved, never excluded.
 target_scopes=([query_scope] if query_scope else
                [n.get('temporal_scope',{}) for n in live_targets]
                if query['intent']=='CURRENT_QUERY' else [])
 out_of_scope=[]
 if target_scopes:
  for n in nodes:
   if statuses[n['state_id']]=='CURRENT' and all(
      scope_overlap(n.get('temporal_scope',{}),target)=='DISJOINT'
      for target in target_scopes):
    out_of_scope.append({'state':dict(n),'status':'CURRENT',
                         'reason':'VALIDITY_SCOPE_DISJOINT_FROM_TARGET',
                         'target_scopes':target_scopes})
 outside_ids={item['state']['state_id'] for item in out_of_scope}
 usable=[]
 for n in nodes:
  if statuses[n['state_id']]=='CURRENT' and n['state_id'] not in downweighted|conditional|outside_ids:
   item=dict(n)
   if n['state_id'] in scoped_usable:item['applicability_scopes']=scoped_usable[n['state_id']]
   usable.append(item)
 rejected=[{'state':dict(n),'status':statuses[n['state_id']]} for n in nodes if statuses[n['state_id']]!='CURRENT']
 historical=[dict(n) for n in nodes if query['intent']=='HISTORICAL_QUERY' and statuses[n['state_id']]=='HISTORICAL' and n['state_id'] in relevant]
 # Historical context being mentioned is not itself a current premise.
 premise_refs=set(query['premise_ids'])
 if query['intent']=='ACTION_REQUEST':premise_refs|=relevant
 stale=sorted(i for i in relevant if statuses[i]=='STALE' or
              (statuses[i]=='HISTORICAL' and i in premise_refs)) if query['intent']!='HISTORICAL_QUERY' else []
 uncertain=[i for i in relevant if statuses[i] in {'UNCERTAIN','RECOMPUTE'} or i in downweighted|conditional]
 action_refs=set(query['action_ids'])|{i for i in relevant if by_id[i]['node_type']=='ACTION'}
 blocked=sorted(i for i in action_refs if statuses[i]!='CURRENT' or by_id[i].get('polarity')=='NEGATED' or (by_id[i].get('event_frame') or {}).get('action_status')=='CANCELLED' or i in downweighted|conditional)
 advice_only=bool(relevant) and all(by_id[i].get('claim_mode')=='ADVICE' for i in relevant)
 unsafe_request=query['intent']=='ACTION_REQUEST' and (bool(stale or uncertain) or not relevant or advice_only)
 has_action_request=query['intent']=='ACTION_REQUEST'
 has_action_context=bool(action_refs) or has_action_request
 execution_allowed=has_action_request and not blocked and not unsafe_request and query['intent']!='HISTORICAL_QUERY'
 policy='reject_stale_premise' if stale else ('clarify' if uncertain or (query['intent']=='ACTION_REQUEST' and advice_only) else 'proceed')
 return {'response_policy':policy,'usable_current_evidence':usable,'rejected_noncurrent_context':rejected,
  'historical_context':historical,'stale_premise_ids':stale,'reassessment_ids':sorted(uncertain),
  'blocked_original_action_ids':blocked,'original_action_execution_allowed':execution_allowed,
  'conditional_validity':cascade['scoped_impacts'],'query_scope':query_scope,
  'target_scopes':target_scopes,'out_of_scope_current_context':out_of_scope,
  'scope_qualified_evidence_ids':sorted(scoped_usable),
  'recomputation_obligations':[{'state':dict(by_id[i]),'status':statuses[i],
                               'instruction':'Reassess the recorded derivation using CURRENT independent inputs'}
                              for i in sorted(needs_recompute & relevant)],
  'allowed_action_decisions':(['NO_ACTION'] if query['intent']=='HISTORICAL_QUERY' or not has_action_context else
                             ['NO_ACTION','REVISE_PLAN','BLOCK','CLARIFY'] if not execution_allowed else
                             ['NO_ACTION','EXECUTE','REVISE_PLAN','BLOCK','CLARIFY'])}
