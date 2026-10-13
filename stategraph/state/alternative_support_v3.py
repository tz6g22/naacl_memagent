"""Independent support selection with explicit dependency/provenance exclusions."""
from __future__ import annotations
from typing import Any, Mapping

ALTERNATIVE_SUPPORT_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'fully_supported': {'type': 'boolean'},
        'independence_established': {'type': 'boolean'},
        'support_ids': {'type': 'array', 'items': {'type': 'string'}},
        'reason': {'type': 'string', 'maxLength': 240},
    }, 'required': ['fully_supported', 'independence_established', 'support_ids', 'reason'],
}
_SUPPORT_FIELD_ORDER = ('support_ids', 'independence_established', 'fully_supported', 'reason')
ALTERNATIVE_SUPPORT_SCHEMA['properties'] = {key: ALTERNATIVE_SUPPORT_SCHEMA['properties'][key]
                                            for key in _SUPPORT_FIELD_ORDER}
ALTERNATIVE_SUPPORT_SCHEMA['required'] = list(_SUPPORT_FIELD_ORDER)

ALTERNATIVE_SUPPORT_SYSTEM = '''Find CURRENT independent evidence fully supporting target_claim in the same scope after failed-path removal; alternative derivations are allowed.
Exclude self, failed-path/dependent and historical evidence. Repetition/missing edges do not prove independence.
INPUT: target_claim, dependent, failed roots/path, blocked IDs, eligible_sources, excluded_sources.
OUTPUT: support_ids, independence_established, fully_supported, reason <=120 characters. Unsupported: false/[]; no invented evidence.'''


def downstream(starts: set[str], edges: list[Mapping[str, Any]]) -> set[str]:
    reachable = set(starts)
    pending = list(starts)
    outgoing: dict[str, list[str]] = {}
    for edge in edges:
        if edge.get('final_decision') not in {'STRICT_DEPENDENCY', 'WEAK_DEPENDENCY'}:
            continue
        outgoing.setdefault(str(edge['prerequisite']), []).append(str(edge['dependent']))
    while pending:
        node = pending.pop()
        for child in outgoing.get(node, []):
            if child not in reachable:
                reachable.add(child)
                pending.append(child)
    return reachable


def support_request(dependent_id: str, nodes: list[Mapping[str, Any]],
                    assessments: list[Mapping[str, Any]], failed_roots: set[str],
                    statuses: Mapping[str, str]) -> dict:
    by_id = {str(node['state_id']): node for node in nodes}
    if dependent_id not in by_id or any(root not in by_id for root in failed_roots):
        raise ValueError('unknown dependent or root')
    failed_path = downstream(failed_roots, assessments)
    self_support_path = downstream({dependent_id}, assessments)
    blocked = failed_path | self_support_path
    eligible, excluded = [], []
    for node_id, node in by_id.items():
        reason = None
        if node_id == dependent_id:
            reason = 'dependent cannot support itself'
        elif node_id in failed_path:
            reason = 'candidate belongs to the failed dependency path'
        elif node_id in self_support_path:
            reason = 'candidate depends on the dependent'
        elif statuses.get(node_id, node.get('source_status', 'CURRENT')) != 'CURRENT':
            reason = 'candidate is not CURRENT'
        if reason:
            excluded.append({'state_id': node_id, 'reason': reason})
        else:
            eligible.append(dict(node))
    dependent = by_id[dependent_id]
    target = {key: dependent.get(key) for key in
              ('node_type', 'entity', 'attribute', 'value', 'polarity', 'temporal_scope', 'condition_scope')}
    if dependent.get('claim_mode') == 'REPORTED':
        target['communicated_content'] = dependent.get('reported_content')
    if dependent['node_type'] == 'ACTION':
        target['required_precondition'] = dependent.get('action_precondition_text') or (dependent.get('event_frame') or {}).get('action_precondition_text')
        target['validity_target'] = 'Execution eligibility under the recorded purpose, not physical performability'
    return {'target_claim': target, 'dependent': dict(dependent), 'failed_roots': sorted(failed_roots),
            'failed_path': sorted(failed_path),
            'failed_root_nodes': [dict(by_id[root]) for root in sorted(failed_roots)], 'blocked_support_ids': sorted(blocked),
            'eligible_sources': eligible, 'excluded_sources': excluded,
            'support_target': 'Exact dependent claim or its required condition, not topical overlap'}


def normalize_support(response: Mapping[str, Any], request: Mapping[str, Any]) -> dict:
    errors = []
    supported = response.get('fully_supported')
    ids = response.get('support_ids')
    eligible = {node['state_id'] for node in request['eligible_sources']}
    if type(supported) is not bool:
        errors.append('fully_supported must be boolean')
    if not isinstance(ids, list) or any(not isinstance(x, str) or x not in eligible for x in ids):
        errors.append('support IDs must be eligible independent nodes')
    elif len(ids) != len(set(ids)):
        errors.append('duplicate support IDs')
    if type(response.get('independence_established')) is not bool:
        errors.append('independence_established must be boolean')
    if supported is True and response.get('independence_established') is not True:
        errors.append('full support requires affirmative independence')
    if supported is True and not ids:
        errors.append('full support requires evidence')
    if supported is False and ids:
        errors.append('unsupported decision cannot list full supports')
    reason = response.get('reason')
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 240:
        errors.append('reason must be short and nonempty')
    return {'dependent': request['dependent']['state_id'], 'raw_result': dict(response),
            'validation': {'passed': not errors, 'errors': errors},
            'fully_supported': supported if not errors else None,
            'support_ids': ids if not errors else [],
            'decision': ('INDEPENDENT_SUPPORT' if supported else 'NO_INDEPENDENT_SUPPORT') if not errors else 'UNRESOLVED',
            'reason': reason, 'excluded_sources': request['excluded_sources']}
