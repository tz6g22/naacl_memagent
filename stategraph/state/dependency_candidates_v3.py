"""High-recall, source-grounded dependency candidate discovery."""

from __future__ import annotations

from typing import Any, Mapping

from stategraph.state.canonical_state import normalize_identity_text


CANDIDATE_SCHEMA_V3: dict[str, Any] = {
    'type': 'object',
    'additionalProperties': False,
    'properties': {
        'pairs': {
            'type': 'array',
            'items': {
                'type': 'object',
                'additionalProperties': False,
                'properties': {
                    'state_id_a': {'type': 'string'},
                    'state_id_b': {'type': 'string'},
                    'reason': {'type': 'string', 'maxLength': 120},
                },
                'required': ['state_id_a', 'state_id_b', 'reason'],
            },
        },
    },
    'required': ['pairs'],
}


CANDIDATE_SYSTEM_V3 = '''Propose unordered support/derivation/precondition pairs; no direction/type/strength or topic-only links. Supplied IDs and pair limit only; reason <=8 words; no gold labels.
INPUT: {"states": [...]}
OUTPUT: {"pairs":[{"state_id_a":"<ID>","state_id_b":"<ID>","reason":"<reason>"}]}'''


def _tokens(value: str) -> set[str]:
    aliases = {'monthly': 'month', 'daily': 'day', 'yearly': 'year', 'annual': 'year'}
    return {
        aliases.get(token, token)
        for token in normalize_identity_text(value).split()
        if len(token) > 2 and token not in {
            'the', 'and', 'for', 'from', 'with', 'that', 'this', 'will', 'was',
            'are', 'has', 'have', 'about', 'into', 'their', 'they', 'then',
        }
    }


def structural_candidate_pairs(states: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Add narrow cross-role identity and shared event/value anchors."""

    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for index, left in enumerate(states):
        left_id = str(left['state_id'])
        left_entity = str(left.get('entity_key') or normalize_identity_text(left.get('entity')))
        left_value = str(left.get('value_key') or normalize_identity_text(left.get('value')))
        for right in states[index + 1:]:
            right_id = str(right['state_id'])
            right_entity = str(right.get('entity_key') or normalize_identity_text(right.get('entity')))
            right_value = str(right.get('value_key') or normalize_identity_text(right.get('value')))
            entity_value_anchor = bool(
                left_entity and right_value and left_entity == right_value
                or right_entity and left_value and right_entity == left_value
            )
            shared_entity = bool(left_entity and left_entity == right_entity)
            shared_event = bool(
                _tokens(str(left.get('value', ''))) & _tokens(str(right.get('value', '')))
                or _tokens(str((left.get('event_frame') or {}).get('event') or ''))
                & _tokens(str((right.get('event_frame') or {}).get('event') or ''))
            )
            derived_endpoint = 'DERIVED_STATE' in {
                str(left.get('node_type') or 'FACT'), str(right.get('node_type') or 'FACT'),
            }
            left_semantics = _tokens(' '.join(str(left.get(key, '')) for key in (
                'entity', 'attribute', 'value',
            )))
            right_semantics = _tokens(' '.join(str(right.get(key, '')) for key in (
                'entity', 'attribute', 'value',
            )))
            shared_derived_concepts = derived_endpoint and len(left_semantics & right_semantics) >= 2
            left_basis = str(
                left.get('derivation_basis_text') or left.get('action_precondition_text') or (left.get('event_frame') or {}).get('dependency_reference_text') or ''
            )
            right_basis = str(
                right.get('derivation_basis_text') or right.get('action_precondition_text') or (right.get('event_frame') or {}).get('dependency_reference_text') or ''
            )

            def reference_matches(reference: str, node: Mapping[str, Any]) -> bool:
                reference_tokens = _tokens(reference)
                concept_tokens = _tokens(' '.join(str(node.get(key, '')) for key in (
                    'entity', 'attribute', 'value', 'evidence',
                )))
                evidence = str(node.get('provenance', {}).get('text') or '')
                concept_tokens |= _tokens(evidence)
                # Discovery is a proposal, not verification: an explicitly named
                # input can have a different canonical label. Preserve anchored
                # partial matches for the semantic typer/verifier to decide.
                return bool(reference_tokens & concept_tokens)

            explicit_reference = (
                left.get('node_type') in {'DERIVED_STATE', 'ACTION', 'CONDITION'}
                and bool(left_basis)
                and reference_matches(left_basis, right)
            ) or (
                right.get('node_type') in {'DERIVED_STATE', 'ACTION', 'CONDITION'}
                and bool(right_basis)
                and reference_matches(right_basis, left)
            )
            shared_entity_reference = bool(
                set(left.get('reference_keys') or []) & set(right.get('reference_keys') or [])
            )
            sources = []
            if entity_value_anchor or (shared_entity and shared_event) or shared_derived_concepts:
                sources.append('structural_anchor')
            if explicit_reference:
                sources.append('explicit_source_reference')
            if shared_entity_reference:
                sources.append('shared_entity_reference')
            if sources:
                key = tuple(sorted((left_id, right_id)))
                pairs[key] = {
                    'state_ids': list(key),
                    'reason': (
                        'Explicit source derivation/precondition reference'
                        if 'explicit_source_reference' in sources
                        else 'Shared explicit object identifier'
                        if 'shared_entity_reference' in sources and 'structural_anchor' not in sources
                        else 'Shared entity and event anchor'
                    ),
                    'candidate_sources': sources,
                }
    return [pairs[key] for key in sorted(pairs)]


def normalize_candidate_output(
    response: Mapping[str, Any], *, states: list[Mapping[str, Any]],
    excluded_direct_pairs: set[frozenset[str]] | None = None,
) -> dict[str, Any]:
    raw_pairs = response.get('pairs')
    if not isinstance(raw_pairs, list):
        raise ValueError('candidate output must contain a pairs array')
    by_id = {str(state['state_id']): state for state in states}
    excluded_direct_pairs = excluded_direct_pairs or set()
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    rejected: list[dict[str, str]] = []
    for index, item in enumerate(raw_pairs):
        if not isinstance(item, Mapping):
            rejected.append({'index': str(index), 'reason': 'not an object'})
            continue
        a, b = str(item.get('state_id_a') or ''), str(item.get('state_id_b') or '')
        reason = str(item.get('reason') or '').strip()
        if a not in by_id or b not in by_id or a == b:
            rejected.append({'index': str(index), 'reason': 'unknown endpoint or self pair'})
            continue
        if not reason or len(reason.split()) > 12:
            rejected.append({'index': str(index), 'reason': 'missing or overlong short reason'})
            continue
        key = tuple(sorted((a, b)))
        if frozenset(key) in excluded_direct_pairs:
            rejected.append({'index': str(index), 'reason': 'direct revision pair is not a dependency candidate'})
            continue
        if key in merged:
            continue
        merged[key] = {'state_ids': list(key), 'reason': reason, 'candidate_sources': ['model_proposal']}

    for item in structural_candidate_pairs(states):
        key = tuple(item['state_ids'])
        if frozenset(key) in excluded_direct_pairs:
            continue
        if key in merged:
            merged[key]['candidate_sources'].append('structural_anchor')
        else:
            merged[key] = item

    observation_order: dict[str, int] = {}
    for state in states:
        observation_order.setdefault(
            str(state.get('provenance', {}).get('observation_id')), len(observation_order),
        )
    candidates = []
    for key in sorted(merged):
        candidate = merged[key]
        nodes = {state_id: by_id[state_id] for state_id in key}
        ordered = sorted(key, key=lambda sid: (
            observation_order.get(str(nodes[sid].get('provenance', {}).get('observation_id')), 10**9), sid,
        ))
        order_a = observation_order.get(str(nodes[ordered[0]].get('provenance', {}).get('observation_id')))
        order_b = observation_order.get(str(nodes[ordered[1]].get('provenance', {}).get('observation_id')))
        candidate.update({
            'endpoint_node_types': {sid: nodes[sid].get('node_type', 'FACT') for sid in key},
            'endpoint_frames': {sid: nodes[sid].get('event_frame') for sid in key},
            'evidence_ids': [nodes[sid].get('provenance', {}).get('evidence_id') for sid in key],
            'observation_order_hint': {
                'earlier_observation': ordered[0], 'later_observation': ordered[1],
                'basis': 'SOURCE_OBSERVATION_ORDER_ONLY_NOT_DEPENDENCY_DIRECTION',
            } if order_a != order_b else None,
        })
        candidates.append(candidate)
    return {
        'candidates': candidates,
        'candidate_count': len(candidates),
        'state_count': len(states),
        'rejected_model_proposals': rejected,
        'complete_response_required': True,
    }


__all__ = ['CANDIDATE_SCHEMA_V3', 'CANDIDATE_SYSTEM_V3', 'normalize_candidate_output', 'structural_candidate_pairs']
