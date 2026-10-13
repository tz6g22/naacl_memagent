"""Relation typing with explicit prerequisite-to-dependent direction checks."""

from __future__ import annotations

import re
from typing import Any, Mapping

from .canonical_state import normalize_identity_text, source_reference_keys


RELATION_TYPING_SCHEMA_V6 = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'dependent': {'type': 'string', 'enum': ['A', 'B', 'NONE']},
        'prerequisite': {'type': 'string', 'enum': ['A', 'B', 'NONE']},
        'relation': {'type': 'string', 'enum': ['DEPENDS_ON', 'DERIVED_FROM', 'AFFECTS_ACTION', 'NONE']},
        'prerequisite_removed_dependent_loses_support': {'type': 'boolean'},
        'dependent_removed_prerequisite_loses_support': {'type': 'boolean'},
        'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
        'evidence_refs': {'type': 'array', 'maxItems': 2, 'items': {'type': 'string', 'enum': ['A', 'B']}},
        'reason': {'type': 'string', 'maxLength': 180},
    },
    'required': ['dependent', 'prerequisite', 'relation',
                 'prerequisite_removed_dependent_loses_support',
                 'dependent_removed_prerequisite_loses_support',
                 'confidence', 'evidence_refs', 'reason'],
}
RELATION_TYPING_SYSTEM_V6 = '''Store prerequisite -> dependent. DEPENDS_ON: support; DERIVED_FROM: named input; AFFECTS_ACTION: pending-action prerequisite. Respect types/claims and participation IDs (COMMITMENT supplies; ASSIGNMENT requires).
Hold other assertions fixed; test both directions. Neither: NONE; both: ambiguous. No topical/transitive links or usefulness-only invalidation.
INPUT: nodes {A,B}, source_semantic_links, candidate_provenance.
OUTPUT: prerequisite/dependent A|B|NONE, relation DEPENDS_ON|DERIVED_FROM|AFFECTS_ACTION|NONE, counterfactual booleans, confidence, <=2 A/B references, reason.
Use the exact schema fields prerequisite_removed_dependent_loses_support, dependent_removed_prerequisite_loses_support and evidence_refs. Ambiguity is not a relation label: preserve both directional test results; the runtime handles ambiguous decisions. Return complete schema JSON.'''


def _tokens(value: Any) -> set[str]:
    tokens = set(normalize_identity_text(value).split())
    aliases = {
        'monthly': 'month', 'daily': 'day', 'yearly': 'year',
        'annual': 'year', 'calculated': 'calculate', 'computed': 'compute',
        'derived': 'derive',
    }
    return {aliases.get(token, token) for token in tokens if token not in {
        'the', 'a', 'an', 'of', 'from', 'on', 'in', 'for', 'to', 'and', 'with',
    }}


def _reference_matches_node(reference: str, node: Mapping[str, Any], *, allow_identifier_anchor: bool = False) -> bool:
    """Match an explicit source reference against a canonical endpoint concept."""

    ref = _tokens(reference)
    if not ref:
        return False
    if allow_identifier_anchor:
        provenance = node.get('provenance') or {}
        evidence_text = str(provenance.get('evidence') or provenance.get('text') or '')
        # A precondition can name an asset's event (e.g. arrival) while its
        # prerequisite records that asset's booking/configuration. This is only
        # an endpoint-grounding guard, never a structural dependency decision.
        if set(source_reference_keys('', reference)).intersection(source_reference_keys('', evidence_text)):
            return True
    entity_attribute = _tokens(' '.join(str(node.get(key) or '') for key in ('entity', 'attribute')))
    head = next(iter(reversed(normalize_identity_text(reference).split())), '')
    aliases = {'monthly': 'month', 'daily': 'day', 'yearly': 'year', 'annual': 'year'}
    head = aliases.get(head, head)
    if head not in entity_attribute and not (
        node.get('node_type') == 'ACTION' and head in _tokens(node.get('action_description'))
    ):
        return False
    concept = _tokens(' '.join(str(node.get(key) or '') for key in (
        'entity', 'attribute', 'value',
    )))
    provenance = node.get('provenance') or {}
    evidence_text = str(provenance.get('evidence') or provenance.get('text') or '')
    evidence = _tokens(evidence_text)
    # A unit may be expressed grammatically in evidence ("per month"). Other
    # reference terms must be part of the canonical entity/attribute/value.
    missing = ref - concept
    unit_aliases = {'month': {'month', 'monthly'}, 'day': {'day', 'daily'}, 'year': {'year', 'annual'}}
    if node.get('node_type') == 'ACTION':
        # Explicit named input + activity head grounds nominal descriptions
        # such as a campaign email versus an email advertising that same code.
        identifiers = {token for token in ref if any(char.isdigit() for char in token)}
        if identifiers and identifiers <= evidence and head in evidence:
            return True
    return all(
        token in evidence or any(alias in evidence for alias in unit_aliases.get(token, set()))
        for token in missing
    )


def _scope_disjoint(prerequisite: Mapping[str, Any], dependent: Mapping[str, Any]) -> bool:
    pre_scope = str((prerequisite.get('temporal_scope') or {}).get('text') or '').casefold()
    dep_scope = str((dependent.get('temporal_scope') or {}).get('text') or '').casefold()
    future_prerequisite = any(term in pre_scope for term in ('next ', 'starting ', 'from next'))
    current_dependent = any(term in dep_scope for term in ('current', 'this year', 'this month', 'today'))
    return future_prerequisite and current_dependent


def _source_order_impossible(
    prerequisite_id: str, dependent_id: str, dependent: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> bool:
    """Reject a source that postdates the derivation it is claimed to support."""

    order = candidate.get('observation_order_hint') or {}
    return bool(
        order.get('later_observation') == prerequisite_id
        and order.get('earlier_observation') == dependent_id
        and dependent.get('node_type') == 'DERIVED_STATE'
    )


def _canonical_relation(
    prerequisite: Mapping[str, Any], dependent: Mapping[str, Any],
) -> tuple[str, str] | None:
    """Return a relation only when canonical schema carries explicit semantics."""

    pre_frame = prerequisite.get('event_frame') or {}
    dep_frame = dependent.get('event_frame') or {}
    support = pre_frame.get('supports_participation_condition_id')
    required = dep_frame.get('requires_participation_condition_id')
    if support and support == required:
        relation = 'AFFECTS_ACTION' if (
            dependent.get('node_type') == 'ACTION'
            and dep_frame.get('action_status') in {'PENDING', 'IN_PROGRESS'}
        ) else 'DEPENDS_ON'
        return relation, 'Explicit participation support satisfies the dependent role precondition.'

    if dependent.get('node_type') == 'DERIVED_STATE' and not _scope_disjoint(prerequisite, dependent):
        basis = dependent.get('derivation_basis_text')
        if isinstance(basis, str) and _reference_matches_node(basis, prerequisite):
            return 'DERIVED_FROM', 'The dependent conclusion explicitly names this source as its derivation basis.'

    if (
        dependent.get('node_type') == 'ACTION'
        and dep_frame.get('action_status') in {'PENDING', 'IN_PROGRESS'}
    ):
        basis = dependent.get('action_precondition_text') or dep_frame.get('action_precondition_text')
        if isinstance(basis, str) and _reference_matches_node(basis, prerequisite):
            return 'AFFECTS_ACTION', 'The pending action explicitly names this prerequisite in its source evidence.'
    return None


def explicit_support_links(candidate: Mapping[str, Any], nodes: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Expose existing canonical support semantics, without reading benchmark labels."""
    a, b = candidate['state_ids']
    links = []
    for pre, dep, pre_ref, dep_ref in ((a,b,'A','B'), (b,a,'B','A')):
        if _scope_disjoint(nodes[pre], nodes[dep]) or _source_order_impossible(pre, dep, nodes[dep], candidate):
            continue
        canonical = _canonical_relation(nodes[pre], nodes[dep])
        if canonical:
            relation, evidence = canonical
            links.append({'prerequisite': pre_ref, 'dependent': dep_ref,
                          'schema_relation': relation, 'source_semantics': evidence})
    return links


def _valid_type(
    relation: str, prerequisite: Mapping[str, Any], dependent: Mapping[str, Any],
) -> bool:
    if relation == 'DERIVED_FROM':
        if dependent.get('node_type') != 'DERIVED_STATE':
            return False
        basis = dependent.get('derivation_basis_text')
        # Endpoint meaning is classified by the saved two-direction semantic
        # decision. A lexical head check must not veto synonyms or packaging
        # differences (the counterfactual verifier still checks dependence).
        return not basis or str(basis) in str(dependent.get('provenance',{}).get('text') or dependent.get('provenance',{}).get('evidence') or '')
    if relation == 'AFFECTS_ACTION':
        frame = dependent.get('event_frame') or {}
        if dependent.get('node_type') != 'ACTION' or frame.get('action_status') not in {'PENDING','IN_PROGRESS'}:
            return False
        basis = dependent.get('action_precondition_text') or frame.get('action_precondition_text')
        return not basis or _reference_matches_node(str(basis), prerequisite, allow_identifier_anchor=True)
    if (relation == 'DEPENDS_ON' and dependent.get('node_type') in {'FACT', 'CONDITION'}
            and prerequisite.get('node_type') in {'ROLE_ASSERTION', 'ACTION'}):
        # A role/action can support execution or a derivation, but does not change
        # a separately asserted resource fact without an explicit validity condition.
        condition = dependent.get('condition_scope')
        if not condition or not _reference_matches_node(str(condition), prerequisite, allow_identifier_anchor=True):
            return False
    if relation == 'DEPENDS_ON' and dependent.get('node_type') == 'ROLE_ASSERTION':
        frame = dependent.get('event_frame') or {}
        pre_frame = prerequisite.get('event_frame') or {}
        participation = pre_frame.get('supports_participation_condition_id')
        required = frame.get('requires_participation_condition_id')
        condition = dependent.get('condition_scope')
        actor = normalize_identity_text(frame.get('actor'))
        actor_fact = prerequisite.get('node_type') in {'FACT','CONDITION','ACTION_PRECONDITION'} and actor and normalize_identity_text(prerequisite.get('entity')) == actor
        event_fact = prerequisite.get('node_type') in {'FACT','CONDITION'} and frame.get('event') and _reference_matches_node(str(frame['event']), prerequisite)
        explicit_condition = condition and _reference_matches_node(str(condition), prerequisite, allow_identifier_anchor=True)
        # A context fact mentioning the same event cannot establish an actor's
        # participation/role eligibility. Require a grounded condition bridge.
        if not (participation and participation == required or actor_fact or event_fact or explicit_condition):
            return False
    if relation == 'DEPENDS_ON' and dependent.get('node_type') == 'DERIVED_STATE':
        basis = dependent.get('derivation_basis_text')
        if basis and not _reference_matches_node(str(basis), prerequisite):
            condition = dependent.get('condition_scope')
            return bool(condition and _reference_matches_node(str(condition), prerequisite))
    return relation == 'DEPENDS_ON'


def _normalize_orientation_checks(
    response: Mapping[str, Any], *, candidate: Mapping[str, Any],
    nodes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate both orientations and emit a strict directed edge or rejection."""

    a, b = candidate['state_ids']
    if a not in nodes or b not in nodes:
        raise ValueError('candidate endpoint does not exist')
    order = candidate.get('observation_order_hint')
    if order is not None and (
        not isinstance(order, Mapping)
        or {order.get('earlier_observation'), order.get('later_observation')} != {a, b}
    ):
        raise ValueError('source-order hint must identify the two candidate endpoints')
    orientations = (
        (a, b, response.get('a_to_b')),
        (b, a, response.get('b_to_a')),
    )
    valid: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for prerequisite_id, dependent_id, raw in orientations:
        if not isinstance(raw, Mapping):
            raise ValueError('both direction checks are required')
        relation = str(raw.get('relation') or '')
        loses_validity = raw.get('dependent_loses_validity_if_prerequisite_removed')
        confidence = raw.get('confidence')
        reason = str(raw.get('reason') or '').strip()
        refs = raw.get('evidence_refs')
        if relation not in {'DEPENDS_ON', 'DERIVED_FROM', 'AFFECTS_ACTION', 'NONE'}:
            raise ValueError('unknown relation label')
        if not isinstance(loses_validity, bool):
            raise ValueError('counterfactual result must be boolean')
        if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            raise ValueError('confidence must be within [0,1]')
        if not isinstance(refs, list) or any(ref not in {'A', 'B'} for ref in refs):
            raise ValueError('evidence_refs must contain only A/B endpoint references')
        if not reason or len(reason) > 180:
            raise ValueError('reason must satisfy the nonempty maxLength=180 schema')

        pre_node, dep_node = nodes[prerequisite_id], nodes[dependent_id]
        source_order_impossible = _source_order_impossible(
            prerequisite_id, dependent_id, dep_node, candidate,
        )
        if _scope_disjoint(pre_node, dep_node) or source_order_impossible:
            diagnostics.append({
                'prerequisite': prerequisite_id, 'dependent': dependent_id,
                'relation': 'NONE', 'counterfactual': False,
                'rejected': (
                    'source observation postdates the dependent derivation'
                    if source_order_impossible
                    else 'temporal scopes do not overlap'
                ),
            })
            continue
        canonical = _canonical_relation(pre_node, dep_node)
        type_mapping = None
        if canonical is not None and relation != 'NONE' and loses_validity:
            relation, canonical_reason = canonical
            loses_validity = True
            reason = canonical_reason
            confidence = 1.0
            confidence_basis = 'EXPLICIT_CANONICAL_SEMANTICS'
        else:
            confidence_basis = 'MODEL_COUNTERFACTUAL'

        dep_frame = dep_node.get('event_frame') or {}
        if (
            relation == 'DEPENDS_ON'
            and dep_node.get('node_type') == 'ACTION'
            and dep_frame.get('action_status') in {'PENDING', 'IN_PROGRESS'}
        ):
            relation = 'AFFECTS_ACTION'
            type_mapping = 'PENDING_ACTION_ENDPOINT_REQUIRES_AFFECTS_ACTION'

        if relation == 'NONE' or not loses_validity:
            diagnostics.append({
                'prerequisite': prerequisite_id, 'dependent': dependent_id,
                'relation': 'NONE', 'counterfactual': False,
            })
            continue
        if not _valid_type(relation, pre_node, dep_node):
            diagnostics.append({
                'prerequisite': prerequisite_id, 'dependent': dependent_id,
                'relation': relation, 'counterfactual': bool(loses_validity),
                'rejected': 'relation is incompatible with dependent type or explicit input/precondition reference',
            })
            continue
        evidence_ids = [
            str((nodes[state_id].get('provenance') or {}).get('evidence_id'))
            for state_id in (prerequisite_id, dependent_id)
        ]
        valid.append({
            'prerequisite': prerequisite_id,
            'dependent': dependent_id,
            'relation': relation,
            'confidence': float(confidence),
            'evidence_ids': evidence_ids,
            'reason': reason,
            '_confidence_basis': confidence_basis,
            '_type_mapping': type_mapping,
        })
        diagnostics.append({
            'prerequisite': prerequisite_id, 'dependent': dependent_id,
            'relation': relation, 'counterfactual': True,
        })

    if len(valid) == 1:
        edge = valid[0]
        basis = edge.pop('_confidence_basis')
        type_mapping = edge.pop('_type_mapping')
        return {
            'status': 'ACCEPTED', 'edge': edge,
            'direction_check': 'ONE_WAY_COUNTERFACTUAL_SUPPORT',
            'confidence_basis': basis, 'type_mapping': type_mapping,
            'directions': diagnostics,
        }
    if len(valid) > 1:
        return {
            'status': 'AMBIGUOUS', 'candidate_edges': [
                {key: value for key, value in edge.items() if not key.startswith('_')}
                for edge in valid
            ],
            'direction_check': 'BOTH_DIRECTIONS_SUPPORTED', 'directions': diagnostics,
        }
    return {
        'status': 'REJECTED', 'candidate_edges': [],
        'direction_check': 'NO_VALID_ONE_WAY_DEPENDENCY', 'directions': diagnostics,
    }


def normalize_relation_decision(response: Mapping[str, Any], *, candidate: Mapping[str, Any],
                                nodes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Decode explicit semantic endpoints; never infer direction from reason text."""
    if set(response) != set(RELATION_TYPING_SCHEMA_V6['required']):
        raise ValueError('relation response must match the explicit-endpoint V6 schema')
    confidence = response['confidence']
    refs = response['evidence_refs']
    reason = response['reason']
    if type(confidence) not in (int, float) or not 0 <= confidence <= 1:
        raise ValueError('confidence must be within [0,1]')
    if not isinstance(refs, list) or len(refs) > 2 or any(ref not in {'A', 'B'} for ref in refs):
        raise ValueError('at most two endpoint evidence references are allowed')
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 180:
        raise ValueError('reason must be nonempty and at most 180 characters')
    if response['relation'] not in {'DEPENDS_ON', 'DERIVED_FROM', 'AFFECTS_ACTION', 'NONE'}:
        raise ValueError('unknown relation label')
    pre, dep = response['prerequisite'], response['dependent']
    relation = response['relation']
    forward = response['prerequisite_removed_dependent_loses_support']
    reverse = response['dependent_removed_prerequisite_loses_support']
    if type(forward) is not bool or type(reverse) is not bool:
        raise ValueError('direction checks must be boolean')
    if relation == 'NONE' or pre == 'NONE' or dep == 'NONE':
        if not (relation == 'NONE' and not forward and not reverse and (pre == dep == 'NONE' or {pre, dep} == {'A', 'B'})):
            raise ValueError('inconsistent NONE decision')
        return {'status': 'REJECTED', 'candidate_edges': [], 'direction_check': 'NO_DIRECTIONAL_SUPPORT', 'raw_result': dict(response)}
    if pre not in {'A','B'} or dep not in {'A','B'} or pre == dep:
        raise ValueError('endpoints must be different A/B references')
    if forward and reverse:
        return {'status': 'AMBIGUOUS', 'candidate_edges': [], 'direction_check': 'BOTH_DIRECTIONS_SUPPORTED', 'raw_result': dict(response)}
    corrected = False
    if reverse and not forward:
        pre, dep = dep, pre
        forward = True
        corrected = True
    if not forward:
        return {'status': 'REJECTED', 'candidate_edges': [], 'direction_check': 'NO_DIRECTIONAL_SUPPORT', 'raw_result': dict(response)}
    selected = dict(relation=relation, dependent_loses_validity_if_prerequisite_removed=True,
                    confidence=response['confidence'], evidence_refs=response['evidence_refs'], reason=response['reason'])
    rejected = dict(relation='NONE', dependent_loses_validity_if_prerequisite_removed=False,
                    confidence=0.0, evidence_refs=[], reason='No reverse dependency asserted.')
    checks = {'a_to_b': selected if pre == 'A' else rejected,
              'b_to_a': selected if pre == 'B' else rejected}
    result = _normalize_orientation_checks(checks, candidate=candidate, nodes=nodes)
    result['raw_result'] = dict(response)
    result['explicit_direction_correction'] = corrected
    return result


__all__ = [
    'RELATION_TYPING_SCHEMA_V6', 'RELATION_TYPING_SYSTEM_V6',
    'normalize_relation_decision',
]
