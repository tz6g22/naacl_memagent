"""Counterfactual strength decisions for canonical, directed typed edges."""

from __future__ import annotations

from typing import Any, Mapping

VERIFICATION_SCHEMA_V3 = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'decision': {'type': 'string', 'enum': [
            'STRICT_DEPENDENCY', 'WEAK_DEPENDENCY', 'NO_DEPENDENCY',
        ]},
        'direction_supported': {'type': 'boolean'},
        'dependent_if_prerequisite_invalid': {
            'type': 'string', 'enum': ['FAILS', 'HOLDS', 'UNDETERMINED'],
        },
        'evidence_refs': {'type': 'array', 'maxItems': 3, 'items': {'type': 'string', 'enum': ['P', 'D', 'O']}},
        'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
        'reason': {'type': 'string', 'maxLength': 240},
    },
    'required': ['decision', 'direction_supported', 'dependent_if_prerequisite_invalid',
                 'evidence_refs', 'confidence', 'reason'],
}

_VERIFICATION_FIELD_ORDER = ('dependent_if_prerequisite_invalid', 'direction_supported',
                             'evidence_refs', 'decision', 'confidence', 'reason')
VERIFICATION_SCHEMA_V3['properties'] = {key: VERIFICATION_SCHEMA_V3['properties'][key] for key in _VERIFICATION_FIELD_ORDER}
VERIFICATION_SCHEMA_V3['required'] = list(_VERIFICATION_FIELD_ORDER)
# Complete alternatives avoid backend handling of partial anyOf schemas.
from copy import deepcopy
_VERIFICATION_BASE_SCHEMA = deepcopy(VERIFICATION_SCHEMA_V3)
VERIFICATION_SCHEMA_V3['anyOf'] = []
for decision, direction, outcomes in (
    ('STRICT_DEPENDENCY', True, ['FAILS']),
    ('WEAK_DEPENDENCY', True, ['HOLDS', 'UNDETERMINED']),
    ('NO_DEPENDENCY', False, ['HOLDS', 'UNDETERMINED']),
):
    branch = deepcopy(_VERIFICATION_BASE_SCHEMA)
    branch['properties']['decision'] = {'type': 'string', 'enum': [decision]}
    branch['properties']['direction_supported'] = {'type': 'boolean', 'enum': [direction]}
    branch['properties']['dependent_if_prerequisite_invalid']['enum'] = outcomes
    VERIFICATION_SCHEMA_V3['anyOf'].append(branch)

VERIFICATION_SYSTEM_V3 = '''Remove P as CURRENT warrant; hold records/other assertions fixed. Test D validity/eligibility. Independent support only; no circular/downstream warrants or invented requirements.
STRICT_DEPENDENCY: grounded bridge + FAILS; WEAK_DEPENDENCY: contributory bridge + HOLDS/UNDETERMINED; NO_DEPENDENCY: no bridge. FAILS=lost warrant; HOLDS=independent support; UNDETERMINED=reassess.
INPUT: tested_claims, validity_contract, condition_bridge, proposed_relation, verification_epoch, source_evidence, remaining_evidence/support_context, intervention.
OUTPUT: strength, outcome, P/D/O references (positive decisions cite P/D), reason <=120 characters; honor STRICT restrictions.
INFLUENCE: influence_present, component VALIDITY|TIMING|PURPOSE|METHOD|VALUE|NONE, references/reason.
Use exact schema fields: decision for strength, dependent_if_prerequisite_invalid for outcome, evidence_refs for references; also return direction_supported and confidence. The separate INFLUENCE call uses its own schema. Return complete JSON matching the supplied restricted schema.'''


def verification_request(edge: Mapping[str, Any], prerequisite: Mapping[str, Any],
                         dependent: Mapping[str, Any], source: Mapping[str, Any]) -> dict:
    """Expose canonical validity scopes without assigning dependency strength."""
    def validity_contract(node: Mapping[str, Any]) -> str:
        frame = node.get('event_frame') or {}
        if node['node_type'] == 'ROLE_ASSERTION' and frame.get('participation_requirement') == 'REQUIRED':
            return 'CURRENT event-role fulfillability under its recorded participation condition, not archival agenda text'
        if node.get('claim_mode') == 'REPORTED':
            return 'CURRENT validity of the communicated content, not the historical fact that the communication occurred'
        if node['node_type'] == 'ACTION':
            return 'CURRENT execution eligibility of the recorded action in its temporal and condition scope'
        if node['node_type'] == 'DERIVED_STATE' and node.get('derivation_mode') == 'ASSUMPTION':
            return 'Provisional forecast/conclusion: loss of an assumed influence requires reassessment, not proof that the conclusion is false'
        if node['node_type'] == 'DERIVED_STATE':
            return 'CURRENT validity of the derived conclusion under its recorded evidence basis and scope'
        return 'CURRENT truth of the recorded claim within its temporal and condition scope'

    pre_frame, dep_frame = prerequisite.get('event_frame') or {}, dependent.get('event_frame') or {}
    support = pre_frame.get('supports_participation_condition_id')
    required = dep_frame.get('requires_participation_condition_id')
    observation = source['new_observation']
    endpoint_observations = {
        prerequisite['provenance']['observation_id'], dependent['provenance']['observation_id'],
    }
    updated_epoch = observation['id'] in endpoint_observations
    history = source['history']
    other_evidence = [dict(item) for item in history if item['id'] not in endpoint_observations]
    verification_claims = {'P': {'kind': prerequisite['node_type'], 'entity': prerequisite['entity'],
                                'attribute': prerequisite['attribute'], 'value': prerequisite['value']},
                           'D': {'kind': dependent['node_type'], 'entity': dependent['entity'],
                                'attribute': dependent['attribute'], 'value': dependent['value']}}
    for ref, node in (('P', prerequisite), ('D', dependent)):
        if node.get('claim_mode') == 'REPORTED':
            verification_claims[ref] = {
                'kind': 'CURRENT_COMMUNICATED_PROPOSITION',
                'content': (node.get('reported_content') or {}).get('evidence_span')
                    or {key: node[key] for key in ('entity', 'attribute', 'value')},
                'content_source_span': (node.get('reported_content') or {}).get('evidence_span'),
                'publisher': node.get('claimant') or node['entity'],
                'source_text': node['provenance']['text'],
                'scope': node['temporal_scope'],
                'target': 'Truth/operational warrant of the communicated content, not whether the publisher once said it',
            }
    if support:
        verification_claims['P'] = {
            'predicate': 'CURRENT_PARTICIPATION_WARRANT', 'value': True,
            'kind': 'SUPPORTED_PARTICIPATION_CONDITION', 'condition_id': support,
            'actor': pre_frame.get('actor'), 'event': pre_frame.get('event'),
            'scope': prerequisite['temporal_scope'],
            'source_basis': 'Canonical event-role commitment supports participation',
        }
    if required:
        verification_claims['D'] = {
            'predicate': 'CURRENT_ROLE_FULFILLABILITY', 'requires_current_participation_warrant': True,
            'kind': 'CONDITIONED_ROLE_FULFILLABILITY', 'required_condition_id': required,
            'actor': dep_frame.get('actor'), 'event': dep_frame.get('event'),
            'role': dep_frame.get('role'), 'scope': dependent['temporal_scope'],
            'source_basis': 'Canonical assigned role requires participation',
        }
    if prerequisite['node_type'] == 'ACTION':
        verification_claims['P'] = {
            'kind': 'CURRENT_ACTIVITY_ELIGIBILITY',
            'predicate': 'WARRANTED_ACTIVITY_UNDER_RECORDED_PURPOSE',
            'description': prerequisite.get('action_description')
                or prerequisite['provenance']['text'],
            'action_status': pre_frame.get('action_status'), 'scope': prerequisite['temporal_scope'],
        }
    if dependent['node_type'] == 'ACTION':
        verification_claims['D'] = {
            'kind': 'ACTION_EXECUTION_ELIGIBILITY',
            'predicate': 'WARRANTED_EXECUTION_UNDER_RECORDED_PURPOSE',
            'not_tested': 'Persistence of scheduling record or physical ability to send/execute',
            'entity': dependent['entity'], 'intended_action': dependent.get('action_description') or dependent['provenance']['text'],
            'required_precondition': dependent.get('action_precondition_text')
                or dep_frame.get('action_precondition_text') or dep_frame.get('dependency_reference_text')
                or dependent.get('condition_scope'),
            'action_status': dep_frame.get('action_status'),
            'scope': dependent['temporal_scope'],
            'purpose_reference': dep_frame.get('purpose_reference_text'),
            'purpose_source_span': dep_frame.get('purpose_source_span'),
            'source_basis': ('Source-recorded action precondition; an instruction is not its own execution warrant'
                if dependent.get('action_precondition_text') or dep_frame.get('action_precondition_text') or dep_frame.get('dependency_reference_text') or dependent.get('condition_scope')
                else 'Source-recorded communication purpose: test the current warrant of the promoted/announced content'
                if dep_frame.get('purpose_reference_text')
                else 'Recorded action without an explicit precondition; establish necessity from source evidence, not from the proposed edge'),
        }
    if dependent['node_type'] == 'DERIVED_STATE':
        verification_claims['D']['derivation_basis'] = dependent.get('derivation_basis_text')
        verification_claims['D']['derivation_mode'] = dependent.get('derivation_mode', 'UNSPECIFIED')
        if dependent.get('derivation_mode') == 'ASSUMPTION':
            verification_claims['D'] = {
                'kind': 'PROVISIONAL_DERIVED_CONCLUSION', 'entity': dependent['entity'],
                'attribute': dependent['attribute'], 'recorded_assumption': dependent['value'],
                'assumed_input': dependent.get('derivation_basis_text'),
                'derivation_mode': 'ASSUMPTION',
                'intervention_outcome': 'Input loss makes the provisional conclusion uncertain; categorical falsity needs explicit necessity evidence',
            }
    return {
        'edge': {key: edge[key] for key in ('prerequisite', 'dependent', 'relation')},
        'prerequisite_node': dict(prerequisite), 'dependent_node': dict(dependent),
        'validity_contract': {'P': validity_contract(prerequisite), 'D': validity_contract(dependent)},
        'verification_claims': verification_claims,
        'counterfactual_contract': {
            'intervention': 'Remove invalid prerequisite as usable support; preserve independent evidence',
            'target': 'Warranted CURRENT validity or execution eligibility, not archival text or mere possibility',
        },
        'verification_epoch': 'POST_OBSERVATION' if updated_epoch else 'PRE_REVISION_HISTORY',
        'other_evidence': other_evidence,
        'condition_context': {
            'prerequisite_supports': support, 'dependent_requires': required,
            'same_canonical_condition': bool(support and support == required),
            'support_scope': prerequisite['temporal_scope'], 'requirement_scope': dependent['temporal_scope'],
        },
        'evidence': {'P': prerequisite['provenance']['text'],
                     'D': dependent['provenance']['text'],
                     'O': observation['text'] if updated_epoch else ''},
    }


def verification_model_request(request: Mapping[str, Any]) -> dict:
    """One unambiguous operative target per endpoint; keep all epoch evidence."""
    return {
        'tested_claims': request['verification_claims'],
        'validity_contract': request['validity_contract'],
        'condition_bridge': request['condition_context'],
        'proposed_relation': request['edge']['relation'],
        'verification_epoch': request['verification_epoch'],
        'source_evidence': request['evidence'],
        'remaining_evidence': request['other_evidence'],
        'remaining_evidence_support_context': request.get('remaining_evidence_support_context', []),
        'intervention': 'Test P as an invalid CURRENT warrant; preserve evidence records, not unconditional truth of quotations',
    }



INFLUENCE_SYSTEM = '''Assess whether changing P warrants reassessing D's source-stated validity, timing, purpose, method or value. Test contributory influence, not categorical necessity. Shared topic alone is insufficient; do not invent requirements.
INPUT: supplied source claims and support context.
OUTPUT: influence_present, influence_component VALIDITY|TIMING|PURPOSE|METHOD|VALUE|NONE, evidence_refs, reason. Positive decisions cite both P and D. Return complete JSON matching the supplied schema; keep reason short.'''
INFLUENCE_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'influence_present': {'type': 'boolean'},
        'influence_component': {'type': 'string', 'enum': ['VALIDITY','TIMING','PURPOSE','METHOD','VALUE','NONE']},
        'evidence_refs': {'type': 'array', 'maxItems': 2, 'items': {'type': 'string', 'enum': ['P','D']}},
        'reason': {'type': 'string', 'maxLength': 160},
    }, 'required': ['influence_present','influence_component','evidence_refs','reason'],
}
INFLUENCE_SCHEMA['anyOf'] = []
for present in (True, False):
    branch = deepcopy({key:value for key,value in INFLUENCE_SCHEMA.items() if key!='anyOf'})
    branch['properties']['influence_present']['enum'] = [present]
    branch['properties']['influence_component']['enum'] = ['VALIDITY','TIMING','PURPOSE','METHOD','VALUE'] if present else ['NONE']
    INFLUENCE_SCHEMA['anyOf'].append(branch)


def influence_model_request(request: Mapping[str, Any]) -> dict:
    """Isolate contributory influence from the later necessity contract."""
    return {
        'prerequisite_source': request['evidence']['P'],
        'dependent_source': request['evidence']['D'],
        'dependent_kind': request['dependent_node']['node_type'],
        'prerequisite_scope': request['condition_context']['support_scope'],
        'dependent_scope': request['condition_context']['requirement_scope'],
        'question': 'If the prerequisite claim changed, would its source-stated timing, purpose, method, value or validity warrant reassessment?',
    }


def needs_influence_assessment(request: Mapping[str, Any]) -> bool:
    claim = request['verification_claims']['D']
    return (claim.get('kind') == 'PROVISIONAL_DERIVED_CONCLUSION' or
            (claim.get('kind') == 'ACTION_EXECUTION_ELIGIBILITY'
             and not claim.get('required_precondition') and not claim.get('purpose_reference')))


def validate_influence(response: Mapping[str, Any]) -> None:
    if set(response) != set(INFLUENCE_SCHEMA['required']):
        raise ValueError('influence assessment schema mismatch')
    present = response['influence_present']
    if type(present) is not bool or not response['reason'].strip():
        raise ValueError('invalid influence decision/reason')
    if response['influence_component'] not in (['VALIDITY','TIMING','PURPOSE','METHOD','VALUE'] if present else ['NONE']):
        raise ValueError('inconsistent influence component')
    refs = response['evidence_refs']
    if not isinstance(refs,list) or any(ref not in {'P','D'} for ref in refs) or len(refs)!=len(set(refs)):
        raise ValueError('invalid influence evidence')
    if present and set(refs) != {'P','D'}:
        raise ValueError('positive influence lacks endpoint evidence')


def verification_response_schema(request: Mapping[str, Any], influence: Mapping[str, Any] | None = None) -> dict:
    """Decode only counterfactual outcomes permitted by the canonical claim contract.

    A provisional assumption records influence, not categorical necessity. Removing
    that influence can require reassessment; this schema never manufactures support.
    An action without a source-recorded condition likewise has unestablished
    necessity. NONE remains possible, and raw generated labels are preserved unchanged.
    """
    schema = deepcopy(VERIFICATION_SCHEMA_V3)
    provisional = request['verification_claims']['D'].get('kind') == 'PROVISIONAL_DERIVED_CONCLUSION'
    ungrounded_action_necessity = (
        request['verification_claims']['D'].get('kind') == 'ACTION_EXECUTION_ELIGIBILITY'
        and not request['verification_claims']['D'].get('required_precondition')
        and not request['verification_claims']['D'].get('purpose_reference')
    )
    # An inferred action association does not establish necessity. Decode a
    # conservative support decision when extraction saved no source condition;
    # preserve that raw decision instead of rewriting STRICT after generation.
    non_strict_contract = provisional or ungrounded_action_necessity
    if non_strict_contract:
        schema['anyOf'] = [branch for branch in schema['anyOf']
                           if branch['properties']['decision']['enum'] != ['STRICT_DEPENDENCY']]
    for obj in [schema, *schema['anyOf']]:
        obj['properties']['evidence_refs']['items']['enum'] = [
            ref for ref, text in request['evidence'].items() if text]
        if non_strict_contract:
            obj['properties']['dependent_if_prerequisite_invalid']['enum'] = ['HOLDS', 'UNDETERMINED']
            obj['properties']['decision']['enum'] = [label for label in obj['properties']['decision']['enum']
                                                    if label != 'STRICT_DEPENDENCY']
    if influence is not None:
        validate_influence(influence)
        direction = influence['influence_present']
        schema['anyOf'] = [branch for branch in schema['anyOf']
                           if branch['properties']['direction_supported']['enum'] == [direction]]
        labels = [branch['properties']['decision']['enum'][0] for branch in schema['anyOf']]
        schema['properties']['direction_supported']['enum'] = [direction]
        schema['properties']['decision']['enum'] = labels
        if direction and non_strict_contract:
            # Support exists, but no independent warrant/necessity has been
            # established at this stage. Alternative support is checked later.
            schema['properties']['dependent_if_prerequisite_invalid']['enum'] = ['UNDETERMINED']
            for branch in schema['anyOf']:
                branch['properties']['dependent_if_prerequisite_invalid']['enum'] = ['UNDETERMINED']
    return schema


def normalize_verification(
    response: Mapping[str, Any], request: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the response without turning malformed positive labels into negatives."""
    errors = []
    if set(response) != set(VERIFICATION_SCHEMA_V3['required']):
        errors.append('response keys do not match schema')
    decision = response.get('decision')
    counterfactual = response.get('dependent_if_prerequisite_invalid')
    direction = response.get('direction_supported')
    confidence = response.get('confidence')
    reason = response.get('reason')
    refs = response.get('evidence_refs')
    if decision not in {'STRICT_DEPENDENCY', 'WEAK_DEPENDENCY', 'NO_DEPENDENCY'}:
        errors.append('unknown decision')
    if counterfactual not in {'FAILS', 'HOLDS', 'UNDETERMINED'}:
        errors.append('unknown counterfactual outcome')
    if type(direction) is not bool:
        errors.append('direction_supported is not boolean')
    if type(confidence) not in (int, float) or not 0 <= confidence <= 1:
        errors.append('confidence outside [0,1]')
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 240:
        errors.append('reason must be nonempty and short')
    evidence = request['evidence']
    if not isinstance(refs, list) or any(ref not in evidence or not evidence[ref] for ref in refs):
        errors.append('invalid evidence references')
    elif decision in {'STRICT_DEPENDENCY', 'WEAK_DEPENDENCY'} and not {'P', 'D'}.issubset(refs):
        errors.append('positive decision lacks endpoint evidence references')
    if decision == 'STRICT_DEPENDENCY' and (direction is not True or counterfactual != 'FAILS'):
        errors.append('STRICT requires directional counterfactual necessity')
    if decision == 'WEAK_DEPENDENCY' and (direction is not True or counterfactual not in {'HOLDS', 'UNDETERMINED'}):
        errors.append('WEAK requires directional non-necessary or uncertain support')
    if decision == 'NO_DEPENDENCY' and direction is not False:
        errors.append('NO_DEPENDENCY cannot claim directional support')
    if decision == 'NO_DEPENDENCY' and counterfactual == 'FAILS':
        errors.append('NO_DEPENDENCY cannot claim counterfactual necessity')
    return {
        'prerequisite': request['edge']['prerequisite'],
        'dependent': request['edge']['dependent'],
        'relation': request['edge']['relation'],
        'raw_decision': decision,
        'normalized_decision': decision if decision in {
            'STRICT_DEPENDENCY', 'WEAK_DEPENDENCY', 'NO_DEPENDENCY',
        } else None,
        'validation': {'passed': not errors, 'errors': errors},
        'final_decision': decision if not errors else 'UNRESOLVED',
        'counterfactual': counterfactual, 'direction_supported': direction,
        'evidence_refs': refs, 'confidence': confidence, 'reason': reason,
        'raw_result': dict(response),
        'verification_claims': request['verification_claims'],
        'verification_epoch': request['verification_epoch'],
        'action_necessity_grounding': (
            'SOURCE_RECORDED_CONDITION' if request['verification_claims']['D'].get('required_precondition')
            else 'UNESTABLISHED_NECESSITY: test source-grounded directional influence separately'
        ) if request['verification_claims']['D'].get('kind') == 'ACTION_EXECUTION_ELIGIBILITY' else 'NOT_AN_ACTION_TEST',
        'counterfactual_contract': request['counterfactual_contract'],
    }
