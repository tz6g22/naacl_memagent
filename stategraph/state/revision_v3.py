"""Direct lifecycle decisions over canonical states; no cascade propagation."""

from __future__ import annotations

import re
from typing import Any, Mapping

from .canonical_state import normalize_identity_text


REVISION_SCHEMA_V3: dict[str, Any] = {
    'type': 'object',
    'additionalProperties': False,
    'properties': {
        'decisions': {
            'type': 'array',
            'items': {
                'type': 'object',
                'additionalProperties': False,
                'properties': {
                    'incoming_state_id': {'type': 'string'},
                    'existing_state_ids': {'type': 'array', 'items': {'type': 'string'}},
                    'relation': {
                        'type': 'string',
                        'enum': [
                            'NEW_FACT', 'CONSISTENT', 'DUPLICATE', 'UPDATE',
                            'EXPLICIT_CONFLICT', 'IMPLICIT_INVALIDATION',
                            'TEMPORARY_EXCEPTION', 'UNCERTAIN',
                        ],
                    },
                    'transition': {
                        'type': 'string',
                        'enum': [
                            'KEEP_CURRENT', 'KEEP_BOTH', 'MARK_HISTORICAL',
                            'MARK_STALE', 'SCOPED_OVERRIDE', 'MARK_UNCERTAIN',
                            'RECOMPUTE',
                        ],
                    },
                    'scope_text': {'type': ['string', 'null']},
                    'evidence_span': {'type': 'string'},
                    'reason': {'type': 'string'},
                    'confidence': {'type': 'number'},
                },
                'required': [
                    'incoming_state_id', 'existing_state_ids', 'relation',
                    'transition', 'scope_text', 'evidence_span', 'reason', 'confidence',
                ],
            },
        },
    },
    'required': ['decisions'],
}


REVISION_SYSTEM_V3 = '''Match canonical slots/functional keys; revise direct assertions only. Separate scopes coexist. Replacement: UPDATE/MARK_HISTORICAL; same-scope denial/broken guarantee: EXPLICIT_CONFLICT/MARK_STALE. Preserve outside TEMPORARY_EXCEPTION scope.
INPUT: existing_states, incoming_states, new_observation_text.
OUTPUT: decisions[] with incoming_state_id/existing_state_ids, relation/transition, scope_text/evidence_span, reason (<=12 words), confidence; one decision per incoming ID.
relation: NEW_FACT|CONSISTENT|DUPLICATE|UPDATE|EXPLICIT_CONFLICT|IMPLICIT_INVALIDATION|TEMPORARY_EXCEPTION|UNCERTAIN.
transition: KEEP_CURRENT|KEEP_BOTH|MARK_HISTORICAL|MARK_STALE|SCOPED_OVERRIDE|MARK_UNCERTAIN|RECOMPUTE.
No benchmark labels/answers.'''


def normalize_revision_output(
    response: Mapping[str, Any], *, incoming_states: list[Mapping[str, Any]],
    existing_states: list[Mapping[str, Any]], source_text: str,
) -> dict[str, Any]:
    raw = response.get('decisions')
    if not isinstance(raw, list):
        raise ValueError('revision output must contain a decisions array')
    incoming_by_id = {str(state['state_id']): state for state in incoming_states}
    existing_by_id = {str(state['state_id']): state for state in existing_states}
    normalized: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f'decision[{index}] must be an object')
        incoming_id = str(item.get('incoming_state_id') or '')
        if incoming_id not in incoming_by_id or incoming_id in normalized:
            raise ValueError(f'decision[{index}] has missing, unknown, or duplicate incoming ID')
        existing_ids = item.get('existing_state_ids')
        if not isinstance(existing_ids, list) or any(str(x) not in existing_by_id for x in existing_ids):
            raise ValueError(f'decision[{index}] references an unknown existing state')
        incoming = incoming_by_id[incoming_id]
        normalization_notes: list[str] = []
        relation = str(item.get('relation') or '')
        target_ids: list[str] = []
        for old_id in existing_ids:
            old = existing_by_id[str(old_id)]
            if old.get('slot_id') != incoming.get('slot_id'):
                old_keys = set(old.get('revision_keys') or [old.get('slot_id')])
                new_keys = set(incoming.get('revision_keys') or [incoming.get('slot_id')])
                if not old_keys.intersection(new_keys):
                    if relation == 'NEW_FACT':
                        normalization_notes.append('UNMATCHED_EXISTING_TARGET_REMOVED_FROM_NEW_FACT')
                        continue
                    old_entity = set(normalize_identity_text(old.get('entity')).split())
                    new_entity = set(normalize_identity_text(incoming.get('entity')).split())
                    explicitly_references_old_value = bool(
                        old.get('value') and str(old.get('value')).casefold() in source_text.casefold()
                    )
                    explicit_value_replacement = bool(
                        old.get('value') and incoming.get('value') and re.search(
                            r'\bfrom\s+' + re.escape(str(old['value']))
                            + r'.*?\bto\s+' + re.escape(str(incoming['value'])),
                            source_text, re.IGNORECASE,
                        )
                    )
                    grounded_slot_alias = (
                        (old.get('attribute_key') == incoming.get('attribute_key')
                         or explicit_value_replacement)
                        and bool(old_entity & new_entity)
                        and explicitly_references_old_value
                    )
                    if not grounded_slot_alias:
                        raise ValueError('direct revision may target only a shared canonical/functional key')
                    normalization_notes.append('EXPLICIT_VALUE_REFERENCE_SLOT_ALIAS')
            target_ids.append(str(old_id))
        transition = str(item.get('transition') or '')
        if relation == 'NEW_FACT' and target_ids:
            raise ValueError('NEW_FACT cannot target an existing state')
        if relation == 'NEW_FACT' and transition == 'KEEP_CURRENT':
            transition = 'KEEP_BOTH'
            normalization_notes.append('NEW_FACT_TRANSITION_NORMALIZED_TO_KEEP_BOTH')
        if relation == 'NEW_FACT' and transition != 'KEEP_BOTH':
            raise ValueError('NEW_FACT must use KEEP_BOTH transition')
        if relation == 'DUPLICATE' and transition not in {'KEEP_CURRENT', 'KEEP_BOTH'}:
            raise ValueError('DUPLICATE cannot destructively revise the existing state')
        scope_text = item.get('scope_text')
        if scope_text is not None and (not isinstance(scope_text, str) or scope_text not in source_text):
            # A duplicate that keeps both assertions does not use a revision
            # scope. Models sometimes repeat the older assertion's scope here;
            # discard this inapplicable field while keeping the source-grounded
            # evidence and the non-destructive decision intact.
            if relation == 'DUPLICATE' and transition in {'KEEP_BOTH', 'KEEP_CURRENT'}:
                scope_text = None
                normalization_notes.append('UNUSED_NONCURRENT_DUPLICATE_SCOPE_CLEARED')
            else:
                raise ValueError('revision scope is not an exact source phrase')
        evidence_span = str(item.get('evidence_span') or '')
        if not evidence_span or evidence_span not in source_text:
            raise ValueError('revision evidence must be an exact source substring')
        explicit_negation = bool(re.search(
            r'\b(?:not|no|never|cannot|can\s*not|can\x27t|unable|without|neither|nor)\b',
            source_text, re.IGNORECASE,
        ))
        if target_ids and incoming.get('polarity') == 'NEGATED' and explicit_negation and any(
            existing_by_id[old_id].get('polarity') == 'AFFIRMED'
            for old_id in target_ids
        ):
            if relation != 'EXPLICIT_CONFLICT' or transition != 'MARK_STALE':
                relation, transition = 'EXPLICIT_CONFLICT', 'MARK_STALE'
                normalization_notes.append('DIRECT_EXPLICIT_NEGATION_NORMALIZED_TO_CONFLICT')
        operation = (incoming.get('property_frame') or {}).get('operation')
        if (target_ids and (incoming.get('property_frame') or {}).get('assertion_kind') == 'SERVICE_RULE'
                and operation and operation['lifecycle'] == 'SUSPENDED'
                and re.search(r'\b(?:only|one[ -]day|one[ -]time)\b', str((incoming.get('temporal_scope') or {}).get('text') or '') + ' ' + str(incoming.get('condition_scope') or ''), re.I)):
            relation, transition = 'TEMPORARY_EXCEPTION', 'SCOPED_OVERRIDE'
            scope_text = (incoming.get('temporal_scope') or {}).get('text') or incoming.get('condition_scope')
            normalization_notes.append('BOUNDED_SERVICE_INTERRUPTION_PRESERVES_GENERAL_RULE')
        record_withdrawal = (
            target_ids and incoming.get('polarity') == 'AFFIRMED'
            and ((incoming.get('node_type') == 'FACT'
                  and normalize_identity_text(incoming.get('value')) in {'cancelled', 'canceled', 'withdrawn', 'rescinded'})
                 or (incoming.get('node_type') == 'ACTION'
                     and (incoming.get('event_frame') or {}).get('action_status') == 'CANCELLED'))
            and not re.search(r'\b(?:never|erroneously|mistakenly|not actually)\b', source_text, re.I)
            and all(existing_by_id[i].get('node_type') == 'FACT'
                    and (re.search(r'\b(?:booking|reservation)\b', normalize_identity_text(existing_by_id[i].get('attribute')))
                         or re.search(r'\b(?:booked|reserved)\b', normalize_identity_text(existing_by_id[i].get('value'))))
                    for i in target_ids)
        )
        if record_withdrawal and (relation != 'UPDATE' or transition != 'MARK_HISTORICAL'):
            # Withdrawing a previously valid reservation supersedes its version;
            # it does not assert that the booking had never been true.
            relation, transition = 'UPDATE', 'MARK_HISTORICAL'
            normalization_notes.append('FACT_RESERVATION_WITHDRAWAL_SUPERSEDES_PRIOR_VERSION')
        if relation == 'UPDATE' and transition == 'MARK_HISTORICAL' and scope_text and target_ids:
            if all(not (existing_by_id[old_id].get('temporal_scope') or {}).get('text')
                   for old_id in target_ids):
                transition = 'SCOPED_OVERRIDE'
                normalization_notes.append('FUTURE_SCOPED_UPDATE_PRESERVES_PRIOR_UNTIL_SCOPE')
        confidence = float(item.get('confidence'))
        if not 0 <= confidence <= 1:
            raise ValueError('revision confidence must be in [0, 1]')
        normalized[incoming_id] = {
            'incoming_state_id': incoming_id,
            'existing_state_ids': target_ids,
            'relation': relation,
            'transition': transition,
            'scope_text': scope_text,
            'evidence_span': evidence_span,
            'reason': str(item.get('reason') or ''),
            'confidence': confidence,
            'normalization_notes': normalization_notes,
        }
    missing = set(incoming_by_id) - set(normalized)
    if missing:
        raise ValueError(f'revision output omitted incoming states: {sorted(missing)}')
    return {
        'decisions': [normalized[state_id] for state_id in incoming_by_id],
        'untouched_existing_state_ids': sorted(
            set(existing_by_id) - {
                old_id for decision in normalized.values()
                for old_id in decision['existing_state_ids']
            }
        ),
    }


__all__ = ['REVISION_SCHEMA_V3', 'REVISION_SYSTEM_V3', 'normalize_revision_output']
