"""Canonical, source-grounded state mentions emitted by Module 1.

This representation separates a stable revision slot from one evidenced
assertion. Time and condition text is preserved without inventing dates.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any


def normalize_identity_text(value: Any) -> str:
    """Normalize typography and whitespace while preserving semantic tokens."""

    text = unicodedata.normalize('NFKC', str(value or '')).casefold()
    return ' '.join(re.findall(r'[^\W_]+', text, flags=re.UNICODE))


def normalize_temporal_identity(value: Any) -> str:
    """Collapse harmless grammatical variants without resolving the date."""

    text = unicodedata.normalize('NFKC', str(value or '')).casefold()
    text = re.sub(r"(?<=[a-z])'s\b", '', text)
    parts = normalize_identity_text(text).split()
    return ' '.join(token for token in parts if token not in {'on', 'at', 'in', 'the'})


def normalize_event_family(value: Any) -> str:
    """Return a conservative event head, excluding agenda/schedule document labels."""

    tokens = normalize_identity_text(value).split()
    document_terms = {'agenda', 'schedule', 'calendar', 'itinerary', 'plan'}
    tokens = [token for token in tokens if token not in document_terms]
    return tokens[-1] if tokens else ''


def _digest(prefix: str, values: tuple[str, ...]) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(',', ':'))
    return f'{prefix}:{hashlib.sha256(payload.encode("utf-8")).hexdigest()}'


def source_reference_keys(case_id: str, source_text: str) -> list[str]:
    """Extract stable, explicit object identifiers for equivalent mentions."""

    keys: set[str] = set()
    for match in re.finditer(r'\b([A-Z]{1,20})(\s*)(\d{2,5})\b', source_text, re.IGNORECASE):
        context = source_text[max(0, match.start() - 36):match.end() + 20].casefold()
        kind = 'code'
        # Whitespace-separated everyday words plus dates are not identifiers.
        if match.group(2) and (not match.group(1).isupper() or not 2 <= len(match.group(1)) <= 4):
            continue
        identifier = ''.join(match.group(1, 3)).casefold()
        keys.add(_digest('reference', (case_id, kind, identifier)))
    for match in re.finditer(r'#\s*(\d{2,})', source_text):
        context = source_text[max(0, match.start() - 36):match.start()].casefold()
        kind = next((label for label in ('order', 'invoice', 'ticket', 'case') if label in context), 'number')
        keys.add(_digest('reference', (case_id, kind, match.group(1))))
    return sorted(keys)


@dataclass(frozen=True, slots=True)
class SourceEvidence:
    observation_id: str
    source_text: str
    span_start: int
    span_end: int

    def __post_init__(self) -> None:
        if self.span_start < 0 or self.span_end <= self.span_start:
            raise ValueError('evidence span must be a non-empty half-open range')
        if self.span_end > len(self.source_text):
            raise ValueError('evidence span is outside its source observation')

    @property
    def text(self) -> str:
        return self.source_text[self.span_start:self.span_end]

    @property
    def evidence_id(self) -> str:
        return _digest('evidence', (
            self.observation_id, str(self.span_start), str(self.span_end), self.text,
        ))

    def to_dict(self) -> dict[str, Any]:
        return {
            'evidence_id': self.evidence_id,
            'observation_id': self.observation_id,
            'text': self.text,
            'span': {'start': self.span_start, 'end': self.span_end},
        }


@dataclass(frozen=True, slots=True)
class CanonicalState:
    """A source-grounded semantic node before revision/dependency decisions."""

    case_id: str
    entity: str
    attribute: str
    value: str
    polarity: str
    temporal_scope: dict[str, Any]
    condition_scope: str | None
    evidence: SourceEvidence
    node_type: str = 'FACT'
    event_frame: dict[str, Any] | None = None
    derivation_basis_text: str | None = None
    action_precondition_text: str | None = None
    confidence: float | None = None
    normalization_notes: tuple[str, ...] = ()
    claim_mode: str = 'DIRECT'
    source_status: str = 'CURRENT'
    claimant: str | None = None
    reported_content: dict[str, Any] | None = None
    source_event_frame: dict[str, Any] | None = None
    property_frame: dict[str, Any] | None = None

    @property
    def entity_key(self) -> str:
        parts = normalize_identity_text(self.entity).split()
        if parts and parts[0] in {'a', 'an', 'the'}:
            parts = parts[1:]
        return ' '.join(parts)

    @property
    def attribute_key(self) -> str:
        key = normalize_identity_text(self.attribute)
        # These nominal wrappers describe the value kind, not a distinct slot:
        # "renewal" and "renewal date" are the same revision target.
        key = re.sub(r'\s+(?:date|time|timestamp)$', '', key)
        # Normalize count assertions expressed either as "contains two items"
        # or as an explicit item_count slot.
        if key in {'contain', 'contains'} and re.search(
            r'\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+items?\b',
            self.value_key,
        ):
            return 'item count'
        if key in {'item count', 'items count', 'item counts'}:
            return 'item count'
        return key

    @property
    def value_key(self) -> str:
        return normalize_identity_text(self.value)

    @property
    def slot_id(self) -> str:
        # Time identifies the event/period being revised. Conditions remain
        # assertion metadata so a later correction can target the same slot.
        if self.claim_mode == 'ADVICE':
            # Distinct recommendations are communication records, not competing
            # values of the recipient's current state or accepted plan.
            return _digest('advice', (self.case_id, self.entity_key, self.value_key,
                                     self.evidence.observation_id))
        temporal_key = normalize_temporal_identity(self.temporal_scope.get('text'))
        return _digest('slot', (
            self.case_id, self.entity_key, self.attribute_key, temporal_key,
        ))

    @property
    def participation_slot_id(self) -> str | None:
        """Functional revision identity shared by roles/actions requiring participation."""

        frame = self.event_frame or {}
        if self.node_type not in {'ROLE_ASSERTION', 'ACTION'}:
            return None
        actor = normalize_identity_text(frame.get('actor'))
        event = normalize_event_family(frame.get('event'))
        if not actor or not event:
            return None
        # Time remains an explicit node scope. Event/actor identity lets a role
        # mention without a repeated timestamp align with the same condition.
        return _digest('participation', (self.case_id, actor, event))

    @property
    def revision_keys(self) -> list[str]:
        keys = [self.slot_id]
        if self.claim_mode == 'ADVICE':
            return keys
        # Property identity is distinct from an assertion's temporal version.
        # Revision must still determine whether scoped assertions conflict.
        if self.node_type in {'FACT', 'DERIVED_STATE', 'ACTION_PRECONDITION'}:
            keys.append(_digest('property', (self.case_id, self.entity_key, self.attribute_key)))
        if self.participation_slot_id is not None:
            keys.append(self.participation_slot_id)
        operation = (self.property_frame or {}).get('operation')
        if operation and (self.property_frame or {}).get('assertion_kind') == 'SERVICE_RULE' and operation.get('owner'):
            # Attribute/scope changes revise facets of the same named service.
            keys.append(_digest('operation', (self.case_id, normalize_identity_text(re.sub(r"['’]s\b", '', operation['owner'])),
                                             normalize_event_family(operation['name']))))
        keys.extend(self.reference_keys)
        return keys

    @property
    def reference_keys(self) -> list[str]:
        # References identify this assertion, not every object in its surrounding
        # sentence. In 'X cancelled; no replacement booked', the replacement
        # must not inherit X's identifier merely because both share evidence.
        frame = self.event_frame or {}
        identity_text = ' '.join(str(value) for value in (
            self.entity, self.value, frame.get('event'), frame.get('role'),
        ) if value)
        return source_reference_keys(self.case_id, identity_text)

    @property
    def state_id(self) -> str:
        return _digest('state', (
            self.slot_id, self.value_key, self.polarity.casefold(), self.evidence.evidence_id,
        ))

    def to_dict(self) -> dict[str, Any]:
        frame = dict(self.event_frame or {})
        event_family = normalize_event_family(frame.get('event'))
        if event_family:
            frame['event_family'] = event_family
        role_mode = frame.get('role_mode')
        participation_key = self.participation_slot_id
        if participation_key is not None:
            frame['participation_condition_id'] = participation_key
            frame['requires_participation_condition_id'] = (
                participation_key
                if role_mode == 'ASSIGNMENT'
                or (
                    self.node_type == 'ACTION'
                    and frame.get('action_status') in {'PENDING', 'IN_PROGRESS'}
                    and frame.get('participation_requirement') == 'REQUIRED'
                )
                else None
            )
            frame['supports_participation_condition_id'] = (
                participation_key
                if self.node_type == 'ROLE_ASSERTION'
                and role_mode in {'COMMITMENT', 'ACTUAL'}
                and self.polarity == 'AFFIRMED'
                else None
            )
        return {
            'state_id': self.state_id,
            'slot_id': self.slot_id,
            'revision_keys': self.revision_keys,
            'reference_keys': self.reference_keys,
            'case_id': self.case_id,
            'entity': self.entity,
            'entity_key': self.entity_key,
            'attribute': self.attribute,
            'attribute_key': self.attribute_key,
            'value': self.value,
            'value_key': self.value_key,
            'polarity': self.polarity,
            'temporal_scope': self.temporal_scope,
            'condition_scope': self.condition_scope,
            'node_type': self.node_type,
            'claim_mode': self.claim_mode,
            'claimant': self.claimant,
            'reported_content': self.reported_content,
            'source_event_frame': self.source_event_frame,
            'property_frame': self.property_frame,
            'source_status': self.source_status,
            'validity_target': (
                'Recorded recommendation by its speaker; not evidence of recipient adoption, execution or current factual truth of the proposed activity'
                if self.claim_mode == 'ADVICE'
                else 'Currently warranted communicated content, not the archival fact that it was announced'
                if self.claim_mode == 'REPORTED'
                else 'CURRENT execution eligibility and preconditions, not the archived record of an arrangement'
                if self.node_type == 'ACTION'
                else 'CURRENT event-role fulfillability under its participation requirements'
                if self.node_type == 'ROLE_ASSERTION'
                else 'CURRENT conclusion under its recorded input evidence'
                if self.node_type == 'DERIVED_STATE'
                else 'CURRENT truth of the recorded proposition in its explicit scope'
            ),
            'event_frame': frame,
            'action_description': self.evidence.text if self.node_type == 'ACTION' else None,
            'derivation_mode': (
                'ASSUMPTION' if self.node_type == 'DERIVED_STATE'
                and re.search(r'\bassumes?\b', self.evidence.text, re.IGNORECASE)
                else 'COMPUTATION' if self.node_type == 'DERIVED_STATE'
                and re.search(r'\b(?:computed|calculated|derived)\s+from\b', self.evidence.text, re.IGNORECASE)
                else 'UNSPECIFIED'
            ),
            'derivation_basis_text': self.derivation_basis_text,
            'action_precondition_text': self.action_precondition_text,
            'provenance': self.evidence.to_dict(),
            'confidence': self.confidence,
            'normalization_notes': list(self.normalization_notes),
        }


__all__ = [
    'CanonicalState', 'SourceEvidence', 'normalize_identity_text',
    'normalize_temporal_identity', 'normalize_event_family',
]
