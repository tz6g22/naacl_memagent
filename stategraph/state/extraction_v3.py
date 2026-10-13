"""Single-pass, observation-local canonical state extraction."""

from __future__ import annotations

import json
import copy
import re
from typing import Any, Mapping

from .canonical_state import CanonicalState, SourceEvidence, normalize_identity_text


EXTRACTION_SCHEMA_V3: dict[str, Any] = {
    'type': 'object',
    'additionalProperties': False,
    'properties': {
        'states': {
            'type': 'array',
            'items': {
                'type': 'object',
                'additionalProperties': False,
                'properties': {
                    'node_type': {
                        'type': 'string',
                        'enum': ['FACT', 'CONDITION', 'ROLE_ASSERTION', 'DERIVED_STATE', 'ACTION', 'ACTION_PRECONDITION'],
                    },
                    'entity': {'type': 'string'},
                    'attribute': {'type': 'string'},
                    'value': {'type': 'string'},
                    'polarity': {'type': 'string', 'enum': ['AFFIRMED', 'NEGATED', 'UNKNOWN']},
                    'temporal_scope': {
                        'type': 'object',
                        'additionalProperties': False,
                        'properties': {
                            'kind': {'type': 'string', 'enum': ['NONE', 'TEXT', 'ABSOLUTE']},
                            'text': {'type': ['string', 'null']},
                            'start': {'type': ['string', 'null']},
                            'end': {'type': ['string', 'null']},
                        },
                        'required': ['kind', 'text', 'start', 'end'],
                    },
                    'reported_content': {
                        'type': ['object', 'null'], 'additionalProperties': False,
                        'properties': {
                            'entity': {'type': 'string'}, 'attribute': {'type': 'string'},
                            'value': {'type': 'string'}, 'evidence_span': {'type': 'string'},
                        }, 'required': ['entity', 'attribute', 'value', 'evidence_span'],
                    },
                    'claimant': {'type': ['string', 'null']},
                    'claim_mode': {'type': 'string', 'enum': ['DIRECT', 'REPORTED', 'ASSUMPTION', 'INTENT', 'ADVICE']},
                    'source_status': {'type': 'string', 'enum': ['CURRENT', 'HISTORICAL', 'UNCERTAIN']},
                    'condition_scope': {'type': ['string', 'null']},
                    'derivation_basis_text': {'type': ['string', 'null']},
                    'action_precondition_text': {'type': ['string', 'null']},
                    'event_frame': {
                        'type': 'object',
                        'additionalProperties': False,
                        'properties': {
                            'actor': {'type': ['string', 'null']},
                            'event': {'type': ['string', 'null']},
                            'role': {'type': ['string', 'null']},
                            'action_status': {
                                'type': 'string',
                                'enum': ['NONE', 'PENDING', 'IN_PROGRESS', 'COMPLETED', 'CANCELLED', 'UNKNOWN'],
                            },
                            'participation_requirement': {
                                'type': 'string', 'enum': ['REQUIRED', 'NOT_APPLICABLE', 'UNKNOWN'],
                            },
                        },
                        'required': ['actor', 'event', 'role', 'action_status', 'participation_requirement'],
                    },
                    'evidence_span': {'type': 'string'},
                    'confidence': {'type': ['number', 'null']},
                },
                'required': [
                    'node_type', 'entity', 'attribute', 'value', 'polarity', 'temporal_scope',
                    'condition_scope', 'derivation_basis_text', 'action_precondition_text',
                    'event_frame', 'evidence_span', 'confidence', 'claim_mode', 'source_status', 'claimant', 'reported_content',
                ],
            },
        },
    },
    'required': ['states'],
}


EXTRACTION_SYSTEM_V3 = '''Extract atomic source assertions with exact evidence; no invented facts/actions/revisions. Preserve entity/value, scope, provenance and event roles. Copy explicit derivation/action inputs; separate reported_content/claimant.
CURRENT by default; HISTORICAL/UNCERTAIN require evidence. Negation preserves positive content; absolute dates require a year; speakers require attribution.
INPUT: case_id, observation_id, observation_text; optional speaker_spans, normalization_feedback.
OUTPUT: states[] with entity/attribute/value, scope/event_frame/evidence and:
node_type: FACT|CONDITION|ROLE_ASSERTION|DERIVED_STATE|ACTION|ACTION_PRECONDITION.
claim_mode: DIRECT|REPORTED|ASSUMPTION|INTENT|ADVICE.
polarity: AFFIRMED|NEGATED|UNKNOWN; time: NONE|TEXT|ABSOLUTE.
PROPERTY_IDENTITY/REPORTED_PROPOSITION provide source-grounded projections.
Return JSON matching the supplied schema, including every required field. Scope/time refers to temporal_scope and condition_scope; evidence refers to evidence_span; CURRENT/HISTORICAL/UNCERTAIN refers to source_status.'''



PROPERTY_IDENTITY_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'assertion_kind': {'type': 'string', 'enum': ['PROPERTY', 'CONFIGURATION_SETPOINT', 'SERVICE_RULE', 'ACTION_RECORD', 'CONTEXT']},
        'operation': {'type': ['object', 'null'], 'additionalProperties': False,
            'properties': {'name': {'type': 'string'}, 'owner': {'type': ['string', 'null']},
                'lifecycle': {'type': 'string', 'enum': ['ACTIVE','PENDING','IN_PROGRESS','CANCELLED','ARCHIVED','SUSPENDED']}},
            'required': ['name','owner','lifecycle']},
        'change_agent': {'type': ['string', 'null']},
        'controlled_object': {'type': ['string', 'null']},
        **{key: {'type': 'string'} for key in ('property_owner', 'attribute', 'value', 'evidence_span')},
    },
    'required': ['assertion_kind', 'operation', 'change_agent', 'controlled_object', 'property_owner', 'attribute', 'value', 'evidence_span'],
}
_service_projection = copy.deepcopy(PROPERTY_IDENTITY_SCHEMA)
_service_projection['properties']['assertion_kind']['enum'] = ['SERVICE_RULE']
_service_projection['properties']['operation']['type'] = 'object'
_service_projection['properties']['operation']['properties']['owner']['type'] = 'string'
_action_projection = copy.deepcopy(PROPERTY_IDENTITY_SCHEMA)
_action_projection['properties']['assertion_kind']['enum'] = ['ACTION_RECORD']
_action_projection['properties']['operation']['type'] = 'object'
_other_projection = copy.deepcopy(PROPERTY_IDENTITY_SCHEMA)
_other_projection['properties']['assertion_kind']['enum'] = ['PROPERTY', 'CONFIGURATION_SETPOINT', 'CONTEXT']
_other_projection['properties']['operation']['type'] = 'null'
PROPERTY_IDENTITY_SCHEMA = {'anyOf': [_service_projection, _action_projection, _other_projection]}
PROPERTY_IDENTITY_SYSTEM = """Separate the semantic roles of one source property assertion.
Identify the assertion using property_hint/value_hint. Hints may contain a mistaken subject;
use source semantics. Return source-grounded roles, not the grammatical subject by default.
SERVICE_RULE is a recurring operational service schedule, or a bounded interruption of
that same service. operation.name is its source-named activity, operation.owner is its
affected source-named person/site/service area (not the provider or changer), lifecycle ACTIVE or SUSPENDED.
When an area is explicitly named, owner MUST be that area, not null. operation.name is
only the activity phrase: omit its area, weekdays and dates, which belong in owner/time.
Example: 'Library deliveries on North Avenue run every Tuesday' -> name 'Library deliveries',
owner 'North Avenue', ACTIVE. 'North Avenue deliveries suspended on 5 May only' -> name
'deliveries', owner 'North Avenue', SUSPENDED. Do not put the area into operation.name. Property names such
as frequency/status are facets of the same operative service, not different services.
Do not classify prices, rate units, discounts or device profiles as service schedules.
ACTION_RECORD is a named task/reminder/communication already scheduled or explicitly retired.
It is one operational record even if source says both cancelled and archived. Its operation
name identifies the task; owner is a source-stated responsible actor or null (never invented).
Use ARCHIVED if explicitly archived, CANCELLED if only cancelled, PENDING if scheduled.
Passive resource bookings/reservations and ordinary financial properties remain PROPERTY.
A nonexistent, unbooked or unconfirmed replacement is a PROPERTY absence assertion,
not a scheduled or retired ACTION_RECORD. Do not invent a task or lifecycle for it.
An original resource and an unconfirmed replacement are distinct subjects. In 'the original
reservation was withdrawn and no alternative was confirmed', withdrawal belongs to the
original reservation; nonconfirmation belongs to the alternative. Do not assign the
alternative's missing booking/confirmation to the original resource. Use the exact
replacement/alternative phrase as property_owner when that is the asserted subject.
CONTEXT is only chronology/background qualifying another assertion, not an independent world
state: 'the inspection had already ended before the outage' does not separately assert an
outage's current timing/status. Keep the substantive ended inspection as ACTION_RECORD;
the 'before the outage' relation is provenance/context. Do not discard distinct actual facts.
operation is non-null only for SERVICE_RULE and ACTION_RECORD; otherwise null.
PROPERTY means a directly asserted object property. CONFIGURATION_SETPOINT means a
profile/scene/rule/configuration sets a target for another object, not a measured physical fact.
For CONFIGURATION_SETPOINT, property_owner is the profile/scene/rule. controlled_object
is the distinct target device/object only when explicitly named, otherwise null. Never set it
to the configuration owner itself. A property of a profile is not automatically a setpoint:
'The quiet profile runs at sunrise' is PROPERTY, owner quiet profile, attribute execution time,
value sunrise, controlled_object null. Scheduling, activation and counts are ordinary properties.
For PROPERTY, controlled_object is null.
change_agent is a person/organization explicitly changing a property, otherwise null.
A change agent is not the property_owner just because it is the sentence subject.
Example: 'An administrator raised the subscription fee to $25' has change_agent administrator,
property_owner subscription fee, attribute amount, value $25.
Example: 'The quiet profile sets the fan to 900 rpm' has property_owner quiet profile,
controlled_object fan, attribute fan speed, value 900 rpm.
Example: 'The fan measures 900 rpm' is PROPERTY, owner fan, controlled_object null.
property_owner excludes the attribute: 'account X status changed to locked' has owner account X,
attribute status, value locked. Do not repeat status in the owner.
All roles must be exact source phrases; typography may normalize. Do not invent objects.
Attribute is a short semantic property name; value keeps the full asserted meaningful amount/content.
Evidence span is an exact contiguous source span supporting the owner/property/value.
Do not emit benchmark labels, expected answers or revisions. Return only the schema JSON."""



REPORTED_PROPOSITION_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {key: {'type': 'string'} for key in ('entity', 'attribute', 'value', 'evidence_span')},
    'required': ['entity', 'attribute', 'value', 'evidence_span'],
}
REPORTED_PROPOSITION_SYSTEM = """Extract the proposition communicated in this observation,
not the fact that a publisher communicated it. Entity is the named object/account/product/code
whose property is communicated. Attribute names that property. Value is its stated condition/amount.
A numeric price/percentage is a value, not an entity. The reporting publisher is not the
proposition subject unless the communicated content actually concerns the publisher.
Evidence span is the exact content clause from source, without the reporting subject/verb.
Example: "A notice says account X is locked" => entity "account X", attribute "status",
value "locked", evidence_span "account X is locked". Do not invent any fact. Return JSON only."""

def _exact_span(source: str, span: str) -> tuple[int, int]:
    if not span or source.count(span) != 1:
        raise ValueError('evidence_span must occur exactly once in the observation')
    start = source.index(span)
    return start, start + len(span)


_TEMPORAL_MARKERS = re.compile(
    r'\b(?:now|today|tomorrow|yesterday|tonight|morning|afternoon|evening|'
    r'monday|tuesday|wednesday|thursday|friday|saturday|sunday|'
    r'january|february|march|april|may|june|july|august|september|october|'
    r'november|december|next|last|this|until|through|between|before|after|'
    r'during|ongoing|hourly|nightly|weekly|monthly|daily|yearly|fortnightly|every|'
    r'am|pm|year|month|week|day|hour|minute|sunrise|sunset|noon|midnight)\b|'
    r'\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}:\d{2}\b', re.IGNORECASE,
)


def _expand_evidence_to_sentence(source: str, start: int, end: int) -> tuple[int, int]:
    left = max(source.rfind(mark, 0, start) for mark in '.!?\n') + 1
    right_candidates = [source.find(mark, end) for mark in '.!?\n']
    right_candidates = [position for position in right_candidates if position >= 0]
    right = min(right_candidates) + 1 if right_candidates else len(source)
    while left < right and source[left].isspace():
        left += 1
    while right > left and source[right - 1].isspace():
        right -= 1
    return left, right


def _expand_structured_heading_to_evidence(
    source: str, start: int, end: int, entity: str,
) -> tuple[int, int]:
    """Retain a source heading that supplies the subject of its evidence item."""
    line_start = source.rfind('\n', 0, start) + 1
    current_line = source[line_start:source.find('\n', end) if source.find('\n', end) >= 0 else len(source)]
    previous_end = line_start
    while previous_end > 0:
        previous_end -= 1
        previous_start = source.rfind('\n', 0, previous_end) + 1
        previous_line = source[previous_start:previous_end].strip()
        if previous_line:
            break
        if line_start - previous_start > 4:
            return start, end
    else:
        return start, end
    match = re.match(r'^\*\*(?P<label>.+?)(?::\*\*|\*\*:)\s*$', previous_line)
    if not match and re.match(r'^\s*\+\s+Example\s*:', current_line, re.IGNORECASE):
        match = re.match(r'^(?:[*-])\s+(?P<label>.+?)\s*:\s*$', previous_line)
    if not match or normalize_identity_text(match.group('label')) != normalize_identity_text(entity):
        return start, end
    return previous_start, end


_WEEKDAY_TIME = re.compile(
    r"^(?P<day>on\s+)?(?P<weekday>monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"(?P<possessive>'s)?\s+(?:at\s+)?(?P<clock>\d{1,2}(?::\d{2})?\s*(?:am|pm))"
    r"(?:\s+(?P<tail>.+))?$", re.IGNORECASE,
)
_NOMINAL_ASSIGNMENT = re.compile(
    r'\blists?\s+(?P<person>[\w .\'-]+?)\s+as\s+'
    r'(?P<role>[\w -]+?)\s+of\s+(?P<object>[^,.!?]+)',
    re.IGNORECASE,
)


def _speaker_grounding(entity, text, start, end, speaker_spans):
    """Ground conversational referents only in the explicit serialized turn."""
    for span in speaker_spans or []:
        if not 0 <= span['start'] <= start < end <= span['end'] <= len(text):
            continue
        speaker = span['speaker']
        if entity == speaker:
            # The turn author grounds communication acts as well as first person.
            return True
        if entity == span.get('addressee') and re.search(r"\b(?:you|your|yours)\b", text[start:end], re.I):
            return True
    return False


def parse_extraction_v3(
    response: Mapping[str, Any], *, case_id: str, observation_id: str,
    source_text: str, speaker_spans: list[dict] | None = None,
) -> list[CanonicalState]:
    raw_states = response.get('states')
    if not isinstance(raw_states, list):
        raise ValueError('extraction output must contain a states array')
    result: list[CanonicalState] = []
    # Dialogue can contain the exact same assertion more than once. These are
    # source-grounded repeated occurrences, not malformed model evidence.
    def exact_span(source: str, phrase: str) -> tuple[int, int]:
        if not speaker_spans or source.count(phrase) <= 1:
            return _exact_span(source, phrase)
        if not phrase:
            raise ValueError('empty source evidence')
        occurrences = [(m.start(), m.end()) for m in re.finditer(re.escape(phrase), source)]
        owner = normalize_identity_text(str(raw.get('entity') or ''))
        if owner in {'user', 'assistant', 'system'}:
            grounded = [span for span in occurrences if _speaker_grounding(owner, source, *span, speaker_spans)]
            if not grounded:
                raise ValueError('repeated evidence cannot be attributed to its stated conversational owner')
            occurrences = grounded
        # A representative occurrence anchors the canonical source span; all
        # occurrences remain in the raw input and are persisted by the runner.
        return occurrences[0]
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for index, raw in enumerate(raw_states):
        if not isinstance(raw, Mapping):
            raise ValueError(f'state[{index}] must be an object')
        if raw.get('is_context_only'):
            exact_span(source_text, str(raw.get('evidence_span') or ''))
            continue
        if raw.get('claim_mode') == 'ADVICE':
            phrase = str(raw.get('evidence_span') or '')
            occurrences = [(m.start(), m.end()) for m in re.finditer(re.escape(phrase), source_text)] if phrase else []
            speakers = {p['speaker'] for start, end in occurrences for p in speaker_spans or []
                        if p['start'] <= start < end <= p['end']}
            if len(speakers) != 1:
                raise ValueError('ADVICE evidence needs an unambiguous explicit turn speaker')
            # Speaker metadata defines the author of a communication act.
            # The model's recipient hint cannot become its canonical owner.
            author = next(iter(speakers))
            raw = dict(raw, entity=author, claimant=author, attribute='recommendation')
        node_type = str(raw.get('node_type') or '').upper()
        if node_type not in {
            'FACT', 'CONDITION', 'ROLE_ASSERTION', 'DERIVED_STATE', 'ACTION', 'ACTION_PRECONDITION',
        }:
            raise ValueError(f'state[{index}] has invalid node_type: {node_type!r}')
        event_frame_raw = raw.get('event_frame')
        if not isinstance(event_frame_raw, Mapping):
            raise ValueError(f'state[{index}] event_frame must be an object')
        event_frame = {
            'actor': event_frame_raw.get('actor'),
            'event': event_frame_raw.get('event'),
            'role': event_frame_raw.get('role'),
            'action_status': str(event_frame_raw.get('action_status') or 'UNKNOWN'),
            'participation_requirement': str(event_frame_raw.get('participation_requirement') or 'UNKNOWN'),
        }
        for field in ('actor', 'event', 'role'):
            if isinstance(event_frame[field], str) and event_frame[field].strip().casefold() in {
                'none', 'null', 'n/a', 'not applicable',
            }:
                event_frame[field] = None
        source_event_frame = None
        if node_type not in {'ACTION', 'ROLE_ASSERTION'} and (
            event_frame.get('actor') is not None or event_frame.get('role') is not None
            or event_frame['action_status'] not in {'NONE', 'UNKNOWN'}
        ):
            # A source event producing a factual result is provenance, not an
            # execution state of that result. Preserve it in a separate field.
            actor = event_frame.get('actor')
            if actor is not None and (not isinstance(actor, str) or (
                    actor not in source_text and not _speaker_grounding(
                        normalize_identity_text(actor), source_text,
                        *exact_span(source_text, str(raw.get('evidence_span') or '')), speaker_spans))):
                raise ValueError('source event actor must be grounded in the observation or explicit turn speaker')
            source_event_frame = dict(event_frame)
            event_frame = dict(actor=None, event=None, role=None, action_status='NONE', participation_requirement='NOT_APPLICABLE')
        if node_type != 'ACTION' and event_frame['action_status'] not in {'NONE', 'UNKNOWN'}:
            raise ValueError(f'state[{index}] non-action node has an action status')
        if raw.get('claim_mode') == 'ADVICE':
            # Advice records the communication, not adoption of its content.
            advice_start, advice_end = exact_span(source_text, str(raw.get('evidence_span') or ''))
            turns = [p for p in speaker_spans or [] if p['start'] <= advice_start < advice_end <= p['end']]
            if (len(turns) != 1 or node_type != 'FACT'
                    or raw.get('entity') != turns[0]['speaker']
                    or raw.get('claimant') != turns[0]['speaker']
                    or raw.get('reported_content') is not None
                    or raw.get('attribute') != 'recommendation'):
                raise ValueError('ADVICE must record its explicit speaker recommendation, not an adopted action or reported fact')
        reported_content = raw.get('reported_content')
        if raw.get('claim_mode') == 'REPORTED':
            if not isinstance(reported_content, Mapping) or set(reported_content) != {'entity','attribute','value','evidence_span'}:
                raise ValueError('REPORTED nodes need an explicit grounded content proposition')
            if any(not isinstance(reported_content[k], str) or not reported_content[k].strip() for k in reported_content):
                raise ValueError('reported content fields must be nonempty strings')
            content_start, content_end = exact_span(source_text, reported_content['evidence_span'])
            grounded_start, grounded_end = _expand_evidence_to_sentence(source_text, content_start, content_end)
            if speaker_spans:
                turns = [p for p in speaker_spans if p['start'] <= content_start < content_end <= p['end']]
                if len(turns) != 1:
                    raise ValueError('reported content spans multiple or unknown speakers')
                grounded_start = max(grounded_start, turns[0]['start'])
                grounded_end = min(grounded_end, turns[0]['end'])
            grounded_content = source_text[grounded_start:grounded_end]
            if (normalize_identity_text(reported_content['entity']) not in normalize_identity_text(grounded_content)
                    and not _speaker_grounding(normalize_identity_text(reported_content['entity']), source_text,
                                               content_start, content_end, speaker_spans)):
                raise ValueError('reported content subject must be grounded in its source span')
            reported_content = dict(reported_content, evidence_span=grounded_content)
            if re.fullmatch(r'\s*(?:[$£€]\s*\d+(?:[.,]\d+)*|\d+(?:[.,]\d+)*\s*%(?:\s*off)?)\s*', reported_content['entity'], re.IGNORECASE):
                raise ValueError('reported entity must name the subject, not its monetary/percentage value')
            # Canonical identity tracks the communicated proposition, while claimant
            # tracks its publisher. Retain the raw response separately for audit.
            raw = dict(raw, entity=reported_content['entity'], attribute=reported_content['attribute'], value=reported_content['value'])
        elif reported_content is not None:
            raise ValueError('only REPORTED nodes can contain a communicated proposition')
        entity = ' '.join(str(raw.get('entity') or '').split())
        attribute = ' '.join(str(raw.get('attribute') or '').split())
        value = ' '.join(str(raw.get('value') or '').split())
        if not entity or not attribute or not value:
            raise ValueError(f'state[{index}] requires entity, attribute, and value')
        claimant = raw.get('claimant')
        if claimant is not None and (not isinstance(claimant, str) or (
                claimant not in source_text and not _speaker_grounding(
                    normalize_identity_text(claimant), source_text,
                    *exact_span(source_text, str(raw.get('evidence_span') or '')), speaker_spans))):
            raise ValueError('claimant must be an exact source phrase or the explicit turn speaker')
        polarity = str(raw.get('polarity') or '').upper()
        if polarity not in {'AFFIRMED', 'NEGATED', 'UNKNOWN'}:
            raise ValueError(f'state[{index}] has invalid polarity: {polarity!r}')
        temporal = raw.get('temporal_scope')
        if not isinstance(temporal, Mapping):
            raise ValueError(f'state[{index}] temporal_scope must be an object')
        kind = str(temporal.get('kind') or '')
        if kind not in {'NONE', 'TEXT', 'ABSOLUTE'}:
            raise ValueError(f'state[{index}] has invalid temporal scope kind')
        time_text = temporal.get('text')
        start, end = temporal.get('start'), temporal.get('end')
        normalization_notes: list[str] = (['ADVICE_COMMUNICATION_OWNER_FROM_EXPLICIT_TURN']
                                           if raw.get('claim_mode') == 'ADVICE' else [])
        if isinstance(start, str) and start.strip().casefold() in {'none', 'null'}:
            start = None
        if isinstance(end, str) and end.strip().casefold() in {'none', 'null'}:
            end = None
        if isinstance(time_text, str) and time_text.strip().casefold() == 'none':
            kind = 'NONE'
            time_text = None
            start = end = None
            normalization_notes.append('NONE_SENTINEL_NORMALIZED')
        elif kind == 'TEXT' and time_text is None:
            kind = 'NONE'
            start = end = None
            normalization_notes.append('EMPTY_TEXT_SCOPE_NORMALIZED_TO_NONE')
        elif kind == 'NONE' and isinstance(time_text, str) and _TEMPORAL_MARKERS.search(time_text):
            kind = 'TEXT'
            start = end = None
            normalization_notes.append('TEMPORAL_TEXT_RECOVERED_FROM_NONE_SCOPE')
        if kind == 'NONE' and any(x is not None for x in (time_text, start, end)):
            raise ValueError(f'state[{index}] NONE time scope must have null values')
        if kind in {'TEXT', 'ABSOLUTE'} and (
            not isinstance(time_text, str) or not time_text.strip() or time_text not in source_text
        ):
            raise ValueError(f'state[{index}] temporal phrase is not source-grounded')
        if kind in {'TEXT', 'ABSOLUTE'} and not _TEMPORAL_MARKERS.search(time_text):
            kind, time_text, start, end = 'NONE', None, None, None
            normalization_notes.append('NON_TEMPORAL_MODEL_SCOPE_CLEARED')
        elif kind == 'ABSOLUTE' and any(
            boundary is not None and str(boundary)[:4] not in re.findall(r'\b\d{4}\b', source_text)
            for boundary in (start, end)
        ):
            kind, start, end = 'TEXT', None, None
            normalization_notes.append('UNSTATED_CALENDAR_YEAR_PRESERVED_AS_TEXT')
        elif kind == 'ABSOLUTE' and not (start or end):
            kind, start, end = 'TEXT', None, None
            normalization_notes.append('INCOMPLETE_ABSOLUTE_SCOPE_PRESERVED_AS_TEXT')
        elif kind == 'ABSOLUTE' and any(
            boundary is not None and not re.match(r'^\d{4}-\d{2}-\d{2}(?:T|$)', str(boundary))
            for boundary in (start, end)
        ):
            kind, start, end = 'TEXT', None, None
            normalization_notes.append('NON_ISO_ABSOLUTE_SCOPE_PRESERVED_AS_TEXT')
        elif kind == 'TEXT' and (start is not None or end is not None):
            # Models sometimes place natural-language clock text in ISO boundary
            # slots. Preserve the source-grounded TEXT scope and clear those
            # incompatible values instead of discarding the whole observation.
            start, end = None, None
            normalization_notes.append('NON_ISO_BOUNDARIES_CLEARED_FROM_TEXT_SCOPE')
        if kind == 'TEXT' and isinstance(time_text, str):
            # A quoted label within the property owner is identity, not validity time.
            quoted_labels = re.findall(r"['\"‘“]([^'\"’”]+)['\"’”]", source_text)
            if (normalize_identity_text(time_text) in {normalize_identity_text(x) for x in quoted_labels}
                    and normalize_identity_text(time_text) in normalize_identity_text(entity)):
                kind, time_text, start, end = 'NONE', None, None, None
                normalization_notes.append('QUOTED_OWNER_LABEL_NOT_TEMPORAL_SCOPE')
        if raw.get('property_frame'):
            # Projection must not put the property's name into its owner identity.
            # Preserve quoted proper names; only remove a repeated trailing attribute
            # when the remaining owner phrase is explicitly present in the source.
            quoted_owner = normalize_identity_text(entity) in {
                normalize_identity_text(x) for x in re.findall(r"['\"‘“]([^'\"’”]+)['\"’”]", source_text)
            }
            trailing = re.search(r'\s+' + re.escape(attribute).replace(r'\ ', r'\s+') + r'$', entity, re.I)
            if trailing and not quoted_owner:
                owner = entity[:trailing.start()].strip()
                if owner and re.search(re.escape(owner), source_text, re.I):
                    entity = owner
                    normalization_notes.append('REPEATED_ATTRIBUTE_REMOVED_FROM_PROPERTY_OWNER')
        attribute_identity = normalize_identity_text(attribute)
        if kind in {'TEXT', 'ABSOLUTE'} and re.search(
            r'\b(?:date|deadline|due|renewal|expiry|expiration|anniversary|birthday)\b',
            attribute_identity,
        ) and isinstance(time_text, str) and time_text.casefold() in value.casefold():
            # For date-valued slots, the date is the property value rather than
            # the time at which some other state holds.
            kind, time_text, start, end = 'NONE', None, None, None
            normalization_notes.append('DATE_VALUE_NOT_TEMPORAL_SCOPE')
        # If a weekday/time expression swallowed the event noun, restore that noun
        # to the proposition value and keep only the actual time in temporal scope.
        if kind == 'TEXT' and isinstance(time_text, str):
            match = _WEEKDAY_TIME.match(time_text)
            if match and match.group('tail'):
                value_tokens = normalize_identity_text(value)
                if value_tokens == normalize_identity_text(time_text):
                    value = match.group('tail').strip(' ,;:.')
                    prefix = (match.group('weekday') or '') + (match.group('possessive') or '')
                    time_text = f"{prefix} {match.group('clock')}"
                    normalization_notes.append('EVENT_NOUN_RESTORED_FROM_TEMPORAL_SCOPE')
        if kind == 'TEXT' and isinstance(time_text, str) and time_text.casefold() in value.casefold():
            without_time = re.sub(re.escape(time_text), '', value, flags=re.IGNORECASE)
            without_time = re.sub(r'\s+', ' ', without_time).strip(' ,;:.')
            without_time = re.sub(r'\b(?:on|at|during|in)$', '', without_time, flags=re.IGNORECASE).strip()
            if without_time:
                value = without_time
                normalization_notes.append('TEMPORAL_MODIFIER_MOVED_FROM_VALUE_TO_SCOPE')
        condition = raw.get('condition_scope')
        if condition is not None and (
            not isinstance(condition, str) or condition not in source_text
        ):
            raise ValueError(f'state[{index}] condition is not source-grounded')
        derivation_basis_text = raw.get('derivation_basis_text')
        action_precondition_text = raw.get('action_precondition_text')
        for field, reference in (
            ('derivation_basis_text', derivation_basis_text),
            ('action_precondition_text', action_precondition_text),
        ):
            if reference is not None and (
                not isinstance(reference, str) or not reference.strip() or reference not in source_text
            ):
                raise ValueError(f'state[{index}] {field} must be an exact source phrase or null')
        if node_type == 'DERIVED_STATE' and re.search(r'\bassumes?\b', source_text, re.IGNORECASE):
            assumed_input = re.search(r'\bassumes?\s+.+?\s+from\s+(?:the\s+)?(?P<basis>[^,.;!?]+)', source_text, re.IGNORECASE)
            if assumed_input:
                derivation_basis_text = assumed_input.group('basis').strip()
                normalization_notes.append('ASSUMED_INPUT_REFERENCE_PRESERVED')
        if node_type == 'DERIVED_STATE' and derivation_basis_text is None:
            basis = re.search(
                r'\b(?:computed|calculated|derived)\s+from\s+(?:the\s+)?(?P<basis>[^,.;!?]+)',
                source_text, re.IGNORECASE,
            ) or re.search(
                r'\bbased\s+on\s+(?:the\s+)?(?P<basis>[^,.;!?]+)', source_text, re.IGNORECASE,
            )
            if basis:
                derivation_basis_text = basis.group('basis').strip()
                normalization_notes.append('EXPLICIT_DERIVATION_BASIS_PRESERVED')
        if node_type in {'ACTION', 'ACTION_PRECONDITION'} and action_precondition_text is None:
            basis = re.search(
                r'\b(?:timed\s+to|matching|follows|based\s+on|computed\s+from|determined\s+by|one\s+twelfth\s+of|'
                r'one\s+third\s+of|one\s+half\s+of|one\s+quarter\s+of)\s+'
                r'(?:the\s+)?(?P<basis>[^,.;!?]+)', source_text, re.IGNORECASE,
            )
            if basis:
                action_precondition_text = basis.group('basis').strip()
                normalization_notes.append('EXPLICIT_ACTION_PRECONDITION_PRESERVED')
        evidence_span = str(raw.get('evidence_span') or '')
        try:
            model_start, model_end = exact_span(source_text, evidence_span)
        except ValueError:
            sentence_ranges = [
                (match.start(), match.end())
                for match in re.finditer(r'[^.!?\n]+(?:[.!?]+|$)', source_text)
            ]
            source_tokens = set(normalize_identity_text(source_text).split())
            value_tokens = set(normalize_identity_text(value).split())
            negation_present = bool(re.search(
                r'\b(?:not|no|never|cannot|can\s*not|can\x27t|unable|without|neither|nor)\b',
                source_text, re.IGNORECASE,
            ))
            if (
                len(sentence_ranges) != 1
                or normalize_identity_text(entity) not in normalize_identity_text(source_text)
                or not value_tokens.intersection(source_tokens)
                or (polarity == 'NEGATED' and not negation_present)
            ):
                raise
            model_start, model_end = sentence_ranges[0]
            normalization_notes.append('INVALID_MODEL_EVIDENCE_REPLACED_WITH_SOURCE_SENTENCE')
        span_start, span_end = _expand_evidence_to_sentence(source_text, model_start, model_end)
        parent_start, parent_end = _expand_structured_heading_to_evidence(
            source_text, span_start, span_end, entity,
        )
        if (parent_start, parent_end) != (span_start, span_end):
            span_start, span_end = parent_start, parent_end
            normalization_notes.append('PARENT_LIST_HEADING_INCLUDED_FOR_EVIDENCE')
        grounded_evidence = source_text[span_start:span_end]
        if normalize_identity_text(entity) not in normalize_identity_text(grounded_evidence) and not _speaker_grounding(normalize_identity_text(entity), source_text, model_start, model_end, speaker_spans):
            raise ValueError(f'state[{index}] evidence does not ground the stated entity')
        if speaker_spans:
            containing = [p for p in speaker_spans if p['start'] <= model_start < model_end <= p['end']]
            if len(containing) != 1:
                raise ValueError('evidence spans multiple or unknown speakers')
            span_start = max(span_start, containing[0]['start'])
            span_end = min(span_end, containing[0]['end'])
            grounded_evidence = source_text[span_start:span_end]
            normalization_notes.append('EXPLICIT_CONVERSATION_SPEAKER_PROVENANCE')
        if (span_start, span_end) != (model_start, model_end):
            normalization_notes.append('EVIDENCE_EXPANDED_TO_CONTAINING_SENTENCE')
        if polarity == 'NEGATED' and not re.search(
            r'\b(?:not|no|never|cannot|can\s*not|can\x27t|unable|without|neither|nor)\b',
            grounded_evidence, re.IGNORECASE,
        ):
            polarity = 'AFFIRMED'
            normalization_notes.append('UNSUPPORTED_NEGATION_NORMALIZED_TO_AFFIRMED')
        if node_type == 'FACT' and raw.get('claim_mode') != 'ADVICE' and re.search(
            r'\b(?:computed|calculated|derived)\s+from\b|\bbased\s+on\b|\b(?:estimated|projected|predicted)\s+to\b',
            grounded_evidence, re.IGNORECASE,
        ):
            node_type = 'DERIVED_STATE'
            normalization_notes.append('EXPLICIT_DERIVATION_TYPED_AS_DERIVED_STATE')
        if node_type == 'DERIVED_STATE' and derivation_basis_text is None:
            basis = re.search(
                r'\b(?:computed|calculated|derived)\s+from\s+(?:the\s+)?(?P<basis>[^,.;!?]+)',
                grounded_evidence, re.IGNORECASE,
            ) or re.search(
                r'\bbased\s+on\s+(?:the\s+)?(?P<basis>[^,.;!?]+)',
                grounded_evidence, re.IGNORECASE,
            )
            if basis:
                derivation_basis_text = basis.group('basis').strip()
                normalization_notes.append('EXPLICIT_DERIVATION_BASIS_PRESERVED')
        if node_type == 'FACT' and raw.get('claim_mode') != 'ADVICE' and re.search(
            r'\b(?:arranged|scheduled|planned|agreed|intended)\s+to\s+\w+',
            grounded_evidence, re.IGNORECASE,
        ):
            node_type = 'ACTION'
            event_frame['action_status'] = 'PENDING'
            normalization_notes.append('EXPLICIT_PLANNED_ACTIVITY_TYPED_AS_PENDING_ACTION')
        elif node_type == 'FACT' and raw.get('claim_mode') != 'ADVICE' and re.search(
            r'\b(?:standing|scheduled|recurring)\b.{0,40}\b(?:transfer|payment|pickup|reminder|meeting|delivery)\b',
            grounded_evidence, re.IGNORECASE,
        ):
            node_type = 'ACTION'
            event_frame['action_status'] = 'PENDING'
            normalization_notes.append('RECURRING_OR_STANDING_ACTIVITY_TYPED_AS_PENDING_ACTION')
        if node_type == 'FACT' and raw.get('claim_mode') != 'ADVICE' and re.search(
            r'\b(?:send|sends|sent|queue|queued|schedule|scheduled|deliver|delivers)\b'
            r'.{0,80}\b(?:reminders?|notifications?|alerts?|emails?|messages?)\b',
            grounded_evidence, re.IGNORECASE,
        ) and re.search(
            r'\b(?:daily|weekly|monthly|every|each|once|twice|at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm))\b',
            grounded_evidence, re.IGNORECASE,
        ):
            node_type = 'ACTION'
            event_frame['action_status'] = (
                'PENDING' if re.search(r'\b(?:queued|scheduled)\b', grounded_evidence, re.IGNORECASE)
                else 'IN_PROGRESS'
            )
            if not event_frame.get('actor') and entity in grounded_evidence:
                event_frame['actor'] = entity
            if not event_frame.get('role'):
                verb = re.search(r'\b(send|sends|sent|queue|queued|schedule|scheduled|deliver|delivers)\b',
                                 grounded_evidence, re.IGNORECASE)
                event_frame['role'] = verb.group(1) if verb else None
            if not event_frame.get('event'):
                event = re.search(r'\b(reminders?|notifications?|alerts?|emails?|messages?)\b',
                                  grounded_evidence, re.IGNORECASE)
                event_frame['event'] = event.group(1) if event else None
            basis = re.search(
                r'\b(?:matching|follows|timed\s+to|based\s+on|according\s+to)\s+'
                r'(?:the\s+)?(?P<basis>[^,.;!?]+)', grounded_evidence, re.IGNORECASE,
            )
            if basis and action_precondition_text is None:
                action_precondition_text = basis.group('basis').strip()
            normalization_notes.append('RECURRING_NOTIFICATION_TYPED_AS_ACTION')
        if node_type in {'ACTION', 'ACTION_PRECONDITION'} and action_precondition_text is None:
            basis = re.search(
                r'\b(?:timed\s+to|matching|follows|based\s+on|computed\s+from|determined\s+by|one\s+twelfth\s+of|'
                r'one\s+third\s+of|one\s+half\s+of|one\s+quarter\s+of)\s+'
                r'(?:the\s+)?(?P<basis>[^,.;!?]+)', grounded_evidence, re.IGNORECASE,
            )
            if basis:
                action_precondition_text = basis.group('basis').strip()
                normalization_notes.append('EXPLICIT_ACTION_PRECONDITION_PRESERVED')
        assignment = _NOMINAL_ASSIGNMENT.search(source_text)
        if assignment is not None:
            person = assignment.group('person').strip()
            role = assignment.group('role').strip()
            object_ = assignment.group('object').strip()
            object_key = normalize_identity_text(object_)
            if object_key.split()[:1] in (['a'], ['an'], ['the']):
                object_key = ' '.join(object_key.split()[1:])
            reversed_assignment = (
                normalize_identity_text(entity) == normalize_identity_text(person)
                and normalize_identity_text(attribute) == normalize_identity_text(role)
                and normalize_identity_text(value) == object_key
            )
            ordered_assignment = (
                normalize_identity_text(entity).removeprefix('the ') == object_key
                and normalize_identity_text(attribute) == normalize_identity_text(role)
                and normalize_identity_text(value) == normalize_identity_text(person)
            )
            if reversed_assignment or ordered_assignment:
                # Both tuple orientations must ground the same participant/event;
                # the recording agenda is never the participant in its listed role.
                entity, attribute, value = object_, role, person
                node_type = 'ROLE_ASSERTION'
                event_prefix = source_text[:assignment.start()].strip(' ,;:.')
                event_prefix = re.sub(r'^(?:the|a|an)\s+', '', event_prefix, flags=re.IGNORECASE)
                # Agenda/schedule/calendar identify a document, not the event
                # whose roles it records. Remove only these document heads.
                event_prefix = re.sub(
                    r'\b(?:agenda|schedule|calendar|itinerary|plan)\b', '',
                    event_prefix, flags=re.IGNORECASE,
                )
                event_prefix = ' '.join(event_prefix.split()).strip(' ,;:.')
                event_frame = {
                    'actor': person,
                    'event': event_prefix or object_,
                    'role': role,
                    'action_status': 'NONE',
                    'participation_requirement': 'REQUIRED',
                }
                normalization_notes.append('NOMINAL_ASSIGNMENT_DIRECTION_CANONICALIZED')
                normalization_notes.append('NOMINAL_ASSIGNMENT_TYPED_AS_ROLE_ASSERTION')
        if node_type == 'ROLE_ASSERTION':
            if event_frame.get('event') == attribute or not event_frame.get('event'):
                event_frame['event'] = value
            if not event_frame.get('actor'):
                event_frame['actor'] = entity
            if not event_frame.get('role'):
                event_frame['role'] = attribute
            event_frame['action_status'] = 'NONE'
            if event_frame.get('participation_requirement') == 'UNKNOWN':
                event_frame['participation_requirement'] = 'REQUIRED'
            if re.search(r'\b(?:agenda|schedule|calendar)\b.{0,40}\blists?\b', source_text, re.IGNORECASE):
                event_frame['role_mode'] = 'ASSIGNMENT'
            elif re.search(r'\bwill\b', source_text, re.IGNORECASE):
                event_frame['role_mode'] = 'COMMITMENT'
            elif re.search(r'\b(?:is|was|has been)\b.{0,30}\b(?:chair|presenter|leader|owner|responsible)\b', source_text, re.IGNORECASE):
                event_frame['role_mode'] = 'ACTUAL'
            else:
                event_frame['role_mode'] = 'UNKNOWN'
        elif node_type == 'ACTION' and event_frame.get('event') == attribute:
            event_frame['event'] = value
            if not event_frame.get('role'):
                event_frame['role'] = attribute
        if node_type == 'ACTION':
            # Actor/predicate/object nodes already contain these grounded values;
            # missing frame fields must not erase their participation identity.
            if str(raw.get('node_type')) == 'ACTION' and entity in grounded_evidence and (raw.get('property_frame') or {}).get('assertion_kind') != 'ACTION_RECORD':
                if not event_frame.get('actor'):
                    event_frame['actor'] = entity
                    normalization_notes.append('ACTION_ACTOR_PRESERVED_FROM_CANONICAL_ENTITY')
                if not event_frame.get('event') and value in grounded_evidence and value != time_text:
                    event_frame['event'] = value
                    normalization_notes.append('ACTION_EVENT_PRESERVED_FROM_CANONICAL_OBJECT')
                if not event_frame.get('role') and attribute in grounded_evidence:
                    event_frame['role'] = attribute
            basis = re.search(
                r'\b(?:timed\s+to|matching|follows|based\s+on|computed\s+from|determined\s+by|one\s+twelfth\s+of|'
                r'one\s+third\s+of|one\s+half\s+of|one\s+quarter\s+of)\s+'
                r'(?:the\s+)?(?P<basis>[^,.;!?]+)', grounded_evidence, re.IGNORECASE,
            )
            if basis:
                event_frame['action_precondition_text'] = basis.group('basis').strip()
                normalization_notes.append('EXPLICIT_ACTION_BASIS_PRESERVED')
        if node_type == 'ACTION':
            purpose = re.search(
                r'\b(?P<verb>advertis(?:e|es|ed|ing)|promot(?:e|es|ed|ing)|announc(?:e|es|ed|ing))\s+'
                r'(?P<content>[^,.;!?]+?)(?=\s+(?:is|are|was|were|has|have|on|at|for|through|until|scheduled)\b|[,.;!?]|$)',
                grounded_evidence, re.IGNORECASE,
            )
            if purpose:
                # Preserve a source-stated communication goal separately from
                # an explicit condition. Verification decides its implications.
                event_frame['purpose_reference_text'] = purpose.group('content').strip()
                event_frame['purpose_source_span'] = purpose.group(0).strip()
                normalization_notes.append('EXPLICIT_COMMUNICATION_PURPOSE_PRESERVED')
        reference = re.search(
            r'\b(?:using|through|with|under|based on|because of|matching|follows|timed to)\s+'
            r'(?:that|this|the|his|her|their|its)\s+(?P<reference>[^,.;!?]+)',
            grounded_evidence, re.IGNORECASE,
        )
        if reference:
            event_frame['dependency_reference_text'] = reference.group('reference').strip()
            normalization_notes.append('EXPLICIT_DEPENDENCY_REFERENCE_PRESERVED')
        if node_type in {'FACT', 'CONDITION', 'DERIVED_STATE', 'ACTION_PRECONDITION'}:
            # A FACT can retain an event as context (e.g. a room booking), but it is
            # not an event role or actor node.
            event_frame['actor'] = None
            event_frame['role'] = None
            event_frame['action_status'] = 'NONE'

        if not (node_type == 'FACT' and event_frame.get('event')) and node_type not in {
            'ROLE_ASSERTION', 'ACTION',
        }:
            event_frame['event'] = None
        for field in ('actor', 'event', 'role'):
            frame_value = event_frame.get(field)
            if frame_value is not None:
                if not isinstance(frame_value, str):
                    raise ValueError(f'state[{index}] event_frame.{field} must be text or null')
                value_tokens = set(normalize_identity_text(frame_value).split())
                source_tokens = set(normalize_identity_text(source_text).split())
                if not value_tokens or not value_tokens.intersection(source_tokens):
                    if field in {'event', 'role'} and node_type != 'ROLE_ASSERTION':
                        event_frame[field] = None
                        normalization_notes.append(f'UNSUPPORTED_OPTIONAL_{field.upper()}_CLEARED')
                        continue
                    raise ValueError(f'state[{index}] event_frame.{field} has no source token support')
        if event_frame.get('actor') is not None and event_frame['actor'] not in source_text:
            raise ValueError(f'state[{index}] event_frame.actor must be an exact source phrase')
        if node_type == 'ROLE_ASSERTION' and not all(event_frame.get(k) for k in ('actor', 'event', 'role')):
            raise ValueError(f'state[{index}] role assertion needs actor, event, and role')
        if node_type not in {'ROLE_ASSERTION', 'ACTION'} and (
            event_frame.get('actor') is not None or event_frame.get('role') is not None
        ):
            raise ValueError(f'state[{index}] non-event node cannot carry actor/role fields')
        if node_type != 'DERIVED_STATE' and derivation_basis_text is not None:
            raise ValueError(f'state[{index}] only a DERIVED_STATE may have derivation basis')
        if node_type not in {'ACTION', 'ACTION_PRECONDITION'} and action_precondition_text is not None:
            raise ValueError(f'state[{index}] node type cannot have an action precondition')
        evidence = SourceEvidence(observation_id, source_text, span_start, span_end)
        property_frame = raw.get('property_frame')
        if property_frame:
            operation = property_frame.get('operation')
            if operation:
                if operation['name'] not in source_text or operation.get('owner') is not None and operation['owner'] not in source_text:
                    raise ValueError('operation identity is not source-grounded')
            for field in ('change_agent', 'controlled_object'):
                phrase = property_frame.get(field)
                if phrase is not None and (not isinstance(phrase, str) or phrase not in source_text):
                    raise ValueError('property role is not source-grounded: ' + field)
            if property_frame['assertion_kind'] == 'CONFIGURATION_SETPOINT':
                if property_frame.get('controlled_object') and normalize_identity_text(entity) == normalize_identity_text(property_frame['controlled_object']):
                    raise ValueError('configuration owner and controlled object must be distinct')
            elif property_frame.get('controlled_object') is not None:
                raise ValueError('ordinary property cannot invent a controlled object')
        confidence = raw.get('confidence')
        if confidence is not None:
            confidence = float(confidence)
            if not 0 <= confidence <= 1:
                raise ValueError(f'state[{index}] confidence must be in [0, 1]')
        normalized_time = {
            'kind': kind,
            'text': time_text,
            'start': start,
            'end': end,
        }
        state = CanonicalState(
            case_id=case_id,
            entity=entity,
            attribute=attribute,
            value=value,
            polarity=polarity,
            temporal_scope=normalized_time,
            condition_scope=condition,
            evidence=evidence,
            node_type=node_type,
            claim_mode=str(raw.get('claim_mode', 'DIRECT')),
            source_status=str(raw.get('source_status', 'CURRENT')),
            claimant=raw.get('claimant'),
            reported_content=dict(reported_content) if reported_content is not None else None,
            source_event_frame=source_event_frame,
            property_frame=dict(raw['property_frame']) if raw.get('property_frame') else None,
            event_frame=event_frame,
            derivation_basis_text=derivation_basis_text,
            action_precondition_text=action_precondition_text,
            confidence=confidence,
            normalization_notes=tuple(normalization_notes),
        )
        dedup_key = (
            state.node_type, state.entity_key, state.attribute_key, state.value_key,
            json.dumps(state.temporal_scope, sort_keys=True), state.polarity,
        )
        if dedup_key in seen:
            # Cancelled and archived facets describe one retired action, not two
            # independent active facts. Archive metadata qualifies the same record.
            if state.node_type == 'ACTION' and state.source_status == 'HISTORICAL':
                for previous_index, previous in enumerate(result):
                    if (previous.state_id == state.state_id
                            and (previous.event_frame or {}).get('action_status') == 'CANCELLED'
                            and (state.event_frame or {}).get('action_status') == 'CANCELLED'):
                        result[previous_index] = state
                        break
            continue
        seen.add(dedup_key)
        result.append(state)
    return result


__all__ = ['EXTRACTION_SCHEMA_V3', 'EXTRACTION_SYSTEM_V3', 'parse_extraction_v3']
