"""Conservative directed propagation; source scope and alternative warrant are explicit."""
from __future__ import annotations
from collections import deque
import re
import calendar
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping


def _interval(scope: Mapping[str, Any]) -> tuple[datetime, datetime] | None:
    start, end = scope.get('start'), scope.get('end')
    if not start or not end:
        return None
    try:
        lo, hi = datetime.fromisoformat(str(start)), datetime.fromisoformat(str(end))
        if len(str(end)) == 10:
            hi += timedelta(days=1, microseconds=-1)
        if bool(lo.tzinfo) != bool(hi.tzinfo):
            return None
        if lo.tzinfo:
            lo, hi = lo.astimezone(timezone.utc), hi.astimezone(timezone.utc)
        if lo > hi:
            return None
        return lo, hi
    except (TypeError, ValueError):
        return None


def _partial_calendar_day(scope: Mapping[str, Any]) -> tuple[int, int] | None:
    """A literal day/month constraint, without inventing a year or timestamp."""
    text = str(scope.get('text') or '')
    if scope.get('start') or scope.get('end') or re.search(
        r'\b(?:from|between|starting|since|until|through|before|after|every|each)\b', text, re.I
    ):
        return None
    months = {name.casefold(): index for index in range(1, 13)
              for name in (calendar.month_name[index], calendar.month_abbr[index])}
    names = '|'.join(sorted(months, key=len, reverse=True))
    matches = re.findall(r'\b(\d{1,2})(?:st|nd|rd|th)?\s+(' + names + r')\b', text, re.I)
    matches += [(day, month) for month, day in re.findall(
        r'\b(' + names + r')\s+(\d{1,2})(?:st|nd|rd|th)?\b', text, re.I)]
    if len(matches) != 1:
        return None
    day, month = int(matches[0][0]), months[matches[0][1].casefold()]
    if not 1 <= day <= [31,29,31,30,31,30,31,31,30,31,30,31][month-1]:
        return None
    return month, day


def scope_overlap(override: Mapping[str, Any], target_scope: Mapping[str, Any]) -> str:
    left, right = _interval(override), _interval(target_scope)
    if left and right and left[0].tzinfo == right[0].tzinfo:
        return 'DISJOINT' if left[1] < right[0] or right[1] < left[0] else 'KNOWN_OVERLAP'
    left_day, right_day = _partial_calendar_day(override), _partial_calendar_day(target_scope)
    if left_day and right_day and left_day != right_day:
        # Different literal calendar days cannot satisfy the same day constraint.
        # Equal days without years do NOT establish overlap across possible years.
        return 'DISJOINT'
    return 'UNRESOLVED'


def propagate(nodes: list[Mapping[str, Any]], revision: Mapping[str, Any],
              verified: list[Mapping[str, Any]], support: Mapping[str, Any]) -> dict:
    by_id = {str(node['state_id']): node for node in nodes}
    if len(by_id) != len(nodes):
        raise ValueError('duplicate canonical state identity')
    initial = {key: str(node.get('source_status', 'CURRENT')) for key, node in by_id.items()}
    statuses = dict(initial)
    outgoing: dict[str, list[Mapping[str, Any]]] = {}
    for edge in verified:
        if not edge.get('validation', {}).get('passed'):
            raise ValueError('unvalidated dependency decision')
        if edge.get('final_decision') not in {'STRICT_DEPENDENCY', 'WEAK_DEPENDENCY'}:
            continue
        if edge['prerequisite'] not in by_id or edge['dependent'] not in by_id:
            raise ValueError('unknown dependency endpoint')
        outgoing.setdefault(edge['prerequisite'], []).append(edge)
    alternatives = {item['dependent']: item for item in support['assessments']}
    if len(alternatives) != len(support['assessments']):
        raise ValueError('duplicate independent-support assessment')
    if any(not item['validation']['passed'] for item in alternatives.values()):
        raise ValueError('unvalidated independent support')
    roots, protected, scoped = set(), set(), []
    transitions = {'MARK_STALE': 'STALE', 'MARK_HISTORICAL': 'HISTORICAL',
                   'MARK_UNCERTAIN': 'UNCERTAIN', 'RECOMPUTE': 'RECOMPUTE'}
    for decision in revision['decisions']:
        incoming, transition = decision['incoming_state_id'], decision['transition']
        if incoming not in by_id:
            raise ValueError('unknown incoming revision state')
        if decision['existing_state_ids']:
            protected.add(incoming)
        for old in decision['existing_state_ids']:
            if old not in by_id:
                raise ValueError('unknown revision target')
            if initial[old] == 'HISTORICAL':
                continue
            if transition == 'SCOPED_OVERRIDE':
                if not decision.get('scope_text'):
                    raise ValueError('scoped override has no source scope')
                scoped.append({'state_id': old, 'scope': decision['scope_text'],
                               'scope_bounds': by_id[incoming]['temporal_scope'], 'replacement': incoming})
            elif transition in transitions:
                statuses[old] = transitions[transition]
                roots.add(old)

    def rescue(target: str) -> str | None:
        if target in roots or target in protected:
            return 'Direct root/replacement is protected from cyclic reinvalidation'
        if initial[target] == 'HISTORICAL':
            return 'Historical assertion retains its recorded scope'
        assessment = alternatives.get(target)
        if assessment is None:
            raise ValueError('missing independent-support assessment: ' + target)
        if assessment['fully_supported']:
            ids = assessment['support_ids']
            if not ids or any(key == target or statuses.get(key) != 'CURRENT' for key in ids):
                raise ValueError('independent support is no longer CURRENT')
            return 'Independent CURRENT evidence establishes the same operative claim'
        return None

    def effect(edge: Mapping[str, Any], source_status: str) -> str:
        if edge['final_decision'] == 'WEAK_DEPENDENCY' or source_status == 'DOWNWEIGHT':
            return 'DOWNWEIGHT'
        return 'STALE' if (edge['final_decision'] == 'STRICT_DEPENDENCY'
                           and source_status in {'STALE', 'HISTORICAL'}) else 'UNCERTAIN'

    rank = {'CURRENT': 0, 'DOWNWEIGHT': 0.5, 'UNCERTAIN': 1, 'RECOMPUTE': 1, 'STALE': 2, 'HISTORICAL': 2}
    pending = deque((root, root, 0) for root in sorted(roots))
    processed, steps, recompute, downweighted = set(), [], set(), set()
    while pending:
        source, root, depth = pending.popleft()
        source_effect = 'DOWNWEIGHT' if source in downweighted and statuses[source] == 'CURRENT' else statuses[source]
        token = source, root, source_effect
        if token in processed:
            continue
        processed.add(token)
        for edge in outgoing.get(source, []):
            target = edge['dependent']
            keep = rescue(target)
            if keep:
                steps.append({'root': root, 'prerequisite': source, 'dependent': target,
                              'depth': depth + 1, 'effect': 'KEEP', 'reason': keep})
                continue
            proposed = effect(edge, source_effect)
            target_effect = 'DOWNWEIGHT' if target in downweighted and statuses[target] == 'CURRENT' else statuses[target]
            changed = rank[proposed] > rank.get(target_effect, 0)
            steps.append({'root': root, 'prerequisite': source, 'dependent': target,
                          'depth': depth + 1, 'effect': proposed, 'relation': edge['relation'],
                          'strength': edge['final_decision'], 'changed': changed})
            if changed:
                if proposed == 'DOWNWEIGHT':
                    downweighted.add(target)
                else:
                    statuses[target] = proposed
                if by_id[target]['node_type'] == 'DERIVED_STATE':
                    recompute.add(target)
                pending.append((target, root, depth + 1))

    # Conditional effects never silently become global invalidations. Unknown
    # overlap remains an explicit runtime condition, not an invented date match.
    scoped_impacts = []
    for override in scoped:
        local = {override['state_id']: 'STALE'}
        queue = deque([(override['state_id'], 0)])
        while queue:
            source, depth = queue.popleft()
            for edge in outgoing.get(source, []):
                target = edge['dependent']
                if target == override['state_id']:
                    continue
                overlap = scope_overlap(override['scope_bounds'], by_id[target].get('temporal_scope') or {})
                keep = rescue(target)
                if keep or overlap == 'DISJOINT':
                    scoped_impacts.append(dict(override, prerequisite=source, dependent=target,
                                               depth=depth + 1, effect='KEEP', overlap=overlap,
                                               reason=keep or 'Source interval proves disjoint scope'))
                    continue
                proposed = effect(edge, local[source])
                if rank[proposed] <= rank.get(local.get(target, 'CURRENT'), 0):
                    continue
                local[target] = proposed
                scoped_impacts.append(dict(override, prerequisite=source, dependent=target,
                                           depth=depth + 1, effect=proposed, overlap=overlap,
                                           relation=edge['relation'], strength=edge['final_decision']))
                queue.append((target, depth + 1))
    blocked = sorted(key for key, node in by_id.items()
                     if node['node_type'] == 'ACTION' and statuses[key] != 'CURRENT')
    return {'statuses': statuses, 'roots': sorted(roots),
            'invalidated_state_ids': sorted(key for key in by_id
                                           if statuses[key] in {'STALE', 'HISTORICAL'}
                                           and initial[key] != 'HISTORICAL'),
            'uncertain_state_ids': sorted(key for key in by_id if statuses[key] in {'UNCERTAIN', 'RECOMPUTE'}),
            'needs_recompute': sorted(recompute), 'downweighted_state_ids': sorted(downweighted), 'blocked_action_ids': blocked,
            'scoped_overrides': scoped, 'scoped_impacts': scoped_impacts, 'steps': steps}
