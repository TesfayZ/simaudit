"""Deterministic planning and execution for one declared diagnostic question."""
from dataclasses import asdict, dataclass
from importlib.util import find_spec
import json

from .adapters import Adapter
from .attribution import attribute
from .checks import check_assignment, check_observation, _check_order
from .energy import _check_energy
from .refinement import Refinement
from .fit import EnergyFit, fit_parameter
from .laws import LawStudy, log_requirements, screen_laws
from .release import compare_releases
from .report import Report, Status


@dataclass
class WorkflowInputs:
    target: str
    adapter: Adapter | None = None
    states: list | None = None
    actions: list | None = None
    assignment: tuple[str, float] | None = None
    release: dict | None = None   # factory, python_a, python_b and compare_releases options
    interval: float | None = None
    fit: EnergyFit | None = None
    logs: LawStudy | None = None
    logs_use_adapter_observations: bool | None = None


BRANCHES = ('assignment', 'release', 'observation', 'energy', 'order', 'fit', 'screen', 'attribution')
QUESTIONS = (*BRANCHES, 'drift', 'full')


def plan(question, inputs):
    if not isinstance(inputs, WorkflowInputs) or not inputs.target:
        raise ValueError('Supply WorkflowInputs with a target identity')
    if question not in QUESTIONS:
        raise ValueError('Choose an explicit question: ' + ', '.join(QUESTIONS))
    branches = BRANCHES if question == 'full' else {
        'drift': ('energy','order'), 'attribution': ('screen','attribution')
    }.get(question, (question,))
    adapter = inputs.adapter
    caps = adapter.capabilities if adapter else set()
    items = []
    for branch in branches:
        missing = []
        if branch not in ('fit','screen','attribution'):
            if inputs.states is None or inputs.actions is None or len(inputs.states) == 0:
                missing.append('matched state/action probes')
            elif len(inputs.states) != len(inputs.actions):
                raise ValueError('State and action counts differ')
        if branch not in ('release','screen','attribution'):
            required = {'step','native_step'}
            required.update({'assignment': ['assignment'], 'observation': ['observation','reference_observation'],
                             'energy': ['energy']}.get(branch, ()))
            missing.extend(sorted(required - caps))
        if branch in ('energy','order','fit') and adapter:
            if adapter.smooth is not True:
                missing.append('smooth replay regime')
            if adapter.order_blockers or not adapter.features or any(
                    f.status == 'unsupported' and f.present is not False for f in adapter.features):
                missing.append('supported refinement feature inventory')
            if branch != 'order' and adapter.conservative is not True:
                missing.append('passive conservative probes')
            if branch != 'fit' and adapter.declared_order is None:
                missing.append('declared integration order')
        if branch == 'assignment' and inputs.assignment is None:
            missing.append('parameter and intended value')
        if branch == 'release':
            missing.extend(sorted({'factory','python_a','python_b'} - set(inputs.release or {})))
        if branch == 'fit' and inputs.fit is None:
            missing.append('named energy family, bounds, training and held-out starts')
        if branch in ('screen','attribution') and (not inputs.logs or not inputs.logs.laws):
            missing.append('matched trajectory groups and named laws')
        if branch == 'screen' and inputs.logs:
            issue = log_requirements(inputs.logs)
            if issue:
                missing.append(issue)
        if branch == 'attribution':
            if inputs.logs and not inputs.logs.recovery:
                missing.append('recovery dictionary and competing parameter hypotheses')
            if find_spec('sympy') is None:
                missing.append('simaudit[algebra]')
        items.append(dict(check=branch, missing=missing,
                          condition='screen flags a law' if branch == 'attribution' else None))
    return dict(workflow_version='1.0', question=question, target=inputs.target, branches=items)


def _interpret(results):
    by_name = {r.check:r for r in results}
    if 'energy' not in by_name or 'order' not in by_name:
        return None
    energy, order = by_name['energy'], by_name['order']
    pattern = energy.evidence.get('classification')
    if pattern == 'model' and order.status == Status.PASS:
        return 'Energy disagrees with the reference while state order meets its declaration. Inspect the reference and model parameters.'
    if pattern and pattern.startswith('order-') and energy.status == Status.FAIL and order.fault_class == 'integrator_order':
        return 'Both refinements indicate a discretisation issue. A concurrent model discrepancy can be masked.'
    if pattern == 'consistent' and order.status == Status.PASS:
        return 'Both refinement measurements meet their scoped criteria.'
    return 'The refinement results leave the cause unresolved. Inspect both measurements for mixed effects or limited resolution.'


def diagnose(question, inputs):
    record = plan(question, inputs)
    results = []
    a = inputs.adapter
    args = (a, inputs.states, inputs.actions)
    refinement = Refinement()
    integrity_failed = False
    observation_blocker = None
    screen = None
    for item in record['branches']:
        branch = item['check']
        item['results'] = []
        if branch == 'attribution' and screen and screen.status == Status.PASS:
            item['outcome'] = 'no flagged law'
            continue
        if item['missing']:
            item['outcome'] = 'missing inputs'
            continue
        if integrity_failed and branch in ('energy','order','fit','assignment','observation'):
            item['outcome'] = 'blocked by adapter integrity failure'
            continue
        if branch == 'attribution':
            if not screen or 'flagged' not in screen.evidence or screen.status not in (Status.PASS, Status.FAIL):
                item['outcome'] = 'screen unresolved'
                continue
            flagged = screen.evidence['flagged']
            if not flagged:
                item['outcome'] = 'no flagged law'
                continue
            if (observation_blocker or integrity_failed) and inputs.logs_use_adapter_observations is not False:
                item['outcome'] = 'physical attribution blocked by ' + (observation_blocker or 'adapter integrity failure')
                continue
            measurements = []
            item['missing_templates'] = sorted(set(flagged) - inputs.logs.recovery.keys())
            for law in inputs.logs.laws:
                if law.name in flagged and law.name in inputs.logs.recovery:
                    measurements.append(attribute(inputs.logs, law, inputs.logs.recovery[law.name], flagged))
        elif branch == 'release':
            measurements = [compare_releases(states=inputs.states, actions=inputs.actions, **inputs.release)]
        elif branch == 'assignment':
            measurements = [check_assignment(a, *inputs.assignment, inputs.states, inputs.actions)]
        elif branch == 'screen':
            screen = screen_laws(inputs.logs)
            measurements = [screen]
        elif branch == 'observation':
            observation = check_observation(*args)
            if observation.status != Status.PASS:
                observation_blocker = ('observation discrepancy' if observation.status == Status.FAIL
                                       else 'unresolved observation check')
            measurements = [observation]
        elif branch == 'fit':
            measurements = [fit_parameter(a, inputs.fit, interval=inputs.interval)]
        else:
            check = _check_order if branch == 'order' else _check_energy
            measurements = [check(*args, refinement, interval=inputs.interval)]
        item['results'] = list(range(len(results), len(results) + len(measurements)))
        item['outcome'] = 'executed' if measurements else 'missing templates for flagged laws'
        results.extend(measurements)
        integrity_failed |= any(r.fault_class == 'adapter' and branch != 'release' and 'configuration' not in r.evidence
                                for r in measurements)
    report = Report(results)
    complete = all(b['outcome'] in ('executed','no flagged law') and not b.get('missing_templates')
                   for b in record['branches']) and all(r.status in (Status.PASS, Status.FAIL)
                   and r.fault_class != 'adapter' and not r.evidence.get('unresolved')
                   and all(p.get('status') != 'ABSTAIN' for p in r.evidence.get('probes', []))
                   for r in results)
    record.update(coverage='complete' if complete else 'partial', interpretation=_interpret(results),
                  report=asdict(report), exit_code=report.exit_code(),
                  strict_exit_code=int(not complete or report.exit_code(strict=True)))
    json.dumps(record, allow_nan=False)
    return record


def workflow_summary(record):
    """Keep branch selection, measurements and follow-up inputs visible in the CLI."""
    lines = [f"{record['target']} — {record['question']}"]
    results = record.get('report', {}).get('results', [])
    for branch in record['branches']:
        name = branch['check']
        outcome = branch.get('outcome')
        if outcome == 'executed':
            for i in branch['results']:
                row = results[i]
                lines.append(f"{Status(row['status']).value} {name} — {row['reason']}")
            if branch.get('missing_templates'):
                lines.append('  Missing templates for ' + ', '.join(branch['missing_templates']))
        elif branch['missing'] and outcome != 'no flagged law':
            lines.append(f"{name} needs " + ', '.join(branch['missing']))
        else:
            lines.append(f"{name} — {outcome or branch['condition'] or 'ready'}")
    if record.get('interpretation'):
        lines.append(record['interpretation'])
    if 'coverage' in record:
        lines.append('Coverage ' + record['coverage'])
    return '\n'.join(lines)
