import argparse
import json
from pathlib import Path
import sys
from . import __version__, Report, check_assignment, check_energy, check_observation, check_order, compare_releases
from .release import load_factory
from .workflow import QUESTIONS, diagnose, plan, workflow_summary


def main(argv=None):
    parser = argparse.ArgumentParser(description='Scoped diagnostics for any simulator through a trusted module:function adapter factory')
    parser.add_argument('--version',action='version',version=f'simaudit {__version__}')
    sub = parser.add_subparsers(dest='check',required=True)
    for command in ('plan','diagnose'):
        cmd = sub.add_parser(command)
        cmd.add_argument('question', choices=QUESTIONS)
        cmd.add_argument('--inputs', required=True, help='Trusted module:function returning WorkflowInputs')
        cmd.add_argument('--json', type=Path)
        if command == 'diagnose':
            cmd.add_argument('--strict', action='store_true')
    for name in ('assignment','order','observation','release','energy'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--factory',required=True,help='Trusted Python module:function returning simaudit.Adapter')
        cmd.add_argument('--probes',required=True,type=Path,help='JSON object with states and actions arrays')
        cmd.add_argument('--json',type=Path,help='Write full report here')
        cmd.add_argument('--strict',action='store_true')
        cmd.add_argument('--atol',type=float,default={'release':1e-13, 'energy':0.0}.get(name, 1e-12))
        if name=='assignment':
            cmd.add_argument('--parameter',required=True)
            cmd.add_argument('--value',type=float,required=True)
        if name in ('order','energy'): cmd.add_argument('--interval',type=float)
        if name=='energy':
            cmd.add_argument('--energy-noise-floor',type=float,default=0.0,help='Absolute defect-rate floor in energy per unit model time')
        if name=='release':
            cmd.add_argument('--factory-b',help='Explicit second factory for controlled variant comparisons')
            cmd.add_argument('--python-a',required=True)
            cmd.add_argument('--python-b',required=True)
            cmd.add_argument('--steps',type=int,default=5)
            cmd.add_argument('--timeout',type=float,default=60)
    args = parser.parse_args(argv)
    try:
        if args.check in ('plan','diagnose'):
            inputs = load_factory(args.inputs)
            preview = args.check == 'plan'
            workflow = (plan if preview else diagnose)(args.question, inputs)
            output = json.dumps(workflow, indent=2, allow_nan=False)
            if args.json:
                args.json.write_text(output + '\n', encoding='utf-8')
            print(workflow_summary(workflow))
            return 0 if preview else workflow['strict_exit_code' if args.strict else 'exit_code']
        data = json.loads(args.probes.read_text())
        if set(data) != {'states','actions'}: raise ValueError('Probe file must contain exactly states and actions')
        states,actions = data['states'],data['actions']
        if args.check=='release':
            r = compare_releases(args.factory,args.python_a,args.python_b,states,actions,
                                 steps=args.steps,atol=args.atol,timeout=args.timeout,factory_b=args.factory_b)
        else:
            a = load_factory(args.factory)
            if args.check=='assignment': r = check_assignment(a,args.parameter,args.value,states,actions,atol=args.atol)
            elif args.check=='order': r = check_order(a,states,actions,interval=args.interval,atol=args.atol)
            elif args.check=='energy':
                r = check_energy(a, states, actions, interval=args.interval, atol=args.atol,
                                 energy_noise_floor=args.energy_noise_floor)
            else: r = check_observation(a,states,actions,atol=args.atol)
        report = Report([r])
        if args.json: report.write(args.json)
        print(report.summary())
        return report.exit_code(args.strict)
    except (ValueError, TypeError, OSError, ImportError, AttributeError) as exc:
        print(f'simaudit: configuration error: {exc}',file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
