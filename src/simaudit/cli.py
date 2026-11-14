import argparse
import json
from pathlib import Path
import sys
from . import Report, check_assignment, check_observation, check_order, compare_releases
from .release import load_factory


def main(argv=None):
    parser = argparse.ArgumentParser(description='Scoped diagnostics for any simulator through a trusted module:function adapter factory')
    parser.add_argument('--version',action='version',version='simaudit 0.1.0a1')
    sub = parser.add_subparsers(dest='check',required=True)
    for name in ('assignment','order','observation','release'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--factory',required=True,help='Trusted Python module:function returning simaudit.Adapter')
        cmd.add_argument('--probes',required=True,type=Path,help='JSON object with states and actions arrays')
        cmd.add_argument('--json',type=Path,help='Write full report here')
        cmd.add_argument('--strict',action='store_true')
        cmd.add_argument('--atol',type=float,default=1e-13 if name=='release' else 1e-12)
        if name=='assignment':
            cmd.add_argument('--parameter',required=True)
            cmd.add_argument('--value',type=float,required=True)
        if name=='order': cmd.add_argument('--interval',type=float)
        if name=='release':
            cmd.add_argument('--factory-b',help='Explicit second factory for controlled variant comparisons')
            cmd.add_argument('--python-a',required=True)
            cmd.add_argument('--python-b',required=True)
            cmd.add_argument('--steps',type=int,default=5)
            cmd.add_argument('--timeout',type=float,default=60)
    args = parser.parse_args(argv)
    try:
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
