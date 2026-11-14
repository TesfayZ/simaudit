import argparse
import numpy as np
from simaudit import Report, compare_releases

parser=argparse.ArgumentParser()
parser.add_argument('--python-a',required=True)
parser.add_argument('--python-b',required=True)
parser.add_argument('--out',required=True)
a=parser.parse_args()
rng=np.random.default_rng(20261111)
states=rng.uniform(-.2,.2,(24,4)); actions=[i%3 for i in range(24)]
results=[]
for factory,interpreter in [('known_cases:acrobot_book',a.python_b),
                            ('known_cases:acrobot_nips',a.python_a),
                            ('known_cases:acrobot_mass',a.python_a)]:
    results.append(compare_releases('known_cases:acrobot_book',a.python_a,interpreter,
        states,actions,steps=5,factory_b=factory))
Report(results).write(a.out)
for r in results: print(r.status.value,r.reason,r.evidence.get('first_step_max_difference'))
