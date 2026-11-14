import os
from pathlib import Path
import subprocess
import sys
from simaudit import Report


def test_plugin_preserves_abstain(tmp_path):
    (tmp_path/'test_sample.py').write_text('''
from simaudit import Adapter, Feature, check_order

def test_audit(simaudit_check):
    a=Adapter('missing', features=[Feature('dynamics','unsupported','Missing native step')])
    simaudit_check(check_order(a,[[1]],[0]))
''')
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'),PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    command=[sys.executable,'-m','pytest','-p','simaudit.pytest_plugin','-q',str(tmp_path),
             '--simaudit-json',str(tmp_path/'report.json')]
    run=subprocess.run(command,env=env,capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
    assert 'SKIPPED=1' in run.stdout
    assert Report.loads((tmp_path/'report.json').read_text()).counts['SKIPPED']==1
    strict=subprocess.run(command+['--simaudit-strict'],env=env,capture_output=True,text=True)
    assert strict.returncode==1
