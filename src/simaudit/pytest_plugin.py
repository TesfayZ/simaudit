"""Optional pytest fixture preserving all six diagnostic statuses."""
import pytest
from .report import Report, Status


def pytest_addoption(parser):
    group = parser.getgroup('simaudit')
    group.addoption('--simaudit-strict',action='store_true',help='Fail session on any non-pass audit result')
    group.addoption('--simaudit-json',help='Write collected audit results to JSON (serial pytest only)')


def pytest_configure(config):
    config._simaudit_results = []
    if config.getoption('--simaudit-json') and getattr(config.option,'numprocesses',0):
        raise pytest.UsageError('--simaudit-json currently requires serial pytest')


@pytest.fixture
def simaudit_check(request):
    def check(result):
        # Validate now, before appending to the session report.
        Report.loads(Report([result]).dumps())
        request.config._simaudit_results.append(result)
        request.node.user_properties.append(('simaudit_status',result.status.value))
        message = Report([result]).summary()
        if result.status == Status.FAIL: pytest.fail(message,pytrace=False)
        if result.status != Status.PASS: pytest.skip(message)
        return result
    return check


def pytest_sessionfinish(session,exitstatus):
    report = Report(session.config._simaudit_results)
    path = session.config.getoption('--simaudit-json')
    if path: report.write(path)
    if report.results and session.config.getoption('--simaudit-strict') and report.exit_code(True) and exitstatus==0:
        session.exitstatus = 1


def pytest_terminal_summary(terminalreporter,exitstatus,config):
    if config._simaudit_results:
        terminalreporter.section('simaudit (original diagnostic statuses)')
        terminalreporter.write_line(Report(config._simaudit_results).summary())
