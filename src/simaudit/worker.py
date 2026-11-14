import json
from contextlib import redirect_stdout
import sys
from .release import collect, load_factory


def main():
    request = json.load(sys.stdin)
    with redirect_stdout(sys.stderr):
        report = collect(load_factory(request['factory']),request['states'],request['actions'],request['steps'])
    sys.stdout.write(report.dumps())


if __name__ == '__main__':
    main()
