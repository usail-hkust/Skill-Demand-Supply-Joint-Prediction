from subprocess import Popen, PIPE
import os.path, sys

# Import helper function to compare graphs from tests/regressions_tests
sys.path.insert(0, os.path.abspath('../'))
from regression_test_helpers import compare_graphs

vulnfiles = [
    'nullderefrebuildlist'
]

output_types = [
    ('xdot', 'xdot:xdot:core')
]

def generate_vuln_graph(vulnfile, output_type):
    if not os.path.exists('output'):
        os.makedirs('output')

    output_file = 'output/' + vulnfile + '.' + output_type[0]
    input_file = 'input/' + vulnfile + '.dot'
    process = Popen(['dot', '-T' + output_type[1], '-o', output_file, input_file], stdin=PIPE)

    if process.wait() < 0:
        print('An error occurred while generating: ' + output_file)
        exit(1)

failures = 0
for vulnfile in vulnfiles:
    for output_type in output_types:
        generate_vuln_graph(vulnfile, output_type)
        if not compare_graphs(vulnfile, output_type[0]):
            failures += 1

print('')
print('Results for "vuln" regression test:')
print('    Number of tests: ' + str(len(vulnfiles) * len(output_types)))
print('    Number of failures: ' + str(failures))

if not failures == 0:
    exit(1) 
