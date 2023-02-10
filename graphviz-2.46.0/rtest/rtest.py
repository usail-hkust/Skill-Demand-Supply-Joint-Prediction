#!/usr/bin/env python3
#
# Graphviz regression test driver
#
# Also relies on strps.awk.
#
# TODO:
#  Report differences with shared version and with new output.

import os
import shutil
import tempfile
import shutil
import subprocess
import sys
import platform
import argparse
import atexit
import pathlib

TESTFILE = 'tests.txt'                # Test specifications
GRAPHDIR = 'graphs'                   # Directory of input graphs and data
OUTDIR = 'ndata'                      # Directory for test output
OUTHTML = 'nhtml'                     # Directory for html test report
REFDIR = os.environ.get('REFDIR', '') # Directory for expected test output
GENERATE = False                      # If set, generate test data
VERBOSE = False                       # If set, give verbose output
NOOP = False                          # If set, just print list of tests
DOT = os.environ.get('DOT', shutil.which('dot'))
DIFFIMG = os.environ.get('DIFFIMG', shutil.which('diffimg'))

TESTNAME = ''   # name of test
CRASH_CNT = 0
DIFF_CNT = 0
TOT_CNT = 0
TESTTYPES = {}
TMPINFILE = 'tmp{}.gv'.format(os.getpid())
TMPFILE1 = 'tmpnew{}'.format(os.getpid())
TMPFILE2 = 'tmpref{}'.format(os.getpid())

# Read single line, storing it in LINE.
# Returns the line on success, else returns None
def readLine():
  LINE = f3.readline()
  if LINE != '':
    return LINE.strip()
  else:
    return None

# Skip blank lines and comments (lines starting with #)
# Use first real line as the test name
def skipLines():
  while True:
    LINE = readLine()
    if LINE is None:
       return None
    if LINE and not LINE.startswith('#'):
      return LINE

# Subtests have the form: layout format optional_flags
# Store the 3 parts in the arrays ALG, FMT, FLAGS.
# Stop at a blank line
def readSubtests():
  while True:
    LINE = readLine()
    if LINE == '':
      return
    if not LINE.startswith('#'):
      ALG0, FMT0, *FLAGS0 = LINE.split(' ')
      yield {
              'ALG': ALG0,
              'FMT': FMT0,
              'FLAGS': FLAGS0,
            }

def readTest():
  # read test name
  LINE = skipLines()
  if LINE is not None:
    TESTNAME = LINE
  else:
    return None

  # read input graph
  LINE = skipLines()
  if LINE is not None:
    GRAPH = LINE
  else:
    return None

  SUBTESTS = list(readSubtests())
  return {
      'TESTNAME': TESTNAME,
      'GRAPH': GRAPH,
      'SUBTESTS': SUBTESTS,
      }

# Compare old and new output and report if different.
#  Args: testname index fmt
def doDiff(OUTFILE, OUTDIR, REFDIR, testname, subtest_index, fmt):
  global DIFF_CNT
  FILE1 = os.path.join(OUTDIR, OUTFILE)
  FILE2 = os.path.join(REFDIR, OUTFILE)
  F = fmt.split(':')[0]
  # FIXME: Remove when https://gitlab.com/graphviz/graphviz/-/issues/1789 is
  # fixed
  if platform.system() == 'Windows' and \
     F in ['ps', 'gv'] and \
     testname in ['clusters', 'compound', 'rootlabel']:
      print('Warning: Skipping {0} output comparison for test {1}:{2} : format '
            '{3} because the order of clusters in gv or ps output is not '
            'stable on Windows (#1789)'
            .format(F, testname, subtest_index, fmt),
            file=sys.stderr)
      return
  if F in ['ps', 'ps2']:
    with open(TMPFILE1, mode='w') as fd:
      subprocess.check_call(
        ['awk', '-f', 'strps.awk', FILE1],
        stdout=fd,
      )
    with open(TMPFILE2, mode='w') as fd:
      subprocess.check_call(
        ['awk', '-f', 'strps.awk', FILE2],
        stdout=fd,
      )
    returncode = subprocess.call(
      ['diff', '-q', TMPFILE1, TMPFILE2],
      stdout=subprocess.DEVNULL,
    )
  elif F == 'svg':
    with open(TMPFILE1, mode='w') as fd:
      subprocess.check_call(
        ['sed', '/^<!--/d;/-->$/d', FILE1],
        stdout=fd,
      )
    with open(TMPFILE2, mode='w') as fd:
      subprocess.check_call(
        ['sed', '/^<!--/d;/-->$/d', FILE2],
        stdout=fd,
      )
    returncode = subprocess.call(
      ['diff', '-q', '--strip-trailing-cr', TMPFILE1, TMPFILE2],
      stdout=subprocess.DEVNULL,
    )
  elif F == 'png':
    # FIXME: remove when https://gitlab.com/graphviz/graphviz/-/issues/1788 is fixed
    if os.environ.get('build_system') == 'cmake':
      print('Warning: Skipping PNG image comparison for test {0}:{1} : format: '
            '{2} because CMake builds does not contain the diffimg '
            'utility (#1788)'
            .format(testname, subtest_index, fmt),
            file=sys.stderr)
      return
    returncode = subprocess.call(
      [DIFFIMG, FILE1, FILE2, os.path.join(OUTHTML, 'dif_' + OUTFILE)],
    )
    if returncode != 0:
      with open(os.path.join(OUTHTML, 'index.html'), mode='a') as fd:
        fd.write('<p>\n')
        shutil.copyfile(FILE2, os.path.join(OUTHTML, 'old_' + OUTFILE))
        fd.write('<img src="old_{}" width="192" height="192">\n'.format(OUTFILE))
        shutil.copyfile(FILE1, os.path.join(OUTHTML, 'new_' + OUTFILE))
        fd.write('<img src="new_{}" width="192" height="192">\n'.format(OUTFILE))
        fd.write('<img src="dif_{}" width="192" height="192">\n'.format(OUTFILE))
    else:
      os.unlink(os.path.join(OUTHTML, 'dif_' + OUTFILE))
  else:
    returncode = subprocess.call(
      ['diff', '--strip-trailing-cr', FILE2, FILE1],
      stdout=subprocess.DEVNULL,
    )
  if returncode != 0:
    print('Test {0}:{1} : == Failed == {2}'.format(testname, subtest_index, OUTFILE), file=sys.stderr)
    DIFF_CNT += 1
  else:
    if VERBOSE:
      print('Test {0}:{1} : == OK == {2}'.format(testname, subtest_index, OUTFILE), file=sys.stderr)

# Generate output file name given 3 parameters.
#   testname layout format
# If format ends in :*, remove this, change the colons to underscores,
# and append to basename
# If the last two parameters have been used before, add numeric suffix.
def genOutname(name, alg, fmt):
  global TESTTYPES
  fmt_split = fmt.split(':')
  if len(fmt_split) >= 2:
    F = fmt_split[0]
    XFMT = '_' + '_'.join(fmt_split[1:])
  else:
    F=fmt
    XFMT=''

  IDX = alg + XFMT + F
  j = TESTTYPES.get(IDX, 0)
  if j == 0:
    TESTTYPES[IDX] = 1
    J = ''
  else:
    TESTTYPES[IDX]= j + 1
    J = str(j)
  OUTFILE = name + '_' + alg + XFMT + J + '.' + F
  return OUTFILE

def doTest(TEST):
  global TOT_CNT
  global CRASH_CNT
  global TESTTYPES
  TESTNAME = TEST['TESTNAME']
  SUBTESTS = TEST['SUBTESTS']
  if len(SUBTESTS) == 0:
    return
  GRAPH = TEST['GRAPH']
  if GRAPH == '=':
    INFILE = os.path.join(GRAPHDIR, TESTNAME + '.gv')
  elif GRAPH.startswith('graph') or GRAPH.startswith('digraph'):
    with open(TMPINFILE, mode='w') as fd:
      fd.write(GRAPH)
    INFILE = TMPINFILE
  elif os.path.splitext(GRAPH)[1] == '.gv':
    INFILE = os.path.join(GRAPHDIR, GRAPH)
  else:
    print('Unknown graph spec, test {} - ignoring'.format(TESTNAME),
          file=sys.stderr)
    return

  for i, SUBTEST in enumerate(SUBTESTS):
    TOT_CNT += 1
    OUTFILE = genOutname(TESTNAME, SUBTEST['ALG'], SUBTEST['FMT'])
    OUTPATH = os.path.join(OUTDIR, OUTFILE)
    KFLAGS = SUBTEST['ALG']
    TFLAGS = SUBTEST['FMT']
    if KFLAGS: KFLAGS = '-K{}'.format(KFLAGS)
    if TFLAGS: TFLAGS = '-T{}'.format(TFLAGS)
    testcmd = [DOT]
    if KFLAGS: testcmd += [KFLAGS]
    if TFLAGS: testcmd += [TFLAGS]
    testcmd += SUBTEST['FLAGS'] + ['-o', OUTPATH, INFILE]
    if VERBOSE:
      print(' '.join(testcmd))
    if NOOP:
      continue
    # FIXME: Remove when https://gitlab.com/graphviz/graphviz/-/issues/1786 is
    # fixed
    if os.environ.get('build_system') == 'cmake' and \
       SUBTEST['FMT'] == 'png:gd':
      print('Skipping test {0}:{1} : format {2} because CMake builds '
            'does not support format png:gd (#1786)'
            .format(TESTNAME, i, SUBTEST['FMT']),
            file=sys.stderr)
      continue
    # FIXME: Remove when https://gitlab.com/graphviz/graphviz/-/issues/1269 is
    # fixed
    if platform.system() == 'Windows' and \
       os.environ.get('build_system') == 'msbuild' and \
       '-Goverlap=false' in SUBTEST['FLAGS']:
      print('Skipping test {0}:{1} : with flag -Goverlap=false because it fails '
            'with Windows MSBuild builds which are not built with '
            'triangulation library (#1269)'
            .format(TESTNAME, i),
            file=sys.stderr)
      continue
    # FIXME: Remove when https://gitlab.com/graphviz/graphviz/-/issues/1787 is
    # fixed
    if platform.system() == 'Windows' and \
       os.environ.get('build_system') == 'msbuild' and \
       os.environ.get('configuration') == 'Debug' and \
       TESTNAME == 'user_shapes':
      print('Skipping test {0}:{1} : using shapefile because it fails '
            'with Windows MSBuild Debug builds (#1787)'
            .format(TESTNAME, i),
            file=sys.stderr)
      continue
    # FIXME: Remove when https://gitlab.com/graphviz/graphviz/-/issues/1790 is
    # fixed
    if platform.system() == 'Windows' and \
       TESTNAME == 'ps_user_shapes':
      print('Skipping test {0}:{1} : using PostScript shapefile because it '
            'fails with Windows builds (#1790)'
            .format(TESTNAME, i),
            file=sys.stderr)
      continue

    result = subprocess.Popen(
       testcmd,
       universal_newlines=True,
       stderr = subprocess.PIPE,
    )
    _, errout = result.communicate()
    RVAL = result.returncode

    if errout:
      print(errout)

    if RVAL != 0 or not os.path.exists(OUTPATH):
      CRASH_CNT += 1
      print('Test {0}:{1} : == Layout failed =='.format(TESTNAME, i), file=sys.stderr)
      print('  ' + ' '.join(testcmd), file=sys.stderr)
    elif GENERATE:
      continue
    elif os.path.exists(os.path.join(REFDIR, OUTFILE)):
      doDiff(OUTFILE, OUTDIR, REFDIR, TESTNAME, i, SUBTEST['FMT'])
    else:
      print('Test {0}:{1} : == No file {2}/{3} for comparison =='.format(TESTNAME, i, REFDIR, OUTFILE), file=sys.stderr)

  # clear TESTTYPES
  TESTTYPES = {}

def cleanup():
  shutil.rmtree(TMPFILE1, ignore_errors=True)
  shutil.rmtree(TMPFILE2, ignore_errors=True)
  shutil.rmtree(TMPINFILE, ignore_errors=True)
atexit.register(cleanup)

# Set REFDIR
REFDIR = os.environ.get('REFDIR')
if not REFDIR:
  if platform.system() == 'Linux':
    REFDIR = 'linux.x86'
  elif platform.system() == 'Darwin':
    REFDIR = 'macosx'
  elif platform.system() == 'Windows':
    REFDIR = 'nshare'
  else:
    print('Unrecognized system "{0}"'.format(platform.system()), file=sys.stderr)
    REFDIR = 'nshare'

parser = argparse.ArgumentParser(description='Run regression tests.')
parser.add_argument('-g',
                    dest='generate',
                    action='store_true',
                    help='generate test data'
)
parser.add_argument('-v',
                    dest='verbose',
                    action='store_true',
                    help='verbose'
)
parser.add_argument('-n',
                    dest='noop',
                    action='store_true',
                    help='noop'
)
parser.add_argument('testfile',
                    nargs='?',
                    help='test files'
)
args = parser.parse_args()
VERBOSE = args.verbose or args.noop
NOOP = args.noop
GENERATE = args.generate
if GENERATE:
  if not os.path.isdir(REFDIR):
    os.mkdir(OUTDIR)
  OUTDIR = REFDIR

if args.testfile:
  if os.path.exists(args.testfile):
    TESTFILE = args.testfile
  else:
    print('Test file {0} does not exist'.format(args.testfile), file=sys.stderr)
    sys.exit(1)

# Check environment and initialize

if not NOOP:
  if not os.path.isdir(REFDIR):
    print('Test data directory {0} does not exist'.format(REFDIR),
          file=sys.stderr)
    sys.exit(1)

if not os.path.isdir(OUTDIR):
  os.mkdir(OUTDIR)

if not os.path.isdir(OUTHTML):
  os.mkdir(OUTHTML)
for filename in os.listdir(OUTHTML):
  os.unlink(os.path.join(OUTHTML, filename))

if not DOT:
  print('Could not find a value for DOT', file=sys.stderr)
  sys.exit(1)
if not os.path.isfile(DOT) or not os.access(DOT, os.X_OK):
  print('{0} program is not executable'.format(DOT))
  sys.exit(1)

if not GENERATE:
  if DIFFIMG:
    if not os.path.isfile(DIFFIMG) or not os.access(DIFFIMG, os.X_OK):
      print('{0} program is not executable'.format(DIFFIMG))
      sys.exit(1)
  else:
    print('Could not find a value for DIFFIMG', file=sys.stderr)
    # FIXME: Remove workaround for missing diffimg when
    # https://gitlab.com/graphviz/graphviz/-/issues/1788 is fixed
    if os.environ.get('build_system') != 'cmake':
      sys.exit(1)
#    sys.exit(1)


f3 = open(TESTFILE)
while True:
  TEST = readTest()
  if TEST is None:
     break
  doTest(TEST)
if NOOP:
  print('No. tests: ' + str(TOT_CNT), file=sys.stderr)
elif GENERATE:
  print('No. tests: ' + str(TOT_CNT) + ' Layout failures: ' + str(CRASH_CNT), file=sys.stderr)
else:
  print('No. tests: ' +
        str(TOT_CNT) + ' Layout failures: ' +
        str(CRASH_CNT) + ' Changes: ' +
        str(DIFF_CNT), file=sys.stderr)
EXIT_STATUS = CRASH_CNT + DIFF_CNT
sys.exit(EXIT_STATUS)
