import pytest
import platform
import shutil
import signal
import subprocess
import os
import re
import stat
import tempfile

# The terminology used in rtest.py is a little inconsistent. At the
# end it reports the total number of tests, the number of "failures"
# (crashes) and the number of "changes" (which is the number of tests
# where the output file did not match the reference file). However,
# for each test that detects "changes", it prints an error message
# saying "== Failed ==" which thus is not counted as a failure at the
# end.

# First run a subset of all the tests that produces equal output files
# for all platforms and fail the test if there are differences.

def test_regression_subset_differences():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    subprocess.check_call(['python3', 'rtest.py', 'tests_subset.txt'])

# Secondly, run all tests but ignore differences and fail the test
# only if there is a crash. This will leave the differences for png
# output in rtest/nhtml/index.html for review.

def test_regression_failure():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    result = subprocess.Popen(['python3', 'rtest.py'], stderr=subprocess.PIPE,
                              universal_newlines=True)
    text = result.communicate()[1]
    print(text)
    assert "Layout failures: 0" in text
# FIXME: re-enable when all tests pass on all platforms
#    assert result.returncode == 0

def test_165():
    '''
    dot should be able to produce properly escaped xdot output
    https://gitlab.com/graphviz/graphviz/-/issues/165
    '''

    # locate our associated test case in this directory
    input = os.path.join(os.path.dirname(__file__), '165.dot')
    assert os.path.exists(input), 'unexpectedly missing test case'

    # ask Graphviz to translate it to xdot
    output = subprocess.check_output(['dot', '-Txdot', input],
      universal_newlines=True)

    # find the line containing the _ldraw_ attribute
    ldraw = re.search(r'^\s*_ldraw_\s*=(?P<value>.*?)$', output, re.MULTILINE)
    assert ldraw is not None, 'no _ldraw_ attribute in graph'

    # this should contain the label correctly escaped
    assert r'hello \\\" world' in ldraw.group('value'), \
      'unexpected ldraw contents'

def test_165_2():
    '''
    variant of test_165() that checks a similar problem for edges
    https://gitlab.com/graphviz/graphviz/-/issues/165
    '''

    # locate our associated test case in this directory
    input = os.path.join(os.path.dirname(__file__), '165_2.dot')
    assert os.path.exists(input), 'unexpectedly missing test case'

    # ask Graphviz to translate it to xdot
    output = subprocess.check_output(['dot', '-Txdot', input],
      universal_newlines=True)

    # find the lines containing _ldraw_ attributes
    ldraw = re.findall(r'^\s*_ldraw_\s*=(.*?)$', output, re.MULTILINE)
    assert ldraw is not None, 'no _ldraw_ attributes in graph'

    # one of these should contain the label correctly escaped
    assert any(r'hello \\\" world' in l for l in ldraw), \
      'unexpected ldraw contents'

def test_165_3():
    '''
    variant of test_165() that checks a similar problem for graph labels
    https://gitlab.com/graphviz/graphviz/-/issues/165
    '''

    # locate our associated test case in this directory
    input = os.path.join(os.path.dirname(__file__), '165_3.dot')
    assert os.path.exists(input), 'unexpectedly missing test case'

    # ask Graphviz to translate it to xdot
    output = subprocess.check_output(['dot', '-Txdot', input],
      universal_newlines=True)

    # find the lines containing _ldraw_ attributes
    ldraw = re.findall(r'^\s*_ldraw_\s*=(.*?)$', output, re.MULTILINE)
    assert ldraw is not None, 'no _ldraw_ attributes in graph'

    # one of these should contain the label correctly escaped
    assert any(r'hello \\\" world' in l for l in ldraw), \
      'unexpected ldraw contents'

def test_167():
    '''
    using concentrate=true should not result in a segfault
    https://gitlab.com/graphviz/graphviz/-/issues/167
    '''

    # locate our associated test case in this directory
    input = os.path.join(os.path.dirname(__file__), '167.dot')
    assert os.path.exists(input), 'unexpectedly missing test case'

    # process this with dot
    ret = subprocess.call(['dot', '-Tpdf', '-o', os.devnull, input])

    # Graphviz should not have caused a segfault
    assert ret != -signal.SIGSEGV, 'Graphviz segfaulted'

def test_793():
    '''
    Graphviz should not crash when using VRML output with a non-writable current
    directory
    https://gitlab.com/graphviz/graphviz/-/issues/793
    '''

    # create a non-writable directory
    with tempfile.TemporaryDirectory() as tmp:
      os.chmod(tmp, os.stat(tmp).st_mode & ~stat.S_IWRITE)

      # ask the VRML back end to handle a simple graph, using the above as the
      # current working directory
      p = subprocess.Popen(['dot', '-Tvrml', '-o', os.devnull], cwd=tmp)
      p.communicate('digraph { a -> b; }')

    # Graphviz should not have caused a segfault
    assert p.returncode != -signal.SIGSEGV, 'Graphviz segfaulted'

def test_1221():
    '''
    assigning a node to two clusters with newrank should not cause a crash
    https://gitlab.com/graphviz/graphviz/-/issues/1221
    '''

    # locate our associated test case in this directory
    input = os.path.join(os.path.dirname(__file__), '1221.dot')
    assert os.path.exists(input), 'unexpectedly missing test case'

    # process this with dot
    subprocess.check_call(['dot', '-Tsvg', '-o', os.devnull, input])

def test_1314():
    '''
    test that a large font size that produces an overflow in Pango is rejected
    https://gitlab.com/graphviz/graphviz/-/issues/1314
    '''

    # locate our associated test case in this directory
    input = os.path.join(os.path.dirname(__file__), '1314.dot')
    assert os.path.exists(input), 'unexpectedly missing test case'

    # ask Graphviz to process it, which should fail
    try:
      subprocess.check_call(['dot', '-Tsvg', '-o', os.devnull, input])
    except subprocess.CalledProcessError:
      return

    # the execution did not fail as expected
    pytest.fail('dot incorrectly exited with success')

def test_1411():
    '''
    parsing strings containing newlines should not disrupt line number tracking
    https://gitlab.com/graphviz/graphviz/-/issues/1411
    '''

    # locate our associated test case in this directory
    input = os.path.join(os.path.dirname(__file__), '1411.dot')
    assert os.path.exists(input), 'unexpectedly missing test case'

    # process it with Graphviz (should fail)
    p = subprocess.Popen(['dot', '-Tsvg', '-o', os.devnull, input],
      stderr=subprocess.PIPE, universal_newlines=True)
    _, output = p.communicate()

    assert p.returncode != 0, 'Graphviz accepted broken input'

    assert 'syntax error in line 17 near \'\\\'' in output, \
      'error message did not identify correct location'

def test_1436():
    '''
    test a segfault from https://gitlab.com/graphviz/graphviz/-/issues/1436 has
    not reappeared
    '''

    # locate our associated test case in this directory
    input = os.path.join(os.path.dirname(__file__), '1436.dot')
    assert os.path.exists(input), 'unexpectedly missing test case'

    # ask Graphviz to process it, which should generate a segfault if this bug
    # has been reintroduced
    subprocess.check_call(['dot', '-Tsvg', '-o', os.devnull, input])

def test_1444():
    '''
    specifying 'headport' as an edge attribute should work regardless of what
    order attributes appear in
    https://gitlab.com/graphviz/graphviz/-/issues/1444
    '''

    # locate the first of our associated tests
    input1 = os.path.join(os.path.dirname(__file__), '1444.dot')
    assert os.path.exists(input1), 'unexpectedly missing test case'

    # ask Graphviz to process it
    p = subprocess.Popen(['dot', '-Tsvg', input1], stdout=subprocess.PIPE,
      stderr=subprocess.PIPE, universal_newlines=True)
    stdout1, stderr = p.communicate()

    assert p.returncode == 0, 'failed to process a headport edge'

    assert stderr.strip() == '', 'emitted an error for a legal graph'

    # now locate our second variant, that simply has the attributes swapped
    input2 = os.path.join(os.path.dirname(__file__), '1444-2.dot')
    assert os.path.exists(input2), 'unexpectedly missing test case'

    # process it identically
    p = subprocess.Popen(['dot', '-Tsvg', input2], stdout=subprocess.PIPE,
      stderr=subprocess.PIPE, universal_newlines=True)
    stdout2, stderr = p.communicate()

    assert p.returncode == 0, 'failed to process a headport edge'

    assert stderr.strip() == '', 'emitted an error for a legal graph'

    assert stdout1 == stdout2, \
      'swapping edge attributes altered the output graph'

def test_1449():
    '''
    using the SVG color scheme should not cause warnings
    https://gitlab.com/graphviz/graphviz/-/issues/1449
    '''

    # start Graphviz
    p = subprocess.Popen(['dot', '-Tsvg', '-o', os.devnull],
      stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
      universal_newlines=True)

    # pass it some input that uses the SVG color scheme
    stdout, stderr = p.communicate('graph g { colorscheme="svg"; }')

    assert p.returncode == 0, 'Graphviz exited with non-zero status'

    assert stderr.strip() == '', 'SVG color scheme use caused warnings'

@pytest.mark.skipif(shutil.which('gvpr') is None, reason='GVPR not available')
def test_1594():
    '''
    GVPR should give accurate line numbers in error messages
    https://gitlab.com/graphviz/graphviz/-/issues/1594
    '''

    # locate our associated test case in this directory
# FIXME: remove cwd workaround when
# https://gitlab.com/graphviz/graphviz/-/issues/1780 is fixed
#    input = os.path.join(os.path.dirname(__file__), '1594.gvpr')
    input = '1594.gvpr'

    # run GVPR with our (malformed) input program
    p = subprocess.Popen(['gvpr', '-f', input], stdin=subprocess.PIPE,
      cwd=os.path.join(os.path.dirname(__file__)),
      stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    _, stderr = p.communicate()

    assert p.returncode != 0, 'GVPR did not reject malformed program'

    assert 'line 3:' in stderr, \
      'GVPR did not identify correct line of syntax error'

def test_1676():
    '''
    https://gitlab.com/graphviz/graphviz/-/issues/1676
    '''

    # locate our associated test case in this directory
    input = os.path.join(os.path.dirname(__file__), '1676.dot')
    assert os.path.exists(input), 'unexpectedly missing test case'

    # run Graphviz with this input
    ret = subprocess.call(['dot', '-Tsvg', '-o', os.devnull, input])

    # this malformed input should not have caused Graphviz to crash
    assert ret != -signal.SIGSEGV, 'Graphviz segfaulted'

def test_1724():
    '''
    passing malformed node and newrank should not cause segfaults
    https://gitlab.com/graphviz/graphviz/-/issues/1724
    '''

    # locate our associated test case in this directory
    input = os.path.join(os.path.dirname(__file__), '1724.dot')
    assert os.path.exists(input), 'unexpectedly missing test case'

    # run Graphviz with this input
    ret = subprocess.call(['dot', '-Tsvg', '-o', os.devnull, input])

    assert ret != -signal.SIGSEGV, 'Graphviz segfaulted'

def test_1767():
    '''
    using the Pango plugin multiple times should produce consistent results
    https://gitlab.com/graphviz/graphviz/-/issues/1767
    '''

    # FIXME: Remove skip when
    # https://gitlab.com/graphviz/graphviz/-/issues/1777 is fixed
    if os.getenv('build_system') == 'msbuild':
      pytest.skip('Windows MSBuild release does not contain any header files (#1777)')

    # find co-located test source
    c_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '1767.c'))
    assert os.path.exists(c_src), 'missing test case'

    # create some scratch space to work in
    with tempfile.TemporaryDirectory() as tmp:

      # compile our test code
      exe = os.path.join(tmp, 'a.exe')
      rt_lib_option = '-MDd' if os.environ.get('configuration') == 'Debug' else '-MD'

      if platform.system() == 'Windows':
          subprocess.check_call(['cl', c_src, '-Fe:', exe, '-nologo',
            rt_lib_option, '-link', 'cgraph.lib', 'gvc.lib'])
      else:
          cc = os.environ.get('CC', 'cc')
          subprocess.check_call([cc, c_src, '-o', exe, '-lcgraph', '-lgvc'])

      # find our co-located dot input
      dot = os.path.abspath(os.path.join(os.path.dirname(__file__), '1767.dot'))
      assert os.path.exists(dot), 'missing test case'

      # run the test
      stdout = subprocess.check_output([exe, dot], universal_newlines=True)

      # FIXME: uncomment this when #1767 is fixed
      # assert stdout == 'Loaded graph:clusters\n' \
      #                  'cluster_0 contains 5 nodes\n' \
      #                  'cluster_1 contains 1 nodes\n' \
      #                  'cluster_2 contains 3 nodes\n' \
      #                  'cluster_3 contains 3 nodes\n' \
      #                  'Loaded graph:clusters\n' \
      #                  'cluster_0 contains 5 nodes\n' \
      #                  'cluster_1 contains 1 nodes\n' \
      #                  'cluster_2 contains 3 nodes\n' \
      #                  'cluster_3 contains 3 nodes\n'

def test_1783():
    '''
    Graphviz should not segfault when passed large edge weights
    https://gitlab.com/graphviz/graphviz/-/issues/1783
    '''

    # locate our associated test case in this directory
    input = os.path.join(os.path.dirname(__file__), '1783.dot')
    assert os.path.exists(input), 'unexpectedly missing test case'

    # run Graphviz with this input
    ret = subprocess.call(['dot', '-Tsvg', '-o', os.devnull, input])

    assert ret != 0, 'Graphviz accepted illegal edge weight'

    assert ret != -signal.SIGSEGV, 'Graphviz segfaulted'

@pytest.mark.skipif(shutil.which('gvedit') is None, reason='Gvedit not available')
def test_1813():
    '''
    gvedit -? should show usage
    https://gitlab.com/graphviz/graphviz/-/issues/1813
    '''

    environ_copy = os.environ.copy()
    environ_copy.pop('DISPLAY', None)
    output = subprocess.check_output(['gvedit', '-?'],
      env=environ_copy,
      universal_newlines=True)

    assert 'Usage' in output, 'gvedit -? did not show usage'

@pytest.mark.skipif(shutil.which('lefty') is None, reason='Lefty not available')
def test_1818():
    '''
    lefty -? should show usage
    https://gitlab.com/graphviz/graphviz/-/issues/1813
    '''

    output = subprocess.check_output(['lefty', '-?'],
      stderr=subprocess.STDOUT,
      universal_newlines=True)

    assert 'Usage' in output, 'lefty -? did not show usage'

def test_1845():
    '''
    rendering sequential graphs to PS should not segfault
    https://gitlab.com/graphviz/graphviz/-/issues/1845
    '''

    # locate our associated test case in this directory
    input = os.path.join(os.path.dirname(__file__), '1845.dot')
    assert os.path.exists(input), 'unexpectedly missing test case'

    # generate a multipage PS file from this input
    subprocess.check_call(['dot', '-Tps', '-o', os.devnull, input])

@pytest.mark.skipif(shutil.which('fdp') is None, reason='fdp not available')
def test_1865():
    '''
    fdp should not read out of bounds when processing node names
    https://gitlab.com/graphviz/graphviz/-/issues/1865
    Note, the crash this test tries to provoke may only occur when run under
    Address Sanitizer or Valgrind
    '''

    # locate our associated test case in this directory
    input = os.path.join(os.path.dirname(__file__), '1865.dot')
    assert os.path.exists(input), 'unexpectedly missing test case'

    # fdp should not crash when processing this file
    subprocess.check_call(['fdp', '-o', os.devnull, input])

@pytest.mark.skipif(shutil.which('fdp') is None, reason='fdp not available')
def test_1876():
    '''
    fdp should not rename nodes with internal names
    https://gitlab.com/graphviz/graphviz/-/issues/1876
    '''

    # a trivial graph to provoke this issue
    input = 'graph { a }'

    # process this with fdp
    p = subprocess.Popen(['fdp'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
      universal_newlines=True)
    output, _ = p.communicate(input)

    assert p.returncode == 0, 'fdp failed to process trivial graph'

    # we should not see any internal names like '%3'
    assert '%' not in output, 'internal name in fdp output'

@pytest.mark.skipif(shutil.which('fdp') is None, reason='fdp not available')
def test_1877():
    '''
    fdp should not fail an assertion when processing cluster edges
    https://gitlab.com/graphviz/graphviz/-/issues/1877
    '''

    # simple input with a cluster edge
    input = 'graph {subgraph cluster_a {}; cluster_a -- b}'

    # fdp should be able to process this
    p = subprocess.Popen(['fdp', '-o', os.devnull], stdin=subprocess.PIPE,
      universal_newlines=True)
    p.communicate(input)
    assert p.returncode == 0

def test_1898():
    '''
    test a segfault from https://gitlab.com/graphviz/graphviz/-/issues/1898 has
    not reappeared
    '''

    # locate our associated test case in this directory
    input = os.path.join(os.path.dirname(__file__), '1898.dot')
    assert os.path.exists(input), 'unexpectedly missing test case'

    # ask Graphviz to process it, which should generate a segfault if this bug
    # has been reintroduced
    subprocess.check_call(['dot', '-Tsvg', '-o', os.devnull, input])

# root directory of this checkout
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# find all HTML files
html = set()
for (prefix, _, files) in os.walk(ROOT):
  for name in files:
    if os.path.splitext(name)[-1].lower() in ('.htm', '.html'):
      html.add(os.path.join(prefix, name))

@pytest.mark.parametrize('src', html)
@pytest.mark.skipif(shutil.which('xmllint') is None, reason='xmllint not found')
def test_html(src: str):

  # Files that we know currently fail this test. If you fix one of these files,
  # remove it from this list.
  # See https://gitlab.com/graphviz/graphviz/-/issues/1861
  FAILING = frozenset(os.path.join(ROOT, x) for x in (
    'doc/FAQ.html',
    'doc/info/colors.html',
    'doc/info/output.html',
    'doc/info/shapes.html',
    'doc/internal_todo.html',
    'lib/inkpot/data/types.html',
    'macosx/graphviz.help/graphviz.html',
  ))

  # ensure this test fails if one of the above files is moved/deleted, to prompt
  # developers to update the list
  assert all(os.path.exists(x) for x in FAILING), 'missing file in FAILING list'

  # FIXME: the macOS Autotools CI build installs to a target within the source
  # tree, so we need to avoid picking up duplicating copies of the FAILING list
  # that are present there
  if platform.system() == 'Darwin' and \
      os.environ.get('build_system') == 'autotools' and \
      src.startswith(os.path.join(ROOT, 'build')):
    pytest.skip('skipping installed copied of source file')

  # validate the file
  p = subprocess.Popen(['xmllint', '--nonet', '--noout', '--html', '--valid',
    src], stderr=subprocess.PIPE, universal_newlines=True)
  _, stderr = p.communicate()

  # If this is expected to fail, demand that it does so. This way the test will
  # fail if someone fixes the file, prompting them to update this test.
  if src in FAILING:
    assert p.returncode != 0 or stderr != ''
    return

  # otherwise, expect it to succeed
  assert p.returncode == 0
  assert stderr == ''

@pytest.mark.parametrize('variant', [1, 2])
@pytest.mark.skipif(shutil.which('gml2gv') is None, reason='gml2gv not available')
def test_1869(variant: int):
    '''
    gml2gv should be able to parse the style, outlineStyle, width and
    outlineWidth GML attributes and map them to the DOT attributes
    style and penwidth respectively
    https://gitlab.com/graphviz/graphviz/-/issues/1869
    '''

    # locate our associated test case in this directory
    input = os.path.join(os.path.dirname(__file__), '1869-{}.gml'.format(variant))
    assert os.path.exists(input), 'unexpectedly missing test case'

    # ask gml2gv to translate it to DOT
    output = subprocess.check_output(['gml2gv', input],
      universal_newlines=True)

    assert 'style=dashed' in output, 'style=dashed not found in DOT output'
    assert 'penwidth=2' in output, 'penwidth=2 not found in DOT output'

@pytest.mark.skipif(shutil.which('twopi') is None, reason='twopi not available')
def test_1907():
    '''
    SVG edges should have title elements that match their names
    https://gitlab.com/graphviz/graphviz/-/issues/1907
    '''

    # a trivial graph to provoke this issue
    input = 'digraph { A -> B -> C }'

    # generate an SVG from this input with twopi
    p = subprocess.Popen(['twopi', '-Tsvg'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
      universal_newlines=True)
    output, _ = p.communicate(input)

    assert '<title>A&#45;&gt;B</title>' in output, \
      'element title not found in SVG'

@pytest.mark.skipif(shutil.which('gvpr') is None, reason='gvpr not available')
def test_1909():
    '''
    GVPR should not output internal names
    https://gitlab.com/graphviz/graphviz/-/issues/1909
    '''

    # locate our associated test case in this directory
    prog = os.path.join(os.path.dirname(__file__), '1909.gvpr')
    graph = os.path.join(os.path.dirname(__file__), '1909.dot')

    # run GVPR with the given input
    p = subprocess.Popen(['gvpr', '-c', '-f', prog, graph],
      stdout=subprocess.PIPE, universal_newlines=True)
    output, _ = p.communicate()

    # FIXME: for some undiagnosed reason, the above command fails on Windows
    if platform.system() == 'Windows':
      assert p.returncode != 0
      return

    assert p.returncode == 0, 'gvpr failed to process graph'

    # we should have produced this graph without names like "%2" in it
    assert output == '// begin\n' \
                     'digraph bug {\n' \
                     '	a -> b;\n' \
                     '	b -> c;\n' \
                     '}\n'

def test_1910():
    '''
    Repeatedly using agmemread() should have consistent results
    https://gitlab.com/graphviz/graphviz/-/issues/1910
    '''

    # FIXME: Remove skip when
    # https://gitlab.com/graphviz/graphviz/-/issues/1777 is fixed
    if os.getenv('build_system') == 'msbuild':
      pytest.skip('Windows MSBuild release does not contain any header files (#1777)')

    # find co-located test source
    c_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '1910.c'))
    assert os.path.exists(c_src), 'missing test case'

    # create some scratch space to work in
    with tempfile.TemporaryDirectory() as tmp:

      # compile our test code
      exe = os.path.join(tmp, 'a.exe')
      rt_lib_option = '-MDd' if os.environ.get('configuration') == 'Debug' else '-MD'

      if platform.system() == 'Windows':
          subprocess.check_call(['cl', c_src, '-Fe:', exe, '-nologo',
            rt_lib_option, '-link', 'cgraph.lib', 'gvc.lib'])
      else:
          cc = os.environ.get('CC', 'cc')
          subprocess.check_call([cc, c_src, '-o', exe, '-lcgraph', '-lgvc'])

      # run the test
      subprocess.check_call([exe])
