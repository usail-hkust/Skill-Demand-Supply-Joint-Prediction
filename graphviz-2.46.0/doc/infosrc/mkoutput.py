#!/usr/bin/env python3
# Takes `outputs` as stdin and generates output.html
# Uses `templates/output.html.j2`
# See `outputs` file for format documentation.

import markupsafe
import re
import sys
from typing import Dict, Tuple
import templates

HEADER_RE = re.compile(r'^:(?P<params>[^:]+):(?P<format>.*)')

# Tuple of command-line-params for an output format, e.g. ('jpg', 'jpeg', 'jpe')
params : Tuple[str, ...] = ()

# Map from tuple of command-line-params to full name of the output format
formats : Dict[Tuple[str, ...], str] = {}

# Map from tuple of command-line-params to an HTML description string
html_descriptions : Dict[Tuple[str, ...], str]  = {}

for line in sys.stdin:
    # Skip comment lines.
    if line.startswith('#'):
        continue

    m = HEADER_RE.match(line)
    if m:
        # This is a header line. Grab out the values.

        # Command-line formats are slash-separated.
        params = tuple(m.group('params').split('/'))

        # Full format name is plain text
        formats[params] = m.group('format')

        # Set an empty string html description, ready to append to.
        html_descriptions[params] = ''
    else:
        # This is an HTML line, possibly a continuation of a previous HTML line.
        html_descriptions[params] += line


template = templates.env().get_template('output.html.j2')
print(template.render(
    formats=formats,
    # Vouch for the HTML descriptions as being safe and not needing auto-HTML-escaping.
    # This is reasonable because the HTML descriptions are not attacker-controlled.
    descriptions={
        params: markupsafe.Markup(desc)
        for params, desc in html_descriptions.items()
    }
))
