import jinja2

def env():
  return jinja2.Environment(
      # Load template files from ./templates/
      loader=jinja2.PackageLoader(
          package_name='templates',
          package_path='templates',
      ),
      # Auto-HTML-escape any html or xml files.
      autoescape=jinja2.select_autoescape(['html', 'xml', 'html.j2', 'xml.j2']),
      # Whitespace control
      trim_blocks=True,
      lstrip_blocks=True,
      # Raise exception on any attempt to access undefined variables.
      undefined=jinja2.StrictUndefined,
  )
