from string import Template
import os

def render(template_path, output_path, **kwargs):
    '''
    Returns True if a new file was written or if the file was modified.
    '''
    with open(template_path) as f:
        template = Template(f.read())
    rendered = template.substitute(kwargs)
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            if f.read() == rendered:
                return False
    with open(output_path, "w") as f:
        f.write(rendered)
    return True
