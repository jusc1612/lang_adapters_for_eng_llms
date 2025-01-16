import os
import sys
from jinja2 import Environment, FileSystemLoader
import yaml
import ast

template_file = sys.argv[1]
task_id = int(sys.argv[2])

# Set up the Jinja2 environment
env = Environment(loader=FileSystemLoader('.'))
template = env.get_template(template_file)
rendered_yaml = template.render(env=os.getenv)

data = yaml.safe_load(rendered_yaml)
config = data["configs"][task_id]

for key, value in config.items():
    print(f"--{key} {value}")

'''for key, value in config.items():
    if key == 'inp_passage':
        print(f'--{key} "{value}"')
    else:
        print(f'--{key} {value}')'''

