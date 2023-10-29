import os
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--first', type=str, required=True)
parser.add_argument('--second', type=str, required=True)

args = parser.parse_args()
first = args.first
second = args.second


with open(first, 'r') as f:
    spec_params = yaml.safe_load(f)
with open(second, 'r') as f:
    spec_params.update(yaml.safe_load(f))

with open('spec_params.yaml', 'w') as f:
     yaml.dump(spec_params, f)
