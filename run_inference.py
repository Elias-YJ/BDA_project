from modelling import inference
import argparse

parser = argparse.ArgumentParser(description='Run inference')
parser.add_argument('-d', '--dir', default='inference')
parser.add_argument('-o', '--only', choices=['logreg', 'hier'], dest='model_list')
parser.add_argument('-c', '--hier-column', dest='hier_column')

args = parser.parse_args()

inference.run_models(output_dir=args.dir)