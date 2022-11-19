from argparse import ArgumentParser
from pickle import dump

from src import config
from src.config import model_out_file_path
from src.data.gaspipeline import load_gaspipeline_dataset
from src.models.randomforest import RandomForestClassification


def main():
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--verbosity", "-v")
    argument_parser.add_argument("--randomforest", action='store_true')

    args = argument_parser.parse_args()
    if args.verbosity is not None:
        config.verbosity = int(args.verbosity)
    models = []
    if args.randomforest:
        model = RandomForestClassification()
        model.train()
        models.append(('randomforest', model))

    for name, model in models:
        with open(model_out_file_path / f'{name}.pkl', 'wb') as f:
            dump(model, f)
        print(model.get_metrics())

if __name__ == "__main__":
    main()