from argparse import ArgumentParser

from src import config
from src.data.gaspipeline import load_gaspipeline_dataset
from src.models.randomforest import RandomForest


def main():
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--verbosity", "-v")
    argument_parser.add_argument("--randomforest", action='store_true')

    args = argument_parser.parse_args()
    config.verbosity = int(args.verbosity)

    dataset = load_gaspipeline_dataset()
    models = []
    if args.randomforest:
        model = RandomForest(dataset)
        model.train()
        models.append(model)

    metrics = []
    for model in models:
        metrics.append(model.get_metrics())

    print(metrics)

if __name__ == "__main__":
    main()