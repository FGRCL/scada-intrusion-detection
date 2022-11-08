from argparse import ArgumentParser

from src.metrics.classification import get_classification_metrics
from src.models.randomforest import RandomForest


def main():
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--randomforest", action='store_true')

    args = argument_parser.parse_args()
    models = []
    if args.randomforest:
        model = RandomForest()
        model.train()
        models.append(model)

    metrics = []
    for model in models:
        metrics.append(model.get_metrics())

    print(metrics)

if __name__ == "__main__":
    main()