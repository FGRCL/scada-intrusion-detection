from argparse import ArgumentParser

from src.data.gaspipeline import load_gaspipeline_dataset
from src.models.randomforest import RandomForest


def main():
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--randomforest", action='store_true')

    args = argument_parser.parse_args()
    dataset = load_gaspipeline_dataset()
    metrics = []
    if args.randomforest:
        model = RandomForest(dataset)
        best_parameters = model.tune()
        metrics.append(best_parameters)

    print(metrics)

if __name__ == "__main__":
    main()