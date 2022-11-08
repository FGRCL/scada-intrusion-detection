from argparse import ArgumentParser

from src.models.randomforest import RandomForest


def main():
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--randomforest", action='store_true')

    args = argument_parser.parse_args()
    metrics = []
    if args.randomforest:
        model = RandomForest()
        best_parameters = model.tune()
        metrics.append(best_parameters)

    print(metrics)

if __name__ == "__main__":
    main()