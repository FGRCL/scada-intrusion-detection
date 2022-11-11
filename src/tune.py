from argparse import ArgumentParser
from json import dump

from pandas import DataFrame

from src import config
from src.config import tuning_out_file_path
from src.models.randomforest import RandomForest


def main():
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--verbosity", "-v")
    argument_parser.add_argument("--randomforest", action='store_true')

    args = argument_parser.parse_args()
    config.verbosity = int(args.verbosity)
    metrics = []
    if args.randomforest:
        model = RandomForest()
        results = model.tune()
        metrics.append(('randomforest', results))

    for metric in metrics:
        name, results = metric
        metric_dataframe = DataFrame(results)
        print(metric_dataframe)
        with open(tuning_out_file_path / f'{name}.json', 'w') as f:
            f.write(metric_dataframe.to_json())


if __name__ == "__main__":
    main()
