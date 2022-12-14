from argparse import ArgumentParser
from json import dump

from pandas import DataFrame

from src import config
from src.config import tuning_out_file_path
from src.models.gmm import GmmTrainer
from src.models.kmeans import KMeansTrainer
from src.models.pca import PcaTrainer
from src.models.randomforest import RandomForestClassification
from src.models.svm import SvmTrainer


def main():
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--verbosity", "-v")
    argument_parser.add_argument("--randomforest", action='store_true')
    argument_parser.add_argument("--kmeans", action='store_true')
    argument_parser.add_argument("--pca", action='store_true')
    argument_parser.add_argument("--gmm", action='store_true')
    argument_parser.add_argument("--svm", action='store_true')

    args = argument_parser.parse_args()
    if args.verbosity is not None:
        config.verbosity = int(args.verbosity)
    metrics = []
    if args.randomforest:
        model = RandomForestClassification()
        results = model.tune()
        metrics.append(('randomforest', results))
    if args.kmeans:
        model = KMeansTrainer()
        results = model.tune()
        metrics.append(('kmeans', results))
    if args.pca:
        model = PcaTrainer()
        results = model.tune()
        metrics.append(('pca', results))
    if args.gmm:
        model = GmmTrainer()
        results = model.tune()
        metrics.append(('gmm', results))
    if args.svm:
        model = SvmTrainer()
        results = model.tune()
        metrics.append(('svm', results))

    for metric in metrics:
        name, results = metric
        metric_dataframe = DataFrame(results)
        print(metric_dataframe)
        with open(tuning_out_file_path / f'{name}.json', 'w') as f:
            f.write(metric_dataframe.to_json())


if __name__ == "__main__":
    main()
