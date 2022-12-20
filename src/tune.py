from argparse import ArgumentParser
from json import dump

from pandas import DataFrame

from src import config
from src.config import tuning_out_file_path
from src.models.adaboost import AdaBoostTrainer
from src.models.gmm import GmmTrainer
from src.models.kmeans import KMeansTrainer
from src.models.knn import KnnTrainer
from src.models.pca import PcaTrainer
from src.models.randomforest import RandomForestClassification
from src.models.svm import SvmTrainer


def main():
    model_arguments = [
        ('randomforest', RandomForestClassification),
        ('kmeans', KMeansTrainer),
        ('pca', PcaTrainer),
        ('gmm', GmmTrainer),
        ('svm', SvmTrainer),
        ('adaboost', AdaBoostTrainer),
        ('knn', KnnTrainer)
    ]

    argument_parser = ArgumentParser()
    argument_parser.add_argument("--verbosity", "-v")
    for name, _ in model_arguments:
        argument_parser.add_argument(f'--{name}', action='store_true')

    args = argument_parser.parse_args()
    if args.verbosity is not None:
        config.verbosity = int(args.verbosity)
    metrics = []
    for name, trainer in model_arguments:
        if getattr(args, name):
            model = trainer()
            results = model.tune()
            metrics.append((name, results))

    for metric in metrics:
        name, results = metric
        metric_dataframe = DataFrame(results)
        print(metric_dataframe)
        with open(tuning_out_file_path / f'{name}.json', 'w') as f:
            f.write(metric_dataframe.to_json())


if __name__ == "__main__":
    main()
