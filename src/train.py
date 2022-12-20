from argparse import ArgumentParser
from pickle import dump

from src import config
from src.config import model_out_file_path
from src.data.gaspipeline import load_gaspipeline_dataset
from src.models.adaboost import AdaBoostTrainer
from src.models.gmm import GmmTrainer
from src.models.kmeans import KMeansTrainer
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
        ('adaboost', AdaBoostTrainer)
    ]

    argument_parser = ArgumentParser()
    argument_parser.add_argument("--verbosity", "-v")
    for name, _ in model_arguments:
        argument_parser.add_argument(f'--{name}', action='store_true')

    args = argument_parser.parse_args()
    if args.verbosity is not None:
        config.verbosity = int(args.verbosity)
    trained_models = []
    for name, trainer in model_arguments:
        if getattr(args, name):
            model = trainer()
            model.train()
            trained_models.append((name, model))


    for name, model in trained_models:
        with open(model_out_file_path / f'{name}.pkl', 'wb') as f:
            dump(model, f)
        print(model.get_metrics())

if __name__ == "__main__":
    main()