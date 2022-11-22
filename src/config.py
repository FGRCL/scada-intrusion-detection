from pathlib import Path

data_file_path = Path(__file__).parent.parent / 'data' / 'gas_final.arff'
model_out_file_path = Path(__file__).parent.parent / 'out' / 'models'
tuning_out_file_path = Path(__file__).parent.parent / 'out' / 'tuning'
f_score_beta = 1
verbosity = 2