from import_packages import *

# initialization & cleaning of df
def initialization(dataset_csv_path):
    # read contents of csv file
    df = pd.read_csv(dataset_csv_path)
    # adding header
    headerList = ['command_address', 'response_address', 'command_memory', 'response_memory', 'command_memory_count',
                  'response_memory_count', 'comm_read_function', 'comm_write_fun', 'resp_read_fun', 'resp_write_fun',
                  'sub_function', 'command_length', 'resp_length', 'gain', 'reset', 'deadband', 'cycletime', 'rate',
                  'setpoint', 'control_mode', 'control_scheme', 'pump', 'solenoid', 'crc_rate', 'measurement', 'time',
                  'result']
    # converting data frame to csv
    df.to_csv(r"results\Dataset.csv", header=headerList, index=False)
    df = pd.read_csv(r"results\Dataset.csv")
    # separate features & labels
    x = df.loc[:, df.columns != 'result']
    y = df['result']
    # find missing values
    print('... Checking missing values ... ')
    print(x.isnull().values.any())
    # check duplicated values
    print('... Checking duplicated values ... ')
    print(x.duplicated().sum())

    return df, x, y, headerList