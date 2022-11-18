# botnet-attack-detection
botnet attack detection using anomaly detection and classification

## Setting up your environments

### Requirements
You'll need the following software installed to setup your environment.
- [python 3.10](https://www.python.org/)
- [pipenv](https://pipenv.pypa.io/en/latest/)

### Development Environment
1. Clone the repository
```shell
git clone git@github.com:FGRCL/botnet-attack-detection.git
```
2. install the environment
```shell
pipenv install
```
3. Change your IDE's python interpreter to the one pipenv just created for the proejct
    - [Follow this guide if you use intellij](https://www.jetbrains.com/help/idea/pipenv.html)
    - [Follow this guide if you use pycharm](https://www.jetbrains.com/help/pycharm/pipenv.html)
    - [Follow this guide if you use vs-code](https://code.visualstudio.com/docs/python/environments#_work-with-python-interpreters)

### Getting the data

1. Get the dataset from [ece.uah.edu](http://www.ece.uah.edu/~thm0009/icsdatasets/gas_final.arff) or ask one of the contributors for a copy of the dataset
2. Add the datasets file to a folder called `data` at the root of the directory.

![img.png](img.png)