# Baysian-modeling-projects

This is home to a set of miscellaneous of Bayesian modeling projects developed using [tensorflow probability](https://www.tensorflow.org/probability) and/or [pymc3](https://docs.pymc.io/). My motivation for developing this is primary as a learning too. The plan is to collect a number of public domain examples and work them using a common framework and codebase using tensorflow probability or pymc3. I will do my best to document source material and any changes/modifications.

## Installation

I maintain dependencies using pipenv. Most work will be done in a notebook dev environment which can be installed and activated using:

```bash
pipenv install --dev
pipenv shell
```

I also created a docker environment that encapsulates (with too many layers) the entire development framework.

## Usage

A jupyterlab notebook can be activated by running the shell script `runjupyter.sh` in the notebooks directory or using docker:

```bash
pipenv run notebooks/runjupyter.sh
```

```bash
docker-compose up
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
