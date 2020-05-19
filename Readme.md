# Baysian-modeling-projects

This is home to a set of miscellaneous of Baysian modeling projects developed using [tensorflow probability](https://www.tensorflow.org/probability). My motivation for developing this is primary as a learning tool to improve my understanding of tensorflow and baysian inference. The plan is to collect a number of public domain examples and work them using a common framework and codebase using tensorflow probability. I will do my best to document source material and any changes/modifications.

## Installation

I maintain dependencies using pipenv. Most work will be done in a notebook dev environment which can be installed and activated using:

```bash
pipenv install --dev
pipenv shell
```

I also created a docker environment that encapsulates (with too many layers) the entire development framework.

## Usage

A jupyterlab notebook can be activated by running the shell script `runjupyter.sh` in the notebooks directory or using docker.

```bash
pipenv run notebooks/runjupyter.sh
```

```bash
docker-compose up
```

## Contributing
I have started this project primarily for my own personal development, but collaborators are, of course, welcome.

## License
[MIT](https://choosealicense.com/licenses/mit/)
