FROM python:3.7

RUN apt-get update \
 && apt-get install -yq --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    sudo \
    locales \
    fonts-liberation \
    libblas-dev\
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Node.js
RUN apt-get install --yes curl
RUN curl --silent --location https://deb.nodesource.com/setup_12.x | sudo bash -
RUN apt-get install --yes nodejs
RUN apt-get install --yes build-essential

# Bundle app source
# Trouble with COPY http://stackoverflow.com/a/30405787/2926832
COPY . /src
# Install app dependencies
RUN cd /src; npm install

RUN pip install --upgrade pip

RUN adduser worker
USER worker
WORKDIR /home/worker

RUN pip install --user pipenv

ENV PATH="/home/worker/.local/bin:${PATH}" \
    PIPENV_SHELL="$SHELL"

COPY --chown=worker:worker Pipfile Pipfile

RUN pipenv install --dev
RUN pipenv run jupyter labextension install @ryantam626/jupyterlab_code_formatter \
 && pipenv install jupyterlab_code_formatter \
 && pipenv run jupyter serverextension enable --py jupyterlab_code_formatter


COPY --chown=worker:worker ./notebooks/runjupyter.sh runjupyter.sh

CMD ["pipenv", "run", "./runjupyter.sh"]


