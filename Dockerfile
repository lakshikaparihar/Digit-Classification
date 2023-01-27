FROM ubuntu:latest
RUN apt-get -y update

RUN apt-get -y upgrade && apt-get -y update && apt-get install -y curl nginx && apt install -y wget && rm -rf /var/lib/apt/lists/*

# Download and setup miniconda
#RUN curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh >> miniconda.sh
#RUN bash ./miniconda.sh -b -p /miniconda; rm ./miniconda.sh;
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

# install in batch (silent) mode, does not edit PATH or .bashrc or .bash_profile
# -p path
# -f force
RUN bash Miniconda3-latest-Linux-x86_64.sh -b
ENV PATH="/miniconda/bin:$PATH"
ENV GUNICORN_CMD_ARGS="--timeout 60 -k gevent --access-logfile - --capture-output --error-logfile preprocess_logs.txt"

WORKDIR /opt/mlflow
COPY mlruns/0/5a779713056440f5b66d27161c02f4a0/artifacts/digit-classification-model .

RUN apt-get -y autoremove && apt-get -y autoclean && apt-get -y update && apt-get install -y python3 && apt install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install mlflow==1.26.1

RUN mkdir -p /opt/ml/model
RUN cp -r * /opt/ml/model/
#RUN python3 -c                 'from mlflow.models.container import _install_pyfunc_deps;                _install_pyfunc_deps(                    "/opt/ml/model",                     install_mlflow=False,                     enable_mlserver=False,                     env_manager="conda")'