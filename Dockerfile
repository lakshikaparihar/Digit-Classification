FROM ubuntu:latest
RUN apt-get -y update

RUN apt-get -y upgrade && apt-get -y update && apt-get install -y curl nginx && apt install -y wget && rm -rf /var/lib/apt/lists/*

# Download and setup miniconda
RUN curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh >> miniconda.sh
RUN bash ./miniconda.sh -b -p /miniconda; rm ./miniconda.sh;
ENV PATH="/miniconda/bin:$PATH"
ENV GUNICORN_CMD_ARGS="--timeout 60 -k gevent --access-logfile - --capture-output --error-logfile preprocess_logs.txt"

WORKDIR /opt/mlflow
COPY mlruns/0/fc000a46d6a641f1847140112ddc31bc/artifacts/digit-classification-model .

RUN apt-get -y autoremove && apt-get -y autoclean && apt-get -y update && apt-get install -y python3 && apt install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install mlflow==2.1.1

RUN mkdir -p /opt/ml/model
RUN cp -r * /opt/ml/model/
RUN python3 -c                 'from mlflow.models.container import _install_pyfunc_deps;                _install_pyfunc_deps(                    "/opt/ml/model",                     install_mlflow=False,                     enable_mlserver=False,                     env_manager="conda")'

ENV MLFLOW_DISABLE_ENV_CREATION="true"
RUN chmod o+rwX /opt/mlflow/
ENTRYPOINT ["python3", "-c", "from mlflow.models import container as C;C._serve('conda')"]