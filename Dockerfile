FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir \
    jupyterlab \
    ipywidgets


WORKDIR /project
RUN mkdir /project/src
COPY ./src /project/src
COPY ./pyproject.toml /project
RUN python -m pip install -e /project