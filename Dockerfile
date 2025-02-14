FROM sinzlab/pytorch:v3.9-torch1.9.0-cuda11.1-dj0.12.7

RUN python -m pip install --upgrade pip

COPY ./jupyter_notebook_config.py /root/.jupyter/

WORKDIR /project
RUN mkdir /project/src
COPY ./src /project/src
COPY ./pyproject.toml /project
RUN python -m pip install -e /project