# ARG BASE_IMAGE=sinzlab/pytorch:v3.9-torch1.9.0-cuda11.1-dj0.12.7

# # Perform multistage build to pull private repo without leaving behind
# # private information (e.g. SSH key, Git token)
# FROM ${BASE_IMAGE} as base
# ARG GITHUB_USER
# ARG DEV_SOURCE
# ARG GITHUB_TOKEN

# WORKDIR /src

FROM sinzlab/pytorch:v3.9-torch1.9.0-cuda11.1-dj0.12.7
# COPY --from=base /src /src

# lines below are necessasry to fix an issue explained here: https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# install screen
# RUN apt-get -y update && apt-get install -y \
#     screen

# install third-party libraries
RUN python -m pip install --upgrade pip
# RUN python -m pip --no-cache-dir install \
#     tqdm \
#     ipdb \
#     lipstick \
#     # git+https://github.com/mohammadbashiri/classicalv1@master \
#     # git+https://github.com/sinzlab/sensorium@main \
#     # git+https://github.com/sacadena/ptrnets \
#     # git+https://github.com/dicarlolab/CORnet \
#     git+https://github.com/lucabaroni/featurevis_mod@v0.1 \
#     git+https://github.com/sinzlab/nnvision@v0.1.1 \
#     git+https://github.com/sinzlab/neuralpredictors@v0.3.0 \
#     numpy==1.22.0 \
#     matplotlib==3.4.3

WORKDIR /project
COPY ./requirements.txt /project
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt
RUN python -m pip --no-cache-dir install \
    lipstick \
    matplotlib==3.4.3

# install the cloned repos
# RUN git clone --depth 1 --branch v0.1.1 https://github.com/sinzlab/nnvision.git
# RUN python -m pip install -e /src/nnvision &&\
#     python -m pip --no-cache-dir install neuralpredictors==0.3.0

# install the current project
# WORKDIR /project
RUN mkdir /project/src
COPY ./src /project/src
COPY ./pyproject.toml /project
RUN python -m pip install -e /project

WORKDIR /project/notebooks
COPY ./jupyter_notebook_config.py /root/.jupyter/