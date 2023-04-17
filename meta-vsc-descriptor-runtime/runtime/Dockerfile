# syntax = docker/dockerfile:1.2
FROM nvidia/cuda:11.3.1-base-ubuntu20.04
# adapated from build file for pangeo images
# https://github.com/pangeo-data/pangeo-docker-images

ARG CPU_OR_GPU=gpu

ENV CONDA_VERSION=4.14.0-0 \
    CONDA_ENV=condaenv \
    RUNTIME_USER=appuser \
    RUNTIME_UID=1000 \
    SHELL=/bin/bash \
    LANG=C.UTF-8  \
    LC_ALL=C.UTF-8 \
    CONDA_DIR=/srv/conda \
    DEBIAN_FRONTEND=noninteractive

ENV NB_PYTHON_PREFIX=${CONDA_DIR}/envs/${CONDA_ENV} \
    DASK_ROOT_CONFIG=${CONDA_DIR}/etc \
    CPU_OR_GPU=${CPU_OR_GPU} \
    HOME=/home/${RUNTIME_USER} \
    PATH=${CONDA_DIR}/bin:${PATH} \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CONDA_DIR}/envs/${CONDA_ENV}/lib

# ======================== root ========================
# initialize paths we will use
RUN mkdir /data
RUN mkdir -p /code_execution

# Create appuser user, permissions, add conda init to startup script
RUN echo "Creating ${RUNTIME_USER} user..." \
    && groupadd --gid ${RUNTIME_UID} ${RUNTIME_USER}  \
    && useradd --create-home --gid ${RUNTIME_UID} --no-log-init --uid ${RUNTIME_UID} ${RUNTIME_USER} \
    && echo ". ${CONDA_DIR}/etc/profile.d/conda.sh ; conda activate ${CONDA_ENV}" > /etc/profile.d/init_conda.sh \
    && chown -R ${RUNTIME_USER}:${RUNTIME_USER} /srv /code_execution \
    && chmod 664 /data

# Install base packages
ARG DEBIAN_FRONTEND=noninteractive
COPY ./runtime/apt.txt /home/${RUNTIME_USER}
RUN rm -f /etc/apt/sources.list.d/cuda.list && rm -f /etc/apt/sources.list.d/nvidia-ml.list
RUN echo "Installing base packages..." \
    && apt-get update --fix-missing \
    && apt-get install -y apt-utils 2> /dev/null \
    && apt-get install -y wget zip tzdata \
    && apt-get install -y libgl1-mesa-glx libstdc++6 \
    && xargs -a /home/${RUNTIME_USER}/apt.txt apt-get install -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /home/${RUNTIME_USER}/apt.txt

# ======================== user ========================
USER ${RUNTIME_USER}

# Install conda
RUN echo "Installing Miniforge..." \
    && URL="https://github.com/conda-forge/miniforge/releases/download/${CONDA_VERSION}/Miniforge3-${CONDA_VERSION}-Linux-x86_64.sh" \
    && wget --quiet ${URL} -O /home/${RUNTIME_USER}/miniconda.sh \
    && /bin/bash /home/${RUNTIME_USER}/miniconda.sh -u -b -p ${CONDA_DIR} \
    && rm /home/${RUNTIME_USER}/miniconda.sh \
    && conda install -y -c conda-forge mamba \
    && mamba clean -afy \
    && find ${CONDA_DIR} -follow -type f -name '*.a' -delete \
    && find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete

# Switch back to root for installing conda packages
COPY ./runtime/environment-${CPU_OR_GPU}.yml /home/${RUNTIME_USER}/environment.yml
RUN mamba env create --name ${CONDA_ENV} -f /home/${RUNTIME_USER}/environment.yml  \
    && mamba clean -afy \
    && conda run pip cache purge \
    && rm /home/${RUNTIME_USER}/environment.yml \
    && find ${CONDA_DIR} -follow -type f -name '*.a' -delete \
    && find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete \
    && find ${CONDA_DIR} -follow -type f -name '*.js.map' -delete

# Copy run script into working dir and set it as the working doie
USER root
# Copy necessary files from VSC2022 repo
COPY vsc2022/vsc/*.py /opt/vsc/
COPY vsc2022/descriptor_eval.py /opt/descriptor_eval.py
COPY vsc2022/LICENSE /opt/LICENSE
COPY ./runtime/validation.py /opt/validation.py
WORKDIR /code_execution
RUN mkdir predictions submission
COPY --chown=appuser:appuser ./runtime/tests /code_execution/tests
COPY --chown=appuser:appuser ./runtime/entrypoint.sh /code_execution/entrypoint.sh
CMD ["/bin/bash", "/code_execution/entrypoint.sh"]