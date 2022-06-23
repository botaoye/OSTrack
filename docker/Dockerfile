FROM mcr.microsoft.com/azureml/o16n-base/python-assets@sha256:20a8b655a3e5b9b0db8ddf70d03d048a7cf49e569c4f0382198b1ee77631a6ad AS inferencing-assets

FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu18.04
USER root:root

ENV com.nvidia.cuda.version $CUDA_VERSION
ENV com.nvidia.volumes.needed nvidia_driver
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
ENV NCCL_DEBUG=INFO
ENV HOROVOD_GPU_ALLREDUCE=NCCL

# Inference
COPY --from=inferencing-assets /artifacts /var/

# Install Common Dependencies
RUN apt-get update && \
    apt-get install -y apt-transport-https && \
    apt-get install -y --no-install-recommends \
    # SSH and RDMA
    libmlx4-1 \
    libmlx5-1 \
    librdmacm1 \
    libibverbs1 \
    libmthca1 \
    libdapl2 \
    dapl2-utils \
    openssh-client \
    openssh-server \
    sudo \
    iproute2 && \
    # Others
    apt-get install -y --no-install-recommends \
    --allow-change-held-packages \
    build-essential \
    bzip2=1.0.6-8.1ubuntu0.2 \
    libbz2-1.0=1.0.6-8.1ubuntu0.2 \
    systemd \
    git \
    wget \
    vim \
    tmux \
    unzip \
    ca-certificates \
    libjpeg-dev \
    cpio \
    jq \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libturbojpeg \
    fuse && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*


# Set timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# create dir for sshd
RUN mkdir -p /var/run/sshd /root/.ssh

# Inference
# Copy logging utilities, nginx and rsyslog configuration files, IOT server binary, etc.
COPY --from=inferencing-assets /artifacts /var/
RUN /var/requirements/install_system_requirements.sh && \
    cp /var/configuration/rsyslog.conf /etc/rsyslog.conf && \
    cp /var/configuration/nginx.conf /etc/nginx/sites-available/app && \
    ln -s /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \
    rm -f /etc/nginx/sites-enabled/default
ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=300
EXPOSE 5001 8883 8888

# Conda Environment
ENV MINICONDA_VERSION latest
ENV PATH /opt/miniconda/bin:$PATH
RUN wget -qO /tmp/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bf -p /opt/miniconda && \
    conda clean -ay && \
    rm -rf /opt/miniconda/pkgs && \
    rm /tmp/miniconda.sh && \
    find / -type d -name __pycache__ | xargs rm -rf
# Open-MPI installation
ENV OPENMPI_VERSION 3.1.2
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-${OPENMPI_VERSION}.tar.gz && \
    tar zxf openmpi-${OPENMPI_VERSION}.tar.gz && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

RUN conda install -c r -y conda python=3.8 pip=20.1.1
RUN conda install -y numpy pyyaml ipython mkl scikit-learn matplotlib pandas setuptools Cython h5py graphviz libgcc mkl-include cmake cffi typing cython && \
    conda install -y -c mingfeima mkldnn && \
    conda install -c anaconda gxx_linux-64
RUN conda clean -ya
#RUN pip install boto3 addict tqdm regex pyyaml opencv-python opencv-contrib-python azureml-defaults nltk spacy future tensorboard wandb filelock tokenizers sentencepiece
RUN pip install regex pyyaml opencv-contrib-python azureml-defaults future

# Set CUDA_ROOT
RUN export CUDA_HOME="/usr/local/cuda"

# Install littleboy related packages

# Install pytorch
RUN conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# Install other packages
RUN pip install PyYAML && \
    pip install easydict && \
    pip install cython && \
    pip install opencv-python && \
    pip install pandas && \
    pip install tqdm && \
    pip install pycocotools && \
    # apt-get install libturbojpeg && \
    pip install jpeg4py && \
    pip install tb-nightly && \
    pip install tikzplotlib && \
    pip install thop && \
    pip install lmdb && \
    pip install scipy && \
    pip install visdom && \
    pip install wandb && \
    pip install tensorboardX && \
    pip install yacs && \
    pip install timm && \
    pip install git+https://github.com.cnpmjs.org/votchallenge/vot-toolkit-python
