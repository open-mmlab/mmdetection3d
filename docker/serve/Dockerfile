ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ARG MMCV="1.3.8"
ARG MMSEGMENTATION="0.14.1"
ARG MMDET="2.14.0"
ARG MMDET3D="0.17.1"

ENV PYTHONUNBUFFERED TRUE

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    ca-certificates \
    g++ \
    openjdk-11-jre-headless \
    # MMDet3D Requirements
    ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/conda/bin:$PATH"
RUN export FORCE_CUDA=1

# TORCHSEVER
RUN pip install torchserve torch-model-archiver

# MMLAB
ARG PYTORCH
ARG CUDA
RUN ["/bin/bash", "-c", "pip install mmcv-full==${MMCV} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${PYTORCH}/index.html"]
RUN pip install mmdet==${MMDET}
RUN pip install mmsegmentation==${MMSEGMENTATION}
RUN pip install mmdet3d==${MMDET3D}


RUN useradd -m model-server \
    && mkdir -p /home/model-server/tmp

COPY entrypoint.sh /usr/local/bin/entrypoint.sh

RUN chmod +x /usr/local/bin/entrypoint.sh \
    && chown -R model-server /home/model-server

COPY config.properties /home/model-server/config.properties
RUN mkdir /home/model-server/model-store && chown -R model-server /home/model-server/model-store

EXPOSE 8080 8081 8082

USER model-server
WORKDIR /home/model-server
ENV TEMP=/home/model-server/tmp
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["serve"]
