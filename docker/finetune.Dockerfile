# Kronos finetune/ (qlib pipeline) GPU training environment.
#
# Linux container so the upstream finetune scripts run unmodified:
# torchrun + nccl DDP works natively, pyqlib installs cleanly.
#
# Build (context = this docker/ dir, nothing is COPYed in):
#   docker build -f docker/finetune.Dockerfile -t kronos-finetune:latest docker
#
# Run (see finetune/run_finetune.sh docs):
#   docker run --gpus all --name kronos-finetune \
#     --network tradingagents-cn_tradingagents-network \
#     -v E:\AIProject\Kronos:/workspace \
#     --mount source=kronos-finetune-data,target=/workspace/data \
#     --mount source=kronos-finetune-outputs,target=/workspace/outputs \
#     -it kronos-finetune:latest
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1

# NOTE: pyqlib/comet_ml may upgrade numpy to 2.x; the resulting stack
# (torch 2.6.0 + numpy 2.4 + pandas 2.2.2 + pyqlib 0.9.7) has been validated
# end-to-end with a synthetic bin roundtrip test. Default PyPI is very slow
# from this network, hence the Tsinghua mirror.
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    pandas==2.2.2 \
    einops==0.8.1 \
    huggingface_hub==0.33.1 \
    safetensors==0.6.2 \
    matplotlib==3.9.3 \
    tqdm==4.67.1 \
    pymongo==4.10.1 \
    comet_ml \
    pyqlib

WORKDIR /workspace
CMD ["bash"]
