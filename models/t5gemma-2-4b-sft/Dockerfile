FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu

WORKDIR /workspace/sft

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py prepare_data.py ./
COPY configs ./configs

ENV HF_HOME=/workspace/hf-cache \
    TRANSFORMERS_CACHE=/workspace/hf-cache \
    HF_DATASETS_CACHE=/workspace/hf-cache

CMD ["python", "train.py", "--config", "configs/sft_flan.yaml"]
