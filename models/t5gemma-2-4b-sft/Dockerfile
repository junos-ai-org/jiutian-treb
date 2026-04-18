FROM runpod/pytorch:1.0.3-cu1281-torch280-ubuntu2404

WORKDIR /workspace/sft

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py prepare_data.py ./
COPY configs ./configs

ENV HF_HOME=/workspace/hf-cache \
    TRANSFORMERS_CACHE=/workspace/hf-cache \
    HF_DATASETS_CACHE=/workspace/hf-cache

CMD ["python", "train.py", "--config", "configs/sft_flan.yaml"]
