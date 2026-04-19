FROM runpod/pytorch:1.0.3-cu1281-torch280-ubuntu2404

# Code lives at /opt/sft (NOT /workspace) because the RunPod network volume
# mounts at /workspace and would shadow any code placed there.
WORKDIR /opt/sft

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py prepare_data.py merge_adapter.py run.sh ./
COPY configs ./configs
RUN chmod +x run.sh

ENV HF_HOME=/workspace/hf-cache \
    TRANSFORMERS_CACHE=/workspace/hf-cache \
    HF_DATASETS_CACHE=/workspace/hf-cache \
    CONFIG=configs/sft_flan.yaml

CMD ["bash", "/opt/sft/run.sh"]
