# T5Gemma eval image — for T5Gemma 2 base + FLAN-SFT adapter on TReB.
# HF transformers + PEFT stack. Mirrors the SFT image's base + cache layout.
FROM runpod/pytorch:1.0.3-cu1281-torch280-ubuntu2404

# Avoid shadowing the /workspace volume with image code (bit us during SFT).
WORKDIR /opt/eval

# Build-time deps: git (for hf_hub + CODE_REPO), openssh-server (sshd in run.sh).
RUN apt-get update \
 && apt-get install -y --no-install-recommends git openssh-server \
 && rm -rf /var/lib/apt/lists/*

# Build context: repo root. Paths are resolved from there.
COPY docker/t5gemma-eval/requirements.txt /opt/eval/requirements.txt
RUN pip install --no-cache-dir -r /opt/eval/requirements.txt

# Eval code + configs live under the experiment directory.
COPY experiments/t5gemma_vs_qwen_treb/eval.py \
     experiments/t5gemma_vs_qwen_treb/score.py \
     experiments/t5gemma_vs_qwen_treb/run.sh \
     experiments/t5gemma_vs_qwen_treb/run_all.sh \
     /opt/eval/
COPY experiments/t5gemma_vs_qwen_treb/configs /opt/eval/configs
RUN chmod +x /opt/eval/run.sh /opt/eval/run_all.sh

# Put HF caches on the network volume by default so restarts are cheap.
ENV HF_HOME=/workspace/hf-cache \
    TRANSFORMERS_CACHE=/workspace/hf-cache \
    HF_DATASETS_CACHE=/workspace/hf-cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1 \
    VARIANTS="t5gemma_base t5gemma_sft"

CMD ["bash", "/opt/eval/run.sh"]
