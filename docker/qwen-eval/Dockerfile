# Qwen eval image — vLLM-based for fast decoder-only generation on TReB.
# Separate image from t5gemma-eval because vLLM pins transformers<5 while
# T5Gemma 2 requires transformers>=5.0.
FROM runpod/pytorch:1.0.3-cu1281-torch280-ubuntu2404

WORKDIR /opt/eval

RUN apt-get update \
 && apt-get install -y --no-install-recommends git openssh-server \
 && rm -rf /var/lib/apt/lists/*

COPY docker/qwen-eval/requirements.txt /opt/eval/requirements.txt
RUN pip install --no-cache-dir -r /opt/eval/requirements.txt

COPY experiments/t5gemma_vs_qwen_treb/eval.py \
     experiments/t5gemma_vs_qwen_treb/score.py \
     experiments/t5gemma_vs_qwen_treb/run.sh \
     experiments/t5gemma_vs_qwen_treb/run_all.sh \
     /opt/eval/
COPY experiments/t5gemma_vs_qwen_treb/configs /opt/eval/configs
RUN chmod +x /opt/eval/run.sh /opt/eval/run_all.sh

ENV HF_HOME=/workspace/hf-cache \
    TRANSFORMERS_CACHE=/workspace/hf-cache \
    HF_DATASETS_CACHE=/workspace/hf-cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1 \
    VARIANTS=qwen

CMD ["bash", "/opt/eval/run.sh"]
