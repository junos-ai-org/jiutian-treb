# T5Gemma 2 4B-4B — FLAN SFT

QLoRA SFT for T5Gemma 2 4B-4B on a 100K-sample FLAN subset. Output is merged
and pushed to `DiffusionTableQA/t5gemma-2-4b-flan-sft(-merged)` for use in the
encoder-vs-decoder TReB experiment.

See `docs/progress.md` and `~/.claude/knowledgebase/treb/t5gemma-sft.kb` for
context and hyperparameter rationale.

## Flow

1. **Build + push image** (AWS build server)
   ```
   # on the build server
   git clone git@github.com:junos-ai-org/jiutian-treb.git
   cd jiutian-treb/models/t5gemma-2-4b-sft
   docker build -t achithanar/t5gemma-sft:latest .
   docker push achithanar/t5gemma-sft:latest
   ```
   Re-pull + rebuild + push whenever `train.py`, `prepare_data.py`,
   `requirements.txt`, or the Dockerfile changes. Config YAML changes
   alone don't require a rebuild if you bind-mount `configs/` on the pod;
   otherwise rebuild to bake the new config.
2. **RunPod: A100 80GB + network volume mounted at `/workspace`.**
   Set pod env vars: `HF_TOKEN`, `WANDB_API_KEY`.
3. **Pre-tokenize FLAN (once, lands on volume):**
   ```
   python prepare_data.py --config configs/sft_flan.yaml
   ```
4. **Train:**
   ```
   python train.py --config configs/sft_flan.yaml
   # resume after spot termination:
   python train.py --config configs/sft_flan.yaml --resume
   ```
   Checkpoints stream to `DiffusionTableQA/t5gemma-2-4b-flan-sft` on HF Hub
   (private, `hub_strategy=checkpoint`).
5. **Merge + push merged model:**
   ```
   python merge_adapter.py \
     --base google/t5gemma-2-4b-4b-ul2 \
     --adapter /workspace/checkpoints/t5gemma-2-4b-flan-sft \
     --out    /workspace/merged/t5gemma-2-4b-flan-sft-merged \
     --push-to DiffusionTableQA/t5gemma-2-4b-flan-sft-merged
   ```

## Files

- `configs/sft_flan.yaml` — all hyperparameters
- `prepare_data.py` — FLAN subset → tokenized splits on disk
- `train.py` — Seq2SeqTrainer + QLoRA loop (push_to_hub enabled)
- `merge_adapter.py` — LoRA merge + push merged model
- `Dockerfile` — slim training image (code + deps only; no data baked in)
- `requirements.txt`, `.env.example`

## Notes

- Data and base weights are downloaded at runtime to `/workspace/hf-cache`
  (the RunPod network volume) — the image stays small.
- `attn_implementation=eager` (flash-attn has known bugs on T5Gemma).
- `task_type=SEQ_2_SEQ_LM` in LoRA (not CAUSAL_LM).
- TRL's `SFTTrainer` does NOT support encoder-decoder — use `Seq2SeqTrainer`.
