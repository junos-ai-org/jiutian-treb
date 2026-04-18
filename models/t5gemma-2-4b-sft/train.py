"""QLoRA SFT for T5Gemma 2 4B-4B on a FLAN subset.

Assumes prepare_data.py has already written tokenized splits to
cfg['data']['tokenized_dir']. If not, falls back to loading + tokenizing
on the fly (slower — not recommended for real runs).

Usage:
    python train.py --config configs/sft_flan.yaml
    python train.py --config configs/sft_flan.yaml --resume
"""
import argparse
import os
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def build_model(cfg):
    m, q, l = cfg["model"], cfg["quantization"], cfg["lora"]

    bnb = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=DTYPES[q["bnb_4bit_compute_dtype"]],
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        m["name_or_path"],
        quantization_config=bnb,
        attn_implementation=m["attn_implementation"],
        dtype=DTYPES[m["dtype"]],
        device_map=m["device_map"],
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg["training"]["gradient_checkpointing"]
    )
    # Defensive: encoder-decoder LoRA needs embedding grads to flow through the
    # encoder. peft's kbit helper already calls this for the kbit+reentrant path,
    # but calling it explicitly is idempotent and covers non-reentrant edge cases.
    model.enable_input_require_grads()
    peft_cfg = LoraConfig(
        task_type=TaskType[l["task_type"]],
        r=l["r"],
        lora_alpha=l["alpha"],
        lora_dropout=l["dropout"],
        target_modules=l["target_modules"],
        bias=l["bias"],
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model


def load_data(cfg):
    tdir = cfg["data"]["tokenized_dir"]
    if Path(tdir).exists():
        return load_from_disk(tdir)
    raise FileNotFoundError(
        f"No tokenized dataset at {tdir}. Run prepare_data.py first."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    t, h = cfg["training"], cfg["hub"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name_or_path"])
    model = build_model(cfg)
    splits = load_data(cfg)

    training_args = Seq2SeqTrainingArguments(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        bf16=t["bf16"],
        optim=t["optim"],
        gradient_checkpointing=t["gradient_checkpointing"],
        logging_steps=t["logging_steps"],
        eval_strategy=t["eval_strategy"],
        eval_steps=t["eval_steps"],
        save_strategy=t["save_strategy"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        predict_with_generate=t["predict_with_generate"],
        generation_max_length=t["generation_max_length"],
        report_to=t["report_to"],
        run_name=t["run_name"],
        seed=t["seed"],
        push_to_hub=h["push_to_hub"],
        hub_model_id=h["hub_model_id"],
        hub_strategy=h["hub_strategy"],
        hub_private_repo=h["hub_private_repo"],
        hub_token=os.environ.get("HF_TOKEN"),
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
        processing_class=tokenizer,     # v5 renamed tokenizer= → processing_class=
        data_collator=collator,
    )

    trainer.train(resume_from_checkpoint=args.resume or None)
    trainer.save_model(t["output_dir"])
    if h["push_to_hub"]:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
