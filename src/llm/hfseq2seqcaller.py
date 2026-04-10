"""HuggingFace Seq2Seq caller for encoder-decoder models (e.g. T5Gemma).

Drop-in replacement for VLLMCaller when the model cannot run on vLLM.
Matches the call_batch / call interface that TReB's reasoners and judges expect.
"""

import torch
from transformers import AutoProcessor, AutoModelForSeq2SeqLM


class HFSeq2SeqCallerConfig:
    def __init__(self, model_name, llmpath, max_model_len, temperature,
                 max_new_tokens=512, batch_size=4):
        self.model_name = model_name
        self.llmpath = llmpath
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size  # GPU sub-batch size to avoid OOM


class HFSeq2SeqCaller:
    def __init__(self, config: HFSeq2SeqCallerConfig):
        self.config = config
        self.max_input_length = config.max_model_len - 128  # match VLLMCaller reservation

        print(f"Loading processor from {config.llmpath}...")
        self.processor = AutoProcessor.from_pretrained(config.llmpath)

        attn_impl = "eager"
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
            print("  flash-attn not installed, using SDPA attention.")

        print(f"Loading model from {config.llmpath} (bfloat16, attn={attn_impl})...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            config.llmpath,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=attn_impl,
        )
        self.model.eval()

        # Resolve the device for input tensors. With device_map="auto" the model
        # may span multiple GPUs; we send inputs to the encoder's first parameter.
        self._input_device = next(self.model.parameters()).device

        # torch.compile can conflict with multi-device splits. Only apply when
        # the entire model lives on a single device.
        devices = {p.device for p in self.model.parameters()}
        if len(devices) == 1:
            print("  Applying torch.compile...")
            self.model = torch.compile(self.model)
        else:
            print(f"  Skipping torch.compile (model spans {len(devices)} devices).")

        print(f"  Ready. max_input={self.max_input_length}, "
              f"max_new_tokens={config.max_new_tokens}")

    def _messages_to_prompt(self, messages: list[dict]) -> str:
        """Convert chat messages to a single prompt string via chat template.

        TReB sends [{"role": "system", ...}, {"role": "user", ...}].
        T5Gemma may not support system role — fall back to single user message.
        """
        add_generation_prompt = messages[-1]["role"] != "assistant"

        try:
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            # Collapse system + user into single user message
            combined = "\n\n".join(m["content"] for m in messages if m["role"] != "assistant")
            fallback_messages = [{"role": "user", "content": combined}]
            try:
                prompt = self.processor.apply_chat_template(
                    fallback_messages, tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            except Exception:
                prompt = combined

        # Strip trailing EOS token (matches VLLMCaller behavior).
        # AutoProcessor stores the tokenizer as .tokenizer; eos_token lives there.
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        eos = getattr(tokenizer, "eos_token", None)
        if eos and prompt.endswith(eos):
            prompt = prompt[: -len(eos)]
        if eos and prompt.endswith(eos + "\n"):
            prompt = prompt[: -len(eos + "\n")]

        return prompt

    def _generate_sub_batch(self, prompts: list[str]) -> list[str]:
        """Run model.generate on a single GPU-sized sub-batch."""
        inputs = self.processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
        )
        inputs = {k: v.to(self._input_device) for k, v in inputs.items()}

        gen_kwargs = {"max_new_tokens": self.config.max_new_tokens}
        if self.config.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.config.temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        responses = self.processor.batch_decode(output_ids, skip_special_tokens=True)
        return [r.strip() for r in responses]

    def call_batch(self, samples: list[list[dict]]) -> list[str]:
        """Match VLLMCaller.call_batch interface.

        Args:
            samples: List of conversations, each a list of message dicts
                     with 'role' and 'content' keys.
        Returns:
            List of generated text strings, one per input conversation.
        """
        prompts = [self._messages_to_prompt(conv) for conv in samples]

        results = []
        bs = self.config.batch_size
        for i in range(0, len(prompts), bs):
            sub_batch = prompts[i: i + bs]
            results.extend(self._generate_sub_batch(sub_batch))

        return results

    def call(self, messages: list[dict]) -> str:
        """Single-conversation convenience method."""
        return self.call_batch([messages])[0]
