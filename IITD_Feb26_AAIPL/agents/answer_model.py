# Qwen3-4B in action.
import os
from pathlib import Path
# Force HF cache to writable location to avoid read-only errors causing timeouts
_CACHE_DIR = Path("hf_models").as_posix()
os.environ.setdefault("HF_HOME", _CACHE_DIR)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", _CACHE_DIR)

import random
import numpy as np
import torch
import time
import re
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Set deterministic seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# HF model ID and local cache directory
_MODEL_ID = "Qwen/Qwen3-4B"
_ADAPTER_DIR = Path("hf_models/a_agent_lora")


class AAgent(object):
    def __init__(self, **kwargs):
        # load the tokenizer and the model from local HF cache
        self.tokenizer = AutoTokenizer.from_pretrained(
            _MODEL_ID, cache_dir=_CACHE_DIR, local_files_only=True, padding_side="left"
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID, cache_dir=_CACHE_DIR, local_files_only=True,
            torch_dtype="auto", device_map="auto"
        )
        # LoRA adapter: load if available, otherwise use base model
        if _ADAPTER_DIR.exists():
            print(f"🔗 Loading LoRA adapter from {_ADAPTER_DIR}")
            self.model = PeftModel.from_pretrained(base_model, str(_ADAPTER_DIR))
        else:
            print(f"⚠️  No LoRA adapter at {_ADAPTER_DIR} — using base model")
            self.model = base_model

    def generate_response(
        self, message: str | List[str], system_prompt: Optional[str] = None, **kwargs
    ) -> tuple[str | List[str], int | None, float | None]:
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
        if isinstance(message, str):
            message = [message]
        # Prepare all messages for batch processing
        all_messages = []
        for msg in message:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg},
            ]
            all_messages.append(messages)

        # convert all messages to text format
        texts = []
        for messages in all_messages:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            texts.append(text)

        # tokenize all texts together with padding
        model_inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)

        tgps_show_var = kwargs.get("tgps_show", False)
        # conduct batch text completion
        if tgps_show_var:
            print(
                "DEBUG: Generation Config: "
                f"max_new_tokens={kwargs.get('max_new_tokens', 220)}, "
                f"do_sample={kwargs.get('do_sample', False)}, "
                f"temperature={kwargs.get('temperature', 1.0)}, "
                f"top_p={kwargs.get('top_p', 1.0)}, "
                f"repetition_penalty={kwargs.get('repetition_penalty', 1.1)}"
            )
            start_time = time.time()
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 220),
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                temperature=kwargs.get("temperature", 1.0),
                top_p=kwargs.get("top_p", 1.0),
                repetition_penalty=kwargs.get("repetition_penalty", 1.1),
                do_sample=kwargs.get("do_sample", False),
            )
        if tgps_show_var:
            generation_time = time.time() - start_time

        # decode the batch
        batch_outs = []
        if tgps_show_var:
            token_len = 0
        for i, (input_ids, generated_sequence) in enumerate(
            zip(model_inputs.input_ids, generated_ids)
        ):
            # extract only the newly generated tokens
            output_ids = generated_sequence[len(input_ids) :].tolist()

            # compute total tokens generated
            if tgps_show_var:
                token_len += len(output_ids)

            # Decode raw tokens first (skip_special_tokens=True strips thinking CONTENT in Qwen3)
            raw = self.tokenizer.decode(output_ids, skip_special_tokens=False)
            # Strip thinking blocks at string level, preserving the answer after them
            raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
            # Strip remaining special token markers (im_start, im_end, etc.)
            raw = re.sub(r'<\|[^|]*\|>', '', raw)
            content = raw.strip()
            batch_outs.append(content)
        if tgps_show_var:
            return (
                batch_outs[0] if len(batch_outs) == 1 else batch_outs,
                token_len,
                generation_time,
            )
        return batch_outs[0] if len(batch_outs) == 1 else batch_outs, None, None


if __name__ == "__main__":
    # Single message (backward compatible)
    ans_agent = AAgent()
    response, tl, gt = ans_agent.generate_response(
        "Solve: 2x + 5 = 15",
        system_prompt="You are a math tutor.",
        tgps_show=True,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
    )
    print(f"Single response: {response}")
    safe_tl = tl if isinstance(tl, (int, float)) else 0
    safe_gt = gt if isinstance(gt, (int, float)) else 0
    print(
        f"Token length: {safe_tl}, Generation time: {safe_gt:.2f} seconds, Tokens per second: {(safe_tl/safe_gt if safe_gt > 0 else 0):.2f}"
    )
    print("-----------------------------------------------------------")

    # Batch processing (new capability)
    messages = [
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "What are the main differences between Python and Java?",
        "What is the significance of the Turing Test in AI?",
        "What is the capital of Japan?",
    ]
    responses, tl, gt = ans_agent.generate_response(
        messages,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
        tgps_show=True,
    )
    print("Responses:")
    for i, resp in enumerate(responses):
        print(f"Message {i+1}: {resp}")
    safe_tl = tl if isinstance(tl, (int, float)) else 0
    safe_gt = gt if isinstance(gt, (int, float)) else 0
    print(
        f"Token length: {safe_tl}, Generation time: {safe_gt:.2f} seconds, Tokens per second: {(safe_tl/safe_gt if safe_gt > 0 else 0):.2f}"
    )
    print("-----------------------------------------------------------")

    # Custom parameters
    response = ans_agent.generate_response(
        "Write a story", temperature=0.8, max_new_tokens=512
    )
    print(f"Custom response: {response}")
