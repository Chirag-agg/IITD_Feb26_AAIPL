# Starting with Qwen3-4B
import time
import os
import torch
from pathlib import Path
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer

_MODEL_ID = "Qwen/Qwen3-4B"
_CACHE_DIR = Path("hf_models").as_posix()

# Keep cache in workspace-friendly location.
os.environ.setdefault("HF_HOME", _CACHE_DIR)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", _CACHE_DIR)

class QAgent(object):
    def __init__(self, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(
            _MODEL_ID,
            cache_dir=_CACHE_DIR,
            local_files_only=True,
            padding_side="left",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID,
            cache_dir=_CACHE_DIR,
            local_files_only=True,
            torch_dtype="auto",
            device_map="auto",
        )

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
            start_time = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 320),
                do_sample=kwargs.get("do_sample", False),
                temperature=kwargs.get("temperature", 1.0),
                top_p=kwargs.get("top_p", 1.0),
                repetition_penalty=kwargs.get("repetition_penalty", 1.1),
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
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

            # remove thinking content using regex
            # result = re.sub(r'<think>[\s\S]*?</think>', '', full_result, flags=re.DOTALL).strip()
            index = (
                len(output_ids) - output_ids[::-1].index(151668)
                if 151668 in output_ids
                else 0
            )

            # decode the full result
            content = self.tokenizer.decode(
                output_ids[index:], skip_special_tokens=True
            ).strip("\n")
            batch_outs.append(content)
        if tgps_show_var:
            return (
                batch_outs[0] if len(batch_outs) == 1 else batch_outs,
                token_len,
                generation_time,
            )
        return batch_outs[0] if len(batch_outs) == 1 else batch_outs, None, None


if __name__ == "__main__":
    # Single example generation
    model = QAgent()
    prompt = f"""
    Question: Generate a hard MCQ based question as well as their 4 choices and its answers on the topic, Number Series.
    Return your response as a valid JSON object with this exact structure:

        {{
            "topic": Your Topic,
            "question": "Your question here ending with a question mark?",
            "choices": [
                "A) First option",
                "B) Second option", 
                "C) Third option",
                "D) Fourth option"
            ],
            "answer": "A",
            "explanation": "Brief explanation of why the correct answer is right and why distractors are wrong"
        }}
    """

    response, tl, tm = model.generate_response(
        prompt,
        tgps_show=True,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
    )
    print("Single example response:")
    print("Response: ", response)
    safe_tl = tl if isinstance(tl, (int, float)) else 0
    safe_tm = tm if isinstance(tm, (int, float)) else 0
    print(
        f"Total tokens: {safe_tl}, Time taken: {safe_tm:.2f} seconds, TGPS: {(safe_tl/safe_tm if safe_tm > 0 else 0):.2f} tokens/sec"
    )
    print("+-------------------------------------------------\n\n")

    # Multi example generation
    prompts = [
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "What are the main differences between Python and Java?",
        "What is the significance of the Turing Test in AI?",
        "What is the capital of Japan?",
    ]
    responses, tl, tm = model.generate_response(
        prompts,
        tgps_show=True,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
    )
    print("\nMulti example responses:")
    for i, resp in enumerate(responses):
        print(f"Response {i+1}: {resp}")
    safe_tl = tl if isinstance(tl, (int, float)) else 0
    safe_tm = tm if isinstance(tm, (int, float)) else 0
    print(
        f"Total tokens: {safe_tl}, Time taken: {safe_tm:.2f} seconds, TGPS: {(safe_tl/safe_tm if safe_tm > 0 else 0):.2f} tokens/sec"
    )
