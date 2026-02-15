#!/usr/bin/python3

import re
import json

from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Any

from .answer_model import AAgent
# from .answer_model_llama import AAgent

class AnsweringAgent(object):
    r"""Agent responsible for answering MCQ questions with confidence scoring"""

    def __init__(self, select_prompt1: bool = True, **kwargs):
        self.agent = AAgent(**kwargs)
        self.select_prompt1 = select_prompt1

    def build_prompt(self, question_data: Dict[str, str | Any]) -> Tuple[str, str]:
        """Generate an answer to the given MCQ question with confidence and reasoning"""

        # Phase 5: Win Optimization Prompts (Deterministic & Targeted)
        sys_prompt1 = (
            "You are a competitive exam logical reasoning expert.\n\n"
            "For each question, follow this STRICT process:\n"
            "1. IDENTIFY TYPE: Syllogism, Seating Arrangement, Series, Blood Relation, or Logical Deduction.\n"
            "2. APPLY PROTOCOL: Use the specific method below.\n"
            "3. ELIMINATE: Disprove options explicitly.\n"
            "4. VERIFY: Check for contradictions.\n"
            "5. OUTPUT: Valid JSON only.\n\n"
            "TOPIC PROTOCOLS:\n"
            "► SEATING ARRANGEMENTS (CRITICAL):\n"
            "   - Construct a POSITIONAL TABLE (1, 2, ..., N).\n"
            "   - Place definitely known entities first.\n"
            "   - Use negative constraints to eliminate positions.\n"
            "   - If circular, map relative positions (Left/Right) carefully.\n"
            "   - CHECK: Does the final arrangement satisfy ALL clues?\n\n"
            "► SYLLOGISMS:\n"
            "   - Use VENN LOGIC.\n"
            "   - 'All A are B' -> A inside B.\n"
            "   - 'Some A are B' -> A intersects B.\n"
            "   - REJECT any conclusion not true in ALL possible diagrams.\n\n"
            "► SERIES:\n"
            "   - Calculate differences (D1, D2).\n"
            "   - Check geometric ratios or prime/square patterns.\n\n"
            "FORMAT CONSISTENCY:\n"
            "Reasoning must be concise (< 120 words).\n"
            "Output JSON MUST be:\n"
            "{\n"
            "  \"answer\": \"A\",\n"
            "  \"reasoning\": \"Type: Seating. 1. A is left of B... 2. C is opposite... Arrangement: [A, C, B...]. Option A is correct because...\"\n"
            "}"
        )
        sys_prompt2 = sys_prompt1

        tmpl = (
            "Question:\n{}\n\n"
            "Options:\n{}\n\n"
            "Solve carefully.\n"
            "Returns ONLY the required JSON."
        )

        prompt = tmpl.format(
            question_data["question"], self._format_choices(question_data["choices"])
        )

        return prompt, sys_prompt1 if self.select_prompt1 else sys_prompt2

    def answer_question(
        self, question_data: Dict | List[Dict], **kwargs
    ) -> Tuple[List[Dict], int | None, float | None]:
        """Generate answer(s) for the given question(s)"""
        if isinstance(question_data, list):
            prompt = []
            for qd in question_data:
                p, sp = self.build_prompt(qd)
                prompt.append(p)
        else:
            prompt, sp = self.build_prompt(question_data)

        resp, tl, gt = self.agent.generate_response(prompt, sp, **kwargs)

        if (
            isinstance(resp, list) and all(isinstance(r, str) for r in resp)
        ) or isinstance(resp, str):
            return resp, tl, gt
        else:
            return ("" if not isinstance(resp, list) else [""] * len(resp), tl, gt)

    def answer_batches(
        self, questions: List[Dict], batch_size: int = 5, **kwargs
    ) -> Tuple[List[Dict], List[int | None], List[float | None]]:
        """Answer questions in batches"""
        answers = []
        tls, gts = [], []
        total_batches = (len(questions) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc="STEPS: ", unit="batch")
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i : i + batch_size]
            batch_answers, tl, gt = self.answer_question(batch_questions, **kwargs)
            if kwargs.get("verbose", False):
                print(f"DEBUG: Generated {tl} tokens (limit: 100)")
            answers.extend(batch_answers)
            tls.append(tl)
            gts.append(gt)
            pbar.update(1)

        pbar.close()
        return answers, tls, gts

    def count_tokens_a(self, text: str) -> int:
        """Count the number of tokens in the text using the agent's tokenizer (matches official oracle)"""
        if not hasattr(self.agent, "tokenizer"):
            raise AttributeError("The agent does not have a tokenizer attribute.")
        return len(self.agent.tokenizer.encode(text, add_special_tokens=False))

    def filter_answers(self, ans: List[str | Dict[str, str]]) -> List[Dict[str, str]]:
        r"""Official oracle filter_answers — matches competition judging exactly.
        Returns list aligned with input (None for failed entries)."""

        def basic_checks(a1: Dict[str, str]) -> bool:
            # Official: only 'answer' key is required
            required_keys = ['answer']
            if not all((key in a1) and isinstance(a1[key], str) for key in required_keys):
                return False
            # Official: if single char and NOT in ABCDabcd → fail
            if len(a1['answer']) == 1 and (a1['answer'] not in 'ABCDabcd'):
                return False
            # Token budget: answer tokens < 50
            check_len = self.count_tokens_a(a1['answer'])
            if check_len < 50:
                # answer + reasoning tokens < 512
                check_len += self.count_tokens_a(a1.get('reasoning', 'None'))
                if check_len < 512:
                    return True
            return False

        filtered_answers = []
        for i, a in enumerate(ans):
            if isinstance(a, dict):
                if basic_checks(a):
                    filtered_answers.append(a)
                else:
                    filtered_answers.append(None)
            elif isinstance(a, str):
                # Basic checks: at least with correct JSON format
                try:
                    a1 = json.loads(a)
                    if basic_checks(a1):
                        filtered_answers.append(a1)
                    else:
                        filtered_answers.append(None)
                except json.JSONDecodeError:
                    # Try extracting JSON from markdown fences
                    cleaned = re.sub(r"```json|```", "", a, flags=re.IGNORECASE).strip()
                    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
                    if match:
                        try:
                            a1 = json.loads(match.group(0))
                            if basic_checks(a1):
                                filtered_answers.append(a1)
                            else:
                                filtered_answers.append(None)
                        except json.JSONDecodeError:
                            print(f"Skipping invalid JSON at index {i}: {a}")
                            filtered_answers.append(None)
                    else:
                        print(f"Skipping invalid JSON at index {i}: {a}")
                        filtered_answers.append(None)
                    continue
            else:
                # If the answer is neither a dict nor a str, skip it
                print(f"Skipping unsupported type at index {i}: {type(a)}")
                filtered_answers.append(None)
        return filtered_answers

    def save_answers(self, answers: List[Any], file_path: str | Path) -> None:
        """Save generated answers to a JSON file"""
        # check for existence of dir
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump([a for a in answers], f, indent=4)

    def _format_choices(self, choices) -> str:
        r"""Format the choices for better readability. Handles both list and dict formats."""
        # Handle dict format: {"A": "text", "B": "text", ...}
        if isinstance(choices, dict):
            formatted = []
            for letter in ["A", "B", "C", "D"]:
                if letter in choices:
                    formatted.append(f"{letter}) {choices[letter].strip()}")
            return " ".join(formatted)
        # Handle list format: ["A) text", "B) text", ...] or ["text1", "text2", ...]
        formatted = []
        for choice in choices:
            if not re.match(r"^[A-D]\)", choice.strip()):
                letter = chr(65 + len(formatted))  # A, B, C, D
                formatted.append(f"{letter}) {choice.strip()}")
            else:
                formatted.append(choice.strip())
        return " ".join(formatted)


# Example usage
if __name__ == "__main__":
    import json
    import yaml
    import argparse
    from utils.build_prompt import auto_json, option_extractor_prompt

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # python -m agents.answer_agent --input_file outputs/filtered_questions.json --output_file outputs/answers.json --batch_size 5 --verbose
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    argparser = argparse.ArgumentParser(description="Run the Answering Agent")
    argparser.add_argument(
        "--input_file",
        type=str,
        default="outputs/filtered_questions.json",
        help="Path to the input JSON file with questions",
    )
    argparser.add_argument(
        "--output_file",
        type=str,
        default="outputs/answers.json",
        help="Path to save the answers",
    )
    argparser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for processing questions"
    )
    argparser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )
    args = argparser.parse_args()

    SELECT_PROMPT1 = False  # Use the first system prompt for answering

    # Load sample questions (assuming they're saved from QuestioningAgent)
    with open(args.input_file, "r") as f:
        sample_questions = json.load(f)

    agent = AnsweringAgent(select_prompt1=SELECT_PROMPT1)

    # gen_kwargs = {"tgps_show": True, "max_new_tokens": 512, "temperature": 0.1, "top_p": 0.9, "do_sample": True}
    gen_kwargs = {"tgps_show": True}
    with open("agen.yaml", "r") as f:
        gen_kwargs.update(yaml.safe_load(f))
    answer, tls, gts = agent.answer_batches(
        questions=sample_questions, batch_size=args.batch_size, **gen_kwargs
    )
    ans = []
    for idx, (q, a) in enumerate(zip(sample_questions, answer)):
        if args.verbose:
            print(f"\n=== Question {idx+1} ===")
            print(f"Question: {q.get('question', 'N/A')}")
            print(f"Expected: {q.get('expected_answer', q.get('answer', 'N/A'))}")
            print(f"Model Answer:\n{a}")
        # Robust JSON extraction: 3-tier fallback
        parsed = None
        # Tier 1: Direct JSON parse
        try:
            parsed = json.loads(a)
        except Exception:
            pass
        # Tier 2: Strip markdown fences, extract first {...} block
        if parsed is None:
            cleaned = re.sub(r"```json|```", "", a)
            match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except Exception:
                    pass
        # Tier 3: Robust Regex Extraction (User Mandated)
        if parsed is None:
            match = re.search(r'"answer"\s*:\s*"([ABCD])"', a)
            if match:
                parsed = {"answer": match.group(1)}
                # Fallback to single letter search as last resort
                letter = re.search(r'\b([A-D])\b', a)
                if letter:
                    parsed = {"answer": letter.group(1), "reasoning": "Fallback extraction."}
        # Final fallback
        if parsed is None:
            if args.verbose:
                print(f"Warning: Failed to extract JSON for Q{idx+1}")
            parsed = {"answer": "X", "reasoning": "Parse error"}
        # Normalize answer to single letter
        if isinstance(parsed, dict) and "answer" in parsed:
            answer_str = str(parsed["answer"]).strip().upper()
            for char in answer_str:
                if char in "ABCD":
                    parsed["answer"] = char
                    break
        ans.append(parsed)

    if args.verbose:
        if gen_kwargs.get("tgps_show", False):
            for idx, (tl, gt) in enumerate(zip(tls, gts)):
                print(f"BATCH - {idx}")
                print(f"Tokens: {tl}, Time: {gt:.3f} seconds")
                print(f"TGPS: {tl/gt:.3f} seconds")
            print("\n" + "=" * 50)
            print(
                f"Total Time: {sum(gts):.3f} seconds; Total Tokens: {sum(tls)}; TGPS: {sum(tls)/sum(gts):.3f} seconds"
            )

    # Save answers
    agent.save_answers(ans, args.output_file)
    filtered_file_name = args.output_file.replace(
        "answers.json", "filtered_answers.json"
    )
    agent.save_answers(agent.filter_answers(ans), filtered_file_name)
