#!/usr/bin/python3

from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Any

from .question_model import QAgent
# from .question_model_llama import QAgent

import random
import json

class QuestioningAgent(object):
    r"""Agent responsible for generating questions"""

    def __init__(self, **kwargs):
        self.agent = QAgent(**kwargs)

    def build_inc_samples(self, inc_samples: List[Dict[str, str]], topic: str) -> str:
        r"""
        Build a string of example questions from the provided samples.
        """
        if not inc_samples:
            return ""
        fmt = (
            "EXAMPLE: {}\n"
            "{{\n"
            '  "topic": "{}",\n'
            '  "question": "{}",\n'
            '  "choices": ["A) {}", "B) {}", "C) {}", "D) {}"],\n'
            '  "answer": "{}",\n'
            '  "explanation": "{}"\n'
            "}}"
        )

        sample_str = ""
        for sample in inc_samples:
            explanation = sample.get("explanation", "")
            question = sample.get("question", "")
            answer = sample.get("answer", "")

            # Normalize choices
            raw_choices = sample.get("choices", [""] * 4)
            norm_choices = []
            if isinstance(raw_choices, dict):
                norm_choices = [raw_choices.get(k, "") for k in ["A", "B", "C", "D"]]
            else:
                # if list, strip prefixes "A) " etc if present
                for i, c in enumerate(raw_choices):
                    c = c.strip()
                    prefix = f"{chr(65+i)})"
                    if c.startswith(prefix):
                        c = c[len(prefix):].strip()
                    norm_choices.append(c)
            
            # Ensure exactly 4 choices
            while len(norm_choices) < 4:
                norm_choices.append("")
            
            sample_str += (
                fmt.format(
                    topic, topic.split("/")[-1], question, *norm_choices[:4], answer, explanation
                )
                + "\n\n"
            )
        return sample_str.strip()

    def clean_json_output(self, text: str) -> str:
        """Heuristic to clean JSON output from LLM"""
        text = text.strip()
        # Remove markdown code blocks
        if "```" in text:
            text = re.sub(r"```json", "", text)
            text = re.sub(r"```", "", text)
        
        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
        return text

    def build_prompt(
        self,
        topic: str,
        wadvsys: bool = True,
        wicl: bool = True,
        inc_samples: List[Dict[str, str]] | None = None,
    ) -> Tuple[str, str]:
        """Generate an MCQ based question on given topic with specified difficulty"""

        if wadvsys:
            sys_prompt = (
                "You are an expert examiner creating concise, multi-step logical reasoning questions.\n"
                "Your goal is to test deep understanding with minimal text.\n"
                "Strictly follow the JSON format constraints. Output ONLY valid JSON."
            )
        else:
            sys_prompt = "You are an examiner tasked with creating concise reasoning questions."
        
        tmpl = (
            "Generate a Structural Logic MCQ on topic: {0}.\n\n"
            "**CONSTRAINT CHECKLIST:**\n"
            "1. **Difficulty**: Enforce MINIMUM 3 deduction steps.\n"
            "2. **Distractors**: Must be based on Reversal mistake, Partial reasoning, or Arithmetic slip.\n"
            "3. **Conciseness**: Question text MUST be < 50 words. Total tokens < 150.\n"
            "4. **Choices**: Exactly 4 options (A-D). Each < 10 words.\n"
            "5. **Explanation**: Clear reasoning < 50 words.\n"
            "6. **Format**: Valid JSON only. NO markdown. NO code fences. NO comments.\n"
            "7. **Unambiguity**: NO logic traps. Must be solvable by rigorous elimination.\n"
            "8. **Alignment**: Reward structured reasoning. Punish shallow pattern guessing.\n\n"
            "{5}"
            "RESPONSE FORMAT (Strict JSON):\n"
            "{{\n"
            '  "topic": "{7}",\n'
            '  "question": "...",\n'
            '  "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],\n'
            '  "answer": "{8}",\n'
            '  "explanation": "..."\n'
            "}}"
        )
        # Remove model's preferential bias for options
        correct_option = random.choice(["A", "B", "C", "D"])
        distractors = ", ".join(
            [opt for opt in ["A", "B", "C", "D"] if opt != correct_option]
        )

        if wicl:
            inc_samples_ex = self.build_inc_samples(inc_samples, topic)
        else:
            inc_samples_ex = ""
        prompt = tmpl.format(
            topic,
            topic,
            correct_option,
            distractors,
            correct_option,
            inc_samples_ex,
            topic,
            topic.split("/")[-1],
            correct_option,
            correct_option,
        )

        return prompt, sys_prompt

    def generate_question(
        self,
        topic: Tuple[str, str] | List[Tuple[str, str]],
        wadvsys: bool,
        wicl: bool,
        inc_samples: Dict[str, List[Dict[str, str]]] | None,
        **gen_kwargs,
    ) -> Tuple[List[str], int | None, float | None]:
        """Generate a question prompt for the LLM with Validation & Retry"""
        
        # 1. Prepare Prompts
        prompts = []
        is_list = isinstance(topic, list)
        topic_list = topic if is_list else [topic]
        
        for t in topic_list:
            p, sp = self.build_prompt(
                f"{t[0]}/{t[1]}", wadvsys, wicl, inc_samples[t[1]] if inc_samples else None
            )
            prompts.append((p, sp))

        final_responses = [""] * len(prompts)
        final_tls = [0] * len(prompts)
        final_gts = [0.0] * len(prompts)
        
        # Indices of items that need generation (initially all)
        pending_indices = list(range(len(prompts)))
        attempts = 0
        max_attempts = 2

        while pending_indices and attempts < max_attempts:
            # Construct batch for pending items
            current_batch_prompts = [prompts[i][0] for i in pending_indices]
            # Use strict sys prompt from first item (assuming batch shares context logic)
            current_sys_prompt = prompts[0][1] 

            # Generate
            try:
                # If only 1 item, generate_response returns str, else list
                # We force list handling
                raw_resp, tls, gts = self.agent.generate_response(
                    current_batch_prompts, current_sys_prompt, **gen_kwargs
                )
                
                # Normalize to list
                if isinstance(raw_resp, str):
                    raw_resp = [raw_resp]
                    tls = [tls] if tls is not None else [0]
                    gts = [gts] if gts is not None else [0]
                elif not isinstance(raw_resp, list):
                     # Handle unexpected return?
                     raw_resp = [""] * len(pending_indices)
                
                # Check formatting and ensure listness for tls/gts
                # If tls is int (batch sum), distribute it
                num_in_batch = len(raw_resp)
                
                if tls is None: 
                    tls = [0] * num_in_batch
                elif isinstance(tls, (int, float)):
                    # Distribute total tokens evenly (approx)
                    avg_tok = tls / num_in_batch if num_in_batch > 0 else 0
                    tls = [avg_tok] * num_in_batch
                
                if gts is None:
                    gts = [0] * num_in_batch
                elif isinstance(gts, (int, float)):
                    avg_g = gts / num_in_batch if num_in_batch > 0 else 0
                    gts = [avg_g] * num_in_batch

            except Exception as e:
                print(f"Generation error: {e}")
                raw_resp = [""] * len(pending_indices)
                tls = [0] * len(pending_indices)
                gts = [0] * len(pending_indices)

            # Validate & Update
            new_pending = []
            for idx_in_batch, original_idx in enumerate(pending_indices):
                resp_text = raw_resp[idx_in_batch]
                
                # Update metrics (cumulative or last attempt? keep last attempt for now)
                final_tls[original_idx] = tls[idx_in_batch]
                final_gts[original_idx] = gts[idx_in_batch]
                
                # Repair
                cleaned = self.clean_json_output(resp_text)
                
                # Check Validity
                is_valid = False
                try:
                    if cleaned:
                        obj = json.loads(cleaned)
                        if self.validate_question(obj):
                            is_valid = True
                except:
                    pass
                
                if is_valid:
                    final_responses[original_idx] = cleaned
                else:
                    if attempts < max_attempts - 1:
                        new_pending.append(original_idx)
                    else:
                        # Final failure, keep empty string to signify failure.
                        final_responses[original_idx] = "" # Discard
            
            pending_indices = new_pending
            attempts += 1
        
        # Return format matching original signature
        # The original `generate_question` returns (List[str], total_tl, total_gt) if topic is list
        # and (str, total_tl, total_gt) if topic is single.
        # `generate_batches` expects total_tl and total_gt for the batch.
        
        total_tl = sum(filter(None, final_tls)) if final_tls else 0
        total_gt = sum(filter(None, final_gts)) if final_gts else 0
        
        if is_list:
            return final_responses, total_tl, total_gt
        else:
            # If original topic was single, return single response and total_tl/gt
            return final_responses[0], total_tl, total_gt

    def generate_batches(
        self,
        num_questions: int,
        topics: Dict[str, List[str]],
        batch_size: int = 5,
        wadvsys: bool = True,
        wicl: bool = True,
        inc_samples: Dict[str, List[Dict[str, str]]] | None = None,
        **kwargs,
    ) -> Tuple[List[str], List[int | None], List[float | None]]:
        r"""
        Generate questions in batches
        ---

        Args:
            - num_questions (int): Total number of questions to generate.
            - topics (Dict[str, List[str]]): Dictionary of topics with subtopics.
            - batch_size (int): Number of questions to generate in each batch.
            - wadvsys (bool): Whether to use advance prompt.
            - wicl (bool): Whether to include in-context learning (ICL) samples.
            - inc_samples (Dict[str, List[Dict[str, str]]]|None): In-context learning samples for the topics.
            - **kwargs: Additional keyword arguments for question generation.

        Returns:
            - Tuple[List[str], List[int | None], List[float | None]]: Generated questions, token lengths, and generation times.
        """
        extended_topics = self.populate_topics(topics, num_questions)
        questions = []
        tls, gts = [], []
        # Calculate total batches including the partial last batch
        total_batches = (len(extended_topics) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc="STEPS: ")

        for i in range(0, len(extended_topics), batch_size):
            batch_topics = extended_topics[i : i + batch_size]
            batch_questions = self.generate_question(
                batch_topics, wadvsys, wicl, inc_samples, **kwargs
            )
            questions.extend(batch_questions[0]), tls.append(
                batch_questions[1]
            ), gts.append(batch_questions[2])
            pbar.update(1)
        # for last batch with less than batch_size
        if len(extended_topics) % batch_size != 0:
            batch_topics = extended_topics[-(len(extended_topics) % batch_size) :]
            batch_questions = self.generate_question(
                batch_topics, wadvsys, wicl, inc_samples, **kwargs
            )
            questions.extend(batch_questions[0]), tls.append(
                batch_questions[1]
            ), gts.append(batch_questions[2])
            pbar.update(1)
        pbar.close()
        return questions, tls, gts

    def count_tokens_q(self, text: str) -> int:
        """Count the number of tokens using model.tokenizer"""
        if not hasattr(self.agent, "tokenizer"):
            raise AttributeError("The agent does not have a tokenizer attribute.")
        return len(self.agent.tokenizer.encode(text, add_special_tokens=False))

    def validate_question(self, q2: Dict[str, Any]) -> bool:
        """Validate question structure and content constraints"""
        # check required keys
        required_keys = ["topic", "question", "choices", "answer"]
        if not all((key in q2) for key in required_keys):
            return False
            
        # check choices format
        if not (isinstance(q2["choices"], list) and len(q2["choices"]) == 4):
            return False
            
        checks = all(
            isinstance(choice, str)
            and len(choice) > 2
            # Check if choices start with A) B) C) D) or just text? 
            # The prompt asks for "A) ...", "B) ...". 
            # But sometimes model outputs just text.
            # We strictly enforce 4 choices.
            # The A/B/C/D prefix check might be too strict if model omits it but structure is list.
            # Let's keep it simple: string and len > 2.
            for choice in q2["choices"]
        )
        if not checks:
            return False

        # check answer format
        # Check token length constraints (relaxed slightly for validation to avoid too many rejects, 
        # but we want STRICT for tournament)
        # prompt says < 150 tokens total.
        # We enforce < 350 max_new_tokens.
        # Let's just check answer is a single letter A/B/C/D.
        ans = q2["answer"]
        if not isinstance(ans, str):
            return False
        
        ans = ans.strip().upper()
        if len(ans) > 1:
            # Maybe it output "A) text". Extract A.
            if ans[0] in "ABCD" and (len(ans) == 1 or ans[1] in " )."):
                ans = ans[0]
            else:
                return False
        
        if ans not in "ABCD":
            return False
            
        return True

    def filter_questions(
        self, questions: List[str | Dict[str, str | Any]]
    ) -> List[Dict[str, str | Any]]:
        correct_format_question = []
        for i, q in enumerate(questions):
            if isinstance(q, dict):
                if self.validate_question(q):
                    correct_format_question.append(q)
            elif isinstance(q, str):
                try:
                    q1 = json.loads(q)
                    if self.validate_question(q1):
                        correct_format_question.append(q1)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON at index {i}: {q[:50]}...")
                    continue
            else:
                continue
        return correct_format_question

    def save_questions(self, questions: Any, file_path: str | Path) -> None:
        """Save generated questions to a JSON file"""
        # Ensure dir exist
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # Save to JSON file
        with open(file_path, "w") as f:
            json.dump(questions, f, indent=4)

    def populate_topics(
        self, topics: Dict[str, List[str]], num_questions: int
    ) -> List[str]:
        """Populate topics randomly to generate num_questions number of topics"""
        if not isinstance(topics, dict):
            raise ValueError(
                "Topics must be a dictionary with topic names as keys and lists of subtopics as values."
            )

        # Phase 3: Weighted topic selection
        weighted_subtopics = []
        weights = []
        for t, sublist in topics.items():
            weight = 1.0
            t_lower = t.lower()
            if "syllogism" in t_lower or "seating" in t_lower:
                weight = 2.5 # Increase priority
            elif "series" in t_lower:
                weight = 0.5 # Decrease priority
            
            for st in sublist:
                weighted_subtopics.append((t, st))
                weights.append(weight)

        if not weighted_subtopics:
            raise ValueError("No subtopics found in the provided topics dictionary.")
        
        selected_topics = random.choices(weighted_subtopics, weights=weights, k=num_questions)
        return selected_topics

    @staticmethod
    def load_icl_samples(file_path: str | Path) -> Dict[str, List[Dict[str, str]]]:
        """Load in-context learning samples from a JSON file"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        with open(file_path, "r") as f:
            samples = json.load(f)
        if not isinstance(samples, dict):
            raise ValueError("Samples must be inside dictionary.")
        return samples


# Example usage
if __name__ == "__main__":
    import argparse
    import yaml

    # ++++++++++++++++++++++++++
    # Run: python -m agents.question_agent --num_questions 20 --output_file outputs/questions.json --batch_size 5 --verbose
    # ++++++++++++++++++++++++++

    argparser = argparse.ArgumentParser(
        description="Generate questions using the QuestioningAgent."
    )
    argparser.add_argument(
        "--num_questions",
        type=int,
        default=10,
        help="Total number of questions to generate.",
    )
    argparser.add_argument(
        "--output_file",
        type=str,
        default="outputs/questions.json",
        help="Output file name to save the generated questions.",
    )
    argparser.add_argument(
        "--batch_size", type=int, default=5, help="Batch size for generating questions."
    )
    argparser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output for debugging."
    )
    args = argparser.parse_args()

    inc_samples = QuestioningAgent.load_icl_samples("assets/topics_example.json")

    # Load topics.json file.
    with open("assets/topics.json") as f:
        topics = json.load(f)

    agent = QuestioningAgent()
    # gen_kwargs = {"tgps_show": True, "max_new_tokens": 1024, "temperature": 0.1, "top_p": 0.9, "do_sample": True}
    gen_kwargs = {"tgps_show": True}
    with open("qgen.yaml", "r") as f:
        gen_kwargs.update(yaml.safe_load(f))

    question, tls, gts = agent.generate_batches(
        num_questions=args.num_questions,
        topics=topics,
        batch_size=args.batch_size,
        wadvsys=True,
        wicl=True,
        inc_samples=inc_samples,
        **gen_kwargs,
    )
    print(f"Generated {len(question)} questions!")
    if args.verbose:
        for q in question:
            print(q, flush=True)
        print("\n" + "=" * 50 + "\n\n")
        if gen_kwargs.get("tgps_show", False):
            print("Time taken per batch generation:", gts)
            print("Tokens generated per batch:", tls)
            print(
                f"Total Time Taken: {sum(gts):.3f} seconds; Total Tokens: {sum(tls)}; TGPS: {sum(tls)/sum(gts):.3f} seconds\n\n"
            )
        print("\n" + "+" * 50 + "\n")

    # check if question is JSON format
    ques = []
    for q in question:
        try:
            json.loads(q)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON format in question: {q}\nError: {e}")
            # use agent itself to extract JSON: Self-Reflection
            # the dictionary is not as expected.
            # TODO: IMPROVE THE FOLLOWING
            prompt = (
                "Extract **ONLY** the topic, question, choices, answer, and explanation while discarding the rest.\n"
                "Also please remove JSON code block text with backticks** like **```json** and **```**.\n\n"
                "String:\n"
                "{}\n\n"
                "Given Format:\n"
                "{{\n"
                '  "topic": "...",\n'
                '  "question": "...",\n'
                '  "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],\n'
                '  "answer": "Only the option letter (A, B, C, or D)",\n'
                '  "explanation": "..."\n'
                "}}"
            )
            q = agent.agent.generate_response(
                prompt.format(q),
                "You are an expert JSON extractor.",
                max_new_tokens=1024,
                temperature=0.0,
                do_sample=False,
            )
        ques.append(q)
    # Save the questions for later analysis
    agent.save_questions(ques, args.output_file)
    filtered_file_name = args.output_file.replace(
        "questions.json", "filtered_questions.json"
    )
    agent.save_questions(agent.filter_questions(ques), filtered_file_name)
    print(f"Saved to {args.output_file}!")

    # ========================================================================================
