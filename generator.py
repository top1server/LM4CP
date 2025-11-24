from __future__ import annotations
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from config import HFConfig, GenerationConfig, IOIConfig


@dataclass
class IOISolutionRecord:
    """
    Một lời giải cụ thể cho 1 subtask IOI với 1 seed.
    Các field chính:
    - year, day, problem_name, problem_id, target_subtask: metadata bài IOI.
    - seed: seed dùng để sinh mẫu này (0..49, ...).
    - prompt: toàn bộ prompt text mà Qwen nhận được (ít nhất là problem statement).
    - generation: raw text model trả về (bao gồm code, có thể kèm noise).
    - code: phần C++ đã tách ra từ generation (bên trong ```cpp ... ```).
    - uuid: id duy nhất, dùng để join/trace sau này.
    - model_kwargs: lưu seed
    - metadata: các info phụ (timestamp, usage, ...).
    """
    year: int
    day: int
    problem_name: str
    problem_id: str
    subtask: str
    target_subtask: str
    seed: int
    prompt: str
    generation: str
    code: str
    uuid: str
    model_kwargs: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "year": self.year,
            "day": self.day,
            "problem_name": self.problem_name,
            "problem_id": self.problem_id,
            "subtask": self.subtask,
            "target_subtask": self.target_subtask,
            "seed": self.seed,
            "prompt": self.prompt,
            "generation": self.generation,
            "code": self.code,
            "uuid": self.uuid,
            "model_kwargs": self.model_kwargs,
            "metadata": self.metadata,
        }


class IOISolutionGenerator:
    def __init__(
        self,
        hf_cfg: HFConfig | None = None,
        gen_cfg: GenerationConfig | None = None,
        ioi_cfg: IOIConfig | None = None,
    ) -> None:
        self.hf_cfg = hf_cfg or HFConfig()
        self.gen_cfg = gen_cfg or GenerationConfig()
        self.ioi_cfg = ioi_cfg or IOIConfig(split="test", year=2024)
        self._model: AutoModelForCausalLM | None = None
        self._tokenizer: AutoTokenizer | None = None


    @property
    def model(self) -> AutoModelForCausalLM:
        if self._model is None:
            print(f"[IOI] Loading model {self.gen_cfg.model_id} on {self.gen_cfg.device}...")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.gen_cfg.model_id,
                dtype="auto",
                device_map="auto" if self.gen_cfg.device == "cuda" else None,
            )
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.gen_cfg.model_id)
        return self._tokenizer


    def _load_ioi_subset(self):
        print(f"[IOI] Loading dataset open-r1/ioi (split={self.ioi_cfg.split})...")
        ds = load_dataset("open-r1/ioi", split=self.ioi_cfg.split)

        if self.ioi_cfg.year is not None:
            ds = ds.filter(lambda ex: ex["year"] == self.ioi_cfg.year)
            print(f"[IOI] Found {len(ds)} subtasks for IOI {self.ioi_cfg.year}.")
        else:
            print(f"[IOI] Found {len(ds)} subtasks for all years.")
        return ds
    

    @staticmethod
    def _build_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
        system_msg = (
            "You are an expert competitive programmer solving IOI problems. "
            "You must output a single C++17 solution that compiles with g++ and "
            "respects the given time and memory limits. "
            "Return ONLY code in a ```cpp fenced block. "
        )

        user_msg = (
            f"{example['problem']}\n"
            "Important:\n"
            "- Use the required function signature and starter code if provided.\n"
            "- Do not add extra main() if the grader expects only functions.\n"
            "- Do not read or write files unless explicitly required.\n"
            "- Avoid extra debug prints.\n"
        )

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

    @staticmethod
    def _extract_code_from_completion(completion_text: str) -> str:
        """
        Tách phần code C++ từ output model.
        - Nếu model trả về dạng ```cpp ... ``` thì lấy phần bên trong.
        - Nếu không thấy block ```cpp thì fallback = toàn bộ completion_text.
        """
        code = completion_text
        if "```cpp" in completion_text:
            part = completion_text.split("```cpp", 1)[1]
            if "```" in part:
                part = part.split("```", 1)[0]
            code = part
        return code.strip()
    
    def _generate_for_example(
        self,
        example: Dict[str, Any],
        seed: int,
    ) -> Tuple[str, str, str, Dict[str, int]]:
        messages = self._build_messages(example)

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer(
            [prompt_text],
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.model.device)

        prompt_tokens = model_inputs.input_ids.shape[1]
        prompt_ids = model_inputs["input_ids"][0]
        
        torch.manual_seed(seed)
        if self.model.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        with torch.no_grad():
            output_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.gen_cfg.max_new_tokens,
                do_sample=True,
                temperature=self.gen_cfg.temperature,
                top_p=self.gen_cfg.top_p,
            )[0]

        completion_ids = output_ids[len(prompt_ids) :]
        completion_text = self.tokenizer.decode(
            completion_ids,
            skip_special_tokens=True,
        ).strip()

        completion_tokens = len(completion_ids)
        total_tokens = prompt_tokens + completion_tokens
        code = self._extract_code_from_completion(completion_text)
        usage = {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(total_tokens),
        }

        return prompt_text, completion_text, code, usage


    def _solution_filename(self, example: Dict[str, Any], seed: int) -> str:
        """
        Đặt tên file .cpp cho 1 (subtask, seed).
        Định dạng:
            {year}_{problem_id}_{subtask}_s{seed:02d}_qwen3_4b_instruct_2507.cpp
        """
        year = example["year"]
        prob_id = example["id"]
        subtask = example["subtask"]
        return f"{year}_{prob_id}_{subtask}_s{seed:02d}_qwen3_4b_instruct_2507.cpp"

    def _load_or_generate_code(
        self,
        example: Dict[str, Any],
        seed: int,
    ) -> Tuple[str, str, str]:
        """
        Nếu đã có file .cpp tương ứng seed này thì load, tránh gọi model lại.
        Ngược lại:
        - Gọi model sinh code với seed.
        - Lưu code ra file .cpp.
        - Trả về (prompt_text, generation, code, usage).
        """
        fname = self._solution_filename(example, seed)
        fpath = self.ioi_cfg.solutions_dir / fname

        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                code = f.read()
            return "", code, code, {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

        prompt_text, generation, code, usage = self._generate_for_example(example, seed)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(code)
        return prompt_text, generation, code, usage


    def _build_record(self, example: Dict[str, Any], seed: int) -> IOISolutionRecord:
        """
        Sinh (hoặc load) code cho (example, seed) và đóng gói thành IOISolutionRecord.
        """
        prompt_text, generation, code, usage = self._load_or_generate_code(example, seed)
        uid = str(uuid.uuid4())
        model_kwargs = {
            "temperature": self.gen_cfg.temperature,
            "top_p": self.gen_cfg.top_p,
            "max_new_tokens": self.gen_cfg.max_new_tokens,
            "seed": seed,
        }

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "usage": usage,
        }

        return IOISolutionRecord(
            year=example["year"],
            day=example["day"],
            problem_name=example["name"],
            problem_id=example["id"],
            subtask=example["subtask"],
            target_subtask=example["subtask"],
            seed=seed,
            prompt=prompt_text,
            generation=generation,
            code=code,
            uuid=uid,
            model_kwargs=model_kwargs,
            metadata=metadata,
        )

    def build_records(self) -> List[Dict[str, Any]]:
        """
        Sinh toàn bộ record cho IOI (theo config, mặc định IOI 2024).
        Mỗi (problem, subtask) sẽ có `num_solutions` record, mỗi record một seed.
        """
        ds = self._load_ioi_subset()
        records: List[Dict[str, Any]] = []

        for ex in tqdm(ds, desc="Generating Qwen3 solutions for IOI"):
            for seed in self.gen_cfg.seeds:
                record = self._build_record(ex, seed)
                records.append(record.to_dict())

        print(
            f"[IOI] Prepared {len(records)} solution records "
            f"({len(ds)} subtasks x {len(self.gen_cfg.seeds)} seeds)."
        )
        return records

    @staticmethod
    def to_dataset(records: List[Dict[str, Any]]) -> Dataset:
        return Dataset.from_list(records)

    def push_to_hf(self, dataset: Dataset, push: bool | None = None) -> None:
        """
        Push dataset solutions lên HuggingFace Hub nếu push=True.
        - Nếu push=None (default), sẽ đọc biến môi trường PUSH_TO_HF.
          + PUSH_TO_HF="1" -> push
          + ngược lại -> không push.
        """
        if push is None:
            push = os.environ.get("PUSH_TO_HF", "0") == "1"

        if not push:
            print("[IOI] PUSH_TO_HF != 1, không push dataset lên HuggingFace.")
            return

        repo_id = self.hf_cfg.generator_repo_id
        print(f"[IOI] Pushing solutions dataset to {repo_id} (split=train)...")
        dataset.push_to_hub(repo_id, split="train")
        print("[IOI] Done pushing to HuggingFace.")


    def run(self, push: bool | None = None) -> Dataset:
        """
        Hàm high-level:
        1) build_records() -> list[dict]
        2) to_dataset() -> Dataset
        3) push_to_hf() (tuỳ chọn)
        Trả về object Dataset, có thể inspect/debug nếu cần.
        """
        records = self.build_records()
        dataset = self.to_dataset(records)
        self.push_to_hf(dataset, push)
        return dataset
    

if __name__ == "__main__":
    generator = IOISolutionGenerator()
    generator.run(push=None)


