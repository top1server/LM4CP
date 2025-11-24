from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import torch
import os


@dataclass(frozen=True)
class HFConfig:
    user: str = "tiendung6b"
    generator_dataset_name: str = "qwen3-4b-instruct-ioi_2024"
    hf_token: str = field(default_factory=lambda: os.environ.get("HF_TOKEN"))

    @property
    def generator_repo_id(self) -> str:
        return f"{self.user}/{self.generator_dataset_name}"


@dataclass(frozen=True)
class GenerationConfig:
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_new_tokens: int = 16384
    # max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.8
    num_solutions: int = 50
    seed_base: int = 0

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def seeds(self) -> list[int]:
        return [self.seed_base + i for i in range(self.num_solutions)]


@dataclass(frozen=True)
class IOIConfig:
    split: str = "test"
    year: Optional[int] = 2024
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)

    @property
    def solutions_dir(self) -> Path:
        directory = self.root / "model_solutions"
        directory.mkdir(parents=True, exist_ok=True)
        return directory
