from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from html_eval.core.llm import LLMClient, NvidiaLLMClient, VLLMClient


@dataclass
class LLMClientConfig:
    llm_source: str = "nvidia"
    model_name: str = "google/gemma-3n-e2b-it"
    api_key: Optional[str] = None  # Used only for API-based clients
    temperature: float = 0.0
    top_p: float = 0.7
    max_tokens: int = 8192
    seed: int = 42
    enable_thinking: bool = False
    
    # vLLM-specific args (engine init, like tensor_parallel, gpu_mem_utilization, etc.)
    engine_args: Dict[str, Any] = field(default_factory=dict)
    lora_modules: Optional[Dict[str, str]] = None  # e.g., {"lora_A_path": "path/to/lora_A", "lora_B_path": "path/to/lora_B"}

    def create_llm_client(self) -> LLMClient:
        config = {
            "model_name": self.model_name,
            "api_key": self.api_key,
            "generation_config": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
            },
            "enable_thinking": self.enable_thinking,
        }

        # only pass engine_args if vLLM
        if self.llm_source == "vllm":
            config["engine_args"] = self.engine_args
            config["lora_modules"] = self.lora_modules
            return VLLMClient(config=config)
        elif self.llm_source == "nvidia":
            return NvidiaLLMClient(config=config)
        else:
            raise ValueError(f"Unsupported llm_source: {self.llm_source}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
