from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Union
from html_eval.core.llm import LLMClient, NvidiaLLMClient, VLLMClient, LiteLLMClient

@dataclass
class LLMClientConfig:
    llm_source: str = "nvidia"
    model_name: str = "google/gemma-3n-e2b-it"
    api_key: Optional[str] = None
    temperature: float = 0.0
    top_p: float = 0.7
    max_tokens: int = 8192
    seed: int = 42
    enable_thinking: bool = False

    # LiteLLM-specific (ignored by other backends)
    api_base: Optional[str] = None          # override endpoint, e.g. for Ollama / LiteLLM proxy
    extra_params: Dict[str, Any] = field(default_factory=dict)  # passed verbatim to litellm.completion
    
    engine_args: Dict[str, Any] = field(default_factory=dict)
    
    # CHANGE: Type hint now supports Dict containing config details
    # Example: { "extractor": {"path": "...", "temperature": 0.1}, "creative": {"path": "...", "temperature": 0.9} }
    lora_modules: Optional[Dict[str, Union[str, Dict[str, Any]]]] = None 

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

        if self.llm_source == "vllm":
            config["engine_args"] = self.engine_args
            config["lora_modules"] = self.lora_modules
            return VLLMClient(config=config)
        elif self.llm_source == "nvidia":
            return NvidiaLLMClient(config=config)
        elif self.llm_source == "litellm":
            config["api_base"] = self.api_base
            config["extra_params"] = self.extra_params
            return LiteLLMClient(config=config)
        else:
            raise ValueError(
                f"Unsupported llm_source: '{self.llm_source}'. "
                "Choose from: 'nvidia', 'vllm', 'litellm'."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)