from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
import os
import random
import time
from typing import List, Iterable, Optional, Any, Callable, Dict

from openai import OpenAI
from openai import RateLimitError

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError:
    raise ImportError("vLLM is not installed. Please install it with 'pip install vllm'")
import threading
_vllm_init_lock = threading.Lock()

def retry_on_ratelimit(max_retries=5, base_delay=1.0, max_delay=10.0):
    def deco(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            delay = base_delay
            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except RateLimitError:
                    if attempt == max_retries - 1:
                        raise
                    sleep = min(max_delay, delay) + random.uniform(0, delay)
                    time.sleep(sleep)
                    delay *= 2
        return wrapped
    return deco


class LLMClient(ABC):
    """
    Abstract base class for calling LLM APIs.
    Provides a default call_batch implementation that calls call_api in parallel.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    @abstractmethod
    def call_api(self, prompt: str, **kwargs) -> str:
        """
        Call the underlying LLM API with a single prompt.
        Must be implemented by subclasses.

        Args:
            prompt: prompt string
            kwargs: vendor-specific options

        Returns:
            response string
        """
        raise NotImplementedError

    def call_batch(
        self,
        prompts: Iterable[str],
        max_workers: int = 8,
        chunk_size: Optional[int] = None,
        raise_on_error: bool = False,
        per_result_callback: Optional[Callable[[int, Optional[str], Optional[Exception]], Any]] = None,
        **call_api_kwargs,
    ) -> List[Optional[str]]:
        """
        Default batch implementation: runs call_api in parallel with a ThreadPoolExecutor.
        Preserves order of input prompts in returned list.

        Args:
            prompts: iterable of prompt strings
            max_workers: max parallel workers
            chunk_size: if set, splits the prompts into chunks of this size and processes sequentially.
                        Useful to limit concurrency or rate.
            raise_on_error: if True, re-raise the exception when any prompt fails after retries.
            per_result_callback: optional function called as callback(idx, result, exception) for each finished prompt.
            call_api_kwargs: forwarded to call_api.

        Returns:
            list of responses (or None for failed items if raise_on_error is False)
        """
        prompts = list(prompts)
        results: List[Optional[str]] = [None] * len(prompts)

        def _submit_range(start_idx: int, end_idx: int):
            # submit jobs for a slice [start_idx, end_idx)
            futures = {}
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for i in range(start_idx, end_idx):
                    fut = ex.submit(self.call_api, prompts[i], **call_api_kwargs)
                    futures[fut] = i
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        res = fut.result()
                        results[idx] = res
                        if per_result_callback:
                            per_result_callback(idx, res, None)
                    except Exception as exc:
                        # If caller wants to raise, do so; otherwise set None and continue
                        if per_result_callback:
                            per_result_callback(idx, None, exc)
                        if raise_on_error:
                            raise
                        results[idx] = None

        if chunk_size is None or chunk_size <= 0:
            # one-shot submit all prompts (bounded by max_workers in each executor)
            _submit_range(0, len(prompts))
        else:
            # process chunks sequentially to throttle
            for start in range(0, len(prompts), chunk_size):
                end = min(start + chunk_size, len(prompts))
                _submit_range(start, end)

        return results


class NvidiaLLMClient(LLMClient):
    """
    Concrete implementation of LLMClient for the NVIDIA API (non-streaming).
    """

    def __init__(self, config: dict):
        super().__init__(config)
        api_key = config.get("api_key") or os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError(
                "API key for NVIDIA must be provided in config['api_key'] or NVIDIA_API_KEY env var."
            )

        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        self.model_name = config.get("model_name", "google/gemma-3-1b-it")

        gen_conf = config.get("generation_config", {})
        self.temperature = gen_conf.get("temperature", 0)
        self.top_p = gen_conf.get("top_p", 0.7)
        self.max_tokens = gen_conf.get("max_tokens", 8192)

    def set_model(self, model_name: str):
        self.model_name = model_name

    @retry_on_ratelimit(max_retries=20, base_delay=0.5, max_delay=5.0)
    def call_api(self, prompt: str, **kwargs) -> str:
        """
        Single prompt call (non-streaming).
        kwargs are forwarded to the underlying API call if needed.
        """
        # print("prompt:", prompt)  # keep optional for debugging
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role":"system","content":"Reasoning: high"},{"role": "user", "content": prompt}],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            extra_body={"chat_template_kwargs": {"thinking": True}},
            # any additional kwargs can be merged here if needed
        )
        return response.choices[0].message.content

    # Optionally override call_batch if the vendor supports true batched calls.
    # For now, we inherit the default implementation from LLMClient.

class VLLMClient(LLMClient):
    def __init__(self, config: dict):
        super().__init__(config)
        
        model_name = config.get("model_name")
        if not model_name:
            raise ValueError("model_name must be provided in the config for VLLMClient")

        self.engine_args = config.get("engine_args", {})
        
        # 1. Separate LoRA paths from their Configs
        self.lora_config_raw = config.get("lora_modules", {}) or {}
        
        # We need to inform vLLM that we are using LoRA
        if self.lora_config_raw or config.get("enable_lora", False):
            self.engine_args["enable_lora"] = True
            # Optional: Increase this if you want to swap between 3 adapters efficiently
            self.engine_args["max_loras"] = min(len(self.lora_config_raw), 4)

        # Initialize vLLM
        with _vllm_init_lock:
            self.llm = LLM(model=model_name, max_lora_rank=128, **self.engine_args)

        # --- NEW: AUTO-DETECT CONTEXT LENGTH ---
        # This grabs the actual limit (e.g., 8192, 32768) from the loaded model config
        self.context_window_size = self.engine_args.get("max_model_len", 2048)
        print(f"VLLMClient: Auto-detected max_model_len = {self.context_window_size}")
        # ---------------------------------------

        # 2. Register Adapters and Store Defaults
        self.lora_requests: Dict[str, LoRARequest] = {}
        self.adapter_defaults: Dict[str, Dict[str, Any]] = {} # Store temp/top_p per adapter
        self._lora_id_counter = 1
        
        for name, data in self.lora_config_raw.items():
            # Handle both string path (legacy) and dict config (new)
            if isinstance(data, str):
                path = data
                defaults = {}
            else:
                path = data.get("path")
                # Extract generation params to save for later
                defaults = {k: v for k, v in data.items() if k != "path"}
            
            self.load_adapter(name, path)
            self.adapter_defaults[name] = defaults

        self._generate_lock = threading.Lock()
        
        # Global Defaults
        gen_conf = config.get("generation_config", {})
        self.temperature = gen_conf.get("temperature", 0.0)
        self.top_p = gen_conf.get("top_p", 1.0)
        self.max_tokens = gen_conf.get("max_tokens", 512)
        self.stop_sequences = gen_conf.get("stop", [])
        self.enable_thinking = config.get("enable_thinking", False)

    def load_adapter(self, name: str, path: str):
        if name in self.lora_requests: return
        self.lora_requests[name] = LoRARequest(
            lora_name=name,
            lora_int_id=self._lora_id_counter,
            lora_path=path
        )
        self._lora_id_counter += 1

    def _format_prompt_with_thinking(self, prompt: str, thinking: bool) -> str:
        if self.enable_thinking or thinking:
            return f"{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            return f"{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def _get_lora_request(self, adapter_name: Optional[str]) -> Optional[LoRARequest]:
        if not adapter_name: return None
        return self.lora_requests.get(adapter_name)

    def _create_sampling_params(self, adapter_name: str = None, **kwargs) -> SamplingParams:
        # Start with Global Defaults
        params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stop": self.stop_sequences
        }
        
        # Override with Adapter Defaults (if defined in config)
        if adapter_name and adapter_name in self.adapter_defaults:
            params.update(self.adapter_defaults[adapter_name])
            
        # Override with Call-time arguments (highest priority)
        params.update({k: v for k, v in kwargs.items() if v is not None})

        
        # --- FIX: Ensure room for generation ---
        max_gen_tokens = params["max_tokens"]
        # Allow prompt to take up remaining space, minus a small buffer (1 token)
        safe_truncate_len = self.context_window_size - max_gen_tokens - 1
        
        if safe_truncate_len < 128:
            print(f"WARNING: Max context ({self.context_window_size}) is too close to max_generation ({max_gen_tokens}). Truncation might be aggressive.")
            # Fallback to prevent crash: ensure at least some context
            safe_truncate_len = self.context_window_size // 2
        # ---------------------------------------

        return SamplingParams(
            temperature=params["temperature"],
            top_p=params["top_p"],
            max_tokens=params["max_tokens"],
            stop=params["stop"],
            truncate_prompt_tokens=safe_truncate_len,
        )

    def call_batch(
        self,
        prompts: Iterable[str],
        adapter_name: str = None,
        thinking: bool = False,
        **call_api_kwargs,
    ) -> List[Optional[str]]:
        
        prompts = list(prompts)
        
        # pass adapter_name to sampling params creator to fetch specific temp
        sampling_params = self._create_sampling_params(adapter_name=adapter_name, **call_api_kwargs)
        
        prompts = [self._format_prompt_with_thinking(p, thinking) for p in prompts]
        lora_req = self._get_lora_request(adapter_name)
        
        # print(f"Using Adapter: {adapter_name} | Temp: {sampling_params.temperature}")

        with self._generate_lock:
            outputs = self.llm.generate(prompts, sampling_params, lora_request=lora_req)

        return [output.outputs[0].text for output in outputs]

    # Don't forget to update call_api similarly if you use it individually
    def call_api(self, prompt: str, adapter_name: str = None, thinking=False, **kwargs) -> str:
        sampling_params = self._create_sampling_params(adapter_name=adapter_name, **kwargs)
        prompt = self._format_prompt_with_thinking(prompt, thinking)
        lora_req = self._get_lora_request(adapter_name)

        with self._generate_lock:
            outputs = self.llm.generate([prompt], sampling_params, lora_request=lora_req)
        return outputs[0].outputs[0].text