"""
llm/llama_engine.py
====================
Llama 3.1 inference — supports TWO modes:

  MODE 1: HF Inference API  (use_api=True)   ← NO download, uses your HF token
  MODE 2: Local model       (use_api=False)  ← downloads ~16GB, runs on GPU/CPU

API mode uses HuggingFace's hosted serverless inference endpoint.
Free tier: ~1000 requests/day. No GPU needed. Response in 2-5 seconds.

Usage
-----
  # API mode (recommended — no download)
  engine = LlamaEngine(use_api=True, hf_token="hf_...")
  text   = engine.generate(prompt)

  # Local mode (download once, runs offline)
  engine = LlamaEngine(use_api=False, load_in_4bit=True)
  text   = engine.generate(prompt)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

log = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("LLAMA_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
SMALL_MODEL   = "meta-llama/Llama-3.2-3B-Instruct"


class LlamaEngine:
    """
    Unified Llama 3 inference engine.

    Parameters
    ----------
    model_name    : HuggingFace model ID
    use_api       : True  → HF Inference API (no download, needs token)
                    False → local model (downloads weights)
    hf_token      : HuggingFace token (required for gated models + API mode)
    device        : "auto" | "cuda" | "cpu"  (local mode only)
    load_in_4bit  : 4-bit quantization (local mode only, saves VRAM)
    max_new_tokens: max tokens to generate
    temperature   : sampling temperature
    """

    def __init__(
        self,
        model_name:     str   = DEFAULT_MODEL,
        use_api:        bool  = True,           # DEFAULT: API mode (no download)
        hf_token:       Optional[str] = None,
        device:         str   = "auto",
        load_in_4bit:   bool  = False,
        load_in_8bit:   bool  = False,
        max_new_tokens: int   = 400,
        temperature:    float = 0.2,
        top_p:          float = 0.9,
    ) -> None:
        self.model_name     = model_name
        self.use_api        = use_api
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature
        self.top_p          = top_p
        self._hf_token      = hf_token or os.environ.get("HF_TOKEN")
        self._loaded        = False

        # Local mode only
        self._device       = device
        self._load_in_4bit = load_in_4bit
        self._load_in_8bit = load_in_8bit
        self._model        = None
        self._tokenizer    = None

        mode = "API (no download)" if use_api else "LOCAL (downloads weights)"
        log.info("[LlamaEngine] Mode=%s  model=%s", mode, model_name)
        print(f"[LlamaEngine] Mode: {mode}")
        print(f"[LlamaEngine] Model: {model_name}")

    # ──────────────────────────────────────────
    # Public: generate
    # ──────────────────────────────────────────

    def generate(self, prompt: str) -> str:
        """Generate text from a Llama 3 chat prompt string."""
        if self.use_api:
            return self._generate_api(prompt)
        else:
            return self._generate_local(prompt)

    # ──────────────────────────────────────────
    # MODE 1: HuggingFace Inference API
    # ──────────────────────────────────────────

    def _generate_api(self, prompt: str) -> str:
        """Call HF serverless inference API — no local model needed."""
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise RuntimeError("pip install huggingface_hub>=0.22")

        if not self._hf_token:
            raise RuntimeError(
                "HF_TOKEN not set. Run: $env:HF_TOKEN='hf_...'\n"
                "Or pass --hf-token to api_server.py"
            )

        client = InferenceClient(
            model = self.model_name,
            token = self._hf_token,
        )

        t0 = time.perf_counter()

        # Extract user message from Llama 3 chat format
        # (HF InferenceClient takes messages list, not raw prompt string)
        user_text = _extract_user_message(prompt)

        response = client.chat_completion(
            messages = [
                {"role": "system", "content": (
                    "You are an AI surveillance analyst. Write exactly one concise "
                    "paragraph (2-3 sentences) analyzing the detected activity. "
                    "Be factual and direct. No bullet points, no headers."
                )},
                {"role": "user", "content": user_text},
            ],
            max_tokens  = self.max_new_tokens,
            temperature = self.temperature,
            top_p       = self.top_p,
        )

        elapsed = time.perf_counter() - t0
        result  = response.choices[0].message.content.strip()
        log.info("[LlamaEngine] API response: %d chars in %.1fs", len(result), elapsed)
        return result

    # ──────────────────────────────────────────
    # MODE 2: Local model (existing behaviour)
    # ──────────────────────────────────────────

    def load(self) -> None:
        """Load local model weights. Called once on first generate()."""
        if self._loaded or self.use_api:
            return

        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            BitsAndBytesConfig,
        )

        if self._device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"\n[LlamaEngine] Loading local model {self.model_name} ...")
        t0 = time.perf_counter()

        token_kwargs = {"token": self._hf_token} if self._hf_token else {}

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, use_fast=True, **token_kwargs
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        bnb_cfg = None
        if (self._load_in_4bit or self._load_in_8bit) and self._device == "cuda":
            try:
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=self._load_in_4bit,
                    load_in_8bit=self._load_in_8bit,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except Exception as e:
                log.warning("[LlamaEngine] bitsandbytes failed: %s", e)

        dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="auto" if self._device == "cuda" else None,
            quantization_config=bnb_cfg,
            low_cpu_mem_usage=True,
            **token_kwargs,
        )
        if self._device == "cpu":
            self._model = self._model.to("cpu")
        self._model.eval()

        elapsed = time.perf_counter() - t0
        print(f"[LlamaEngine] Loaded in {elapsed:.1f}s")
        self._loaded = True

    def _generate_local(self, prompt: str) -> str:
        import torch
        if not self._loaded:
            self.load()

        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=3072, padding=False, add_special_tokens=False,
        ).to(self._model.device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens     = self.max_new_tokens,
                do_sample          = self.temperature > 0,
                temperature        = self.temperature if self.temperature > 0 else 1.0,
                top_p              = self.top_p,
                pad_token_id       = self._tokenizer.eos_token_id,
                repetition_penalty = 1.1,
            )
        new_ids  = output_ids[0][inputs["input_ids"].shape[-1]:]
        raw_text = self._tokenizer.decode(new_ids, skip_special_tokens=True)
        elapsed  = time.perf_counter() - t0
        log.info("[LlamaEngine] Local: %d tokens in %.1fs", len(new_ids), elapsed)
        return raw_text.strip()

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    def unload(self) -> None:
        if self._model is not None:
            del self._model; self._model = None
        if self._tokenizer is not None:
            del self._tokenizer; self._tokenizer = None
        self._loaded = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def __enter__(self): self.load(); return self
    def __exit__(self, *a): self.unload()

    @property
    def is_loaded(self) -> bool:
        return self._loaded or self.use_api


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _extract_user_message(llama3_prompt: str) -> str:
    """Pull the user-turn text from a Llama 3 chat prompt string."""
    try:
        # Find content between user header and eot_id
        marker = "<|start_header_id|>user<|end_header_id|>\n\n"
        start  = llama3_prompt.index(marker) + len(marker)
        end    = llama3_prompt.index("<|eot_id|>", start)
        return llama3_prompt[start:end].strip()
    except ValueError:
        # Fallback: return the whole prompt
        return llama3_prompt
