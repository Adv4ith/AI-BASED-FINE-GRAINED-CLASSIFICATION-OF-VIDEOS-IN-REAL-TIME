"""
llm/__init__.py — Public API for the LLM surveillance explanation package.
"""
from .llama_engine          import LlamaEngine, DEFAULT_MODEL, SMALL_MODEL
from .prompt_builder        import (
    build_context, build_llama3_prompt, build_fallback_paragraph,
    describe_attention, assign_risk, PromptContext, SYSTEM_PROMPT,
)
from .explanation_generator import SurveillanceExplainer, ExplanationOutput
from .report_writer         import ReportWriter

__all__ = [
    "LlamaEngine", "DEFAULT_MODEL", "SMALL_MODEL",
    "build_context", "build_llama3_prompt", "build_fallback_paragraph",
    "describe_attention", "assign_risk", "PromptContext", "SYSTEM_PROMPT",
    "SurveillanceExplainer", "ExplanationOutput",
    "ReportWriter",
]
