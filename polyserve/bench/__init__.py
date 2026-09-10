"""Benchmarking: compare PolyServe's calibrated choice against stock defaults on the same workload."""

from polyserve.bench.compare import ComparisonResult, ComparisonRow, compare, results_path, to_markdown
from polyserve.bench.references import OllamaReference, ollama_tag_for, reference_configs
from polyserve.bench.report import Summary, summarize, write_report

__all__ = [
    "ComparisonResult",
    "ComparisonRow",
    "OllamaReference",
    "compare",
    "ollama_tag_for",
    "reference_configs",
    "results_path",
    "to_markdown",
    "Summary",
    "summarize",
    "write_report",
]
