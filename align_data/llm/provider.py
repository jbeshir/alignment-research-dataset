import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from align_data.settings import LLM_PROVIDER

logger = logging.getLogger(__name__)


@dataclass
class ArticleAnalysis:
    summary: str
    key_points: list[str]
    implication: str
    category: str


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""


@dataclass
class AnalysisResult:
    analysis: ArticleAnalysis
    usage: TokenUsage


class LLMProvider(ABC):
    @abstractmethod
    def analyze_article(self, title: str, text: str, source: str) -> AnalysisResult:
        """Analyze an article and return structured analysis with token usage."""
        ...


def create_llm_provider() -> LLMProvider:
    """Factory function to create an LLM provider based on configuration."""
    if LLM_PROVIDER == "openai":
        from align_data.llm.openai_provider import OpenAIProvider

        return OpenAIProvider()
    else:
        raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}")
