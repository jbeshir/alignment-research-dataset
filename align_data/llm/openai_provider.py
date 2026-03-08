import logging
from typing import Literal

from pydantic import BaseModel

from align_data.embeddings.embedding_utils import openai_client, handle_openai_errors
from align_data.llm.provider import AnalysisResult, ArticleAnalysis, LLMProvider, TokenUsage
from align_data.settings import LLM_MODEL, LLM_REASONING_EFFORT

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an AI alignment research analyst. Analyze the given article and produce a structured analysis in JSON format.

Base your analysis ONLY on the content provided. Do not add information from your own knowledge. If the article is very short or merely links to other resources, keep your analysis brief and proportional to the actual content.

The JSON must have these fields:

- "summary": A 1-3 sentence summary. Write it as a direct description of the subject matter, NOT a description of the article itself. Never start with "The article", "This article", "The post", or "The author". Instead, start with the actual subject. Example: instead of "The article argues that reward hacking is common", write "Reward hacking is common in frontier models because...".

- "key_points": A list of 3-5 key takeaways as strings (fewer for short articles). Each point should state the insight directly, not describe what the article says. Avoid phrases like "The article argues", "The author proposes", "The paper shows". Instead, state the finding or claim itself. Never include empty strings.

- "implication": 1-2 sentences on implications for AI alignment research or practice. Be specific to this article's content rather than generic. Do not start with "For AI alignment".

- "category": Classify into exactly one of these categories:
  - "Interpretability": Mechanistic interpretability, probing, circuits, features, SAEs, attribution, steering vectors
  - "Safety Techniques": Alignment proposals, scalable oversight, debate, monitoring, RLHF/training methods, reward hacking, formal verification, corrigibility
  - "Governance & Policy": AI regulation, policy, organizational strategy, industry analysis
  - "Deception & Misalignment": Alignment faking, scheming, emergent misalignment, deceptive AI behavior, insider threats, sandbagging
  - "AI Capabilities & Behavior": Model capabilities, reasoning, hallucination, jailbreaking, evaluations, benchmarks, prompting
  - "Risks & Strategy": Existential risk, x-risk scenarios, strategic analysis, disempowerment, power dynamics, conflict
  - "Forecasting": Timelines, scaling laws, progress measurement, future scenarios, compute trends
  - "AI & Society": AI consciousness/sentience, ethics, societal impact, persuasion, AI companions
  - "Field Building": Community building, education, careers, organizations, conferences, outreach (not technical benchmarks or tools)
  - "Other": Content that doesn't fit the above

Respond with only valid JSON, no other text."""

MAX_TEXT_CHARS = 400_000


class AnalysisResponse(BaseModel):
    summary: str
    key_points: list[str]
    implication: str
    category: Literal[
        "Interpretability",
        "Safety Techniques",
        "Governance & Policy",
        "Deception & Misalignment",
        "AI Capabilities & Behavior",
        "Risks & Strategy",
        "Forecasting",
        "AI & Society",
        "Field Building",
        "Other",
    ]


class OpenAIProvider(LLMProvider):
    def __init__(self):
        if not openai_client:
            raise RuntimeError(
                "OpenAI client not configured. Set OPENAI_API_KEY environment variable."
            )
        self.client = openai_client

    @handle_openai_errors
    def analyze_article(self, title: str, text: str, source: str) -> AnalysisResult:
        truncated_text = text[:MAX_TEXT_CHARS] if text else ""

        user_message = (
            f"Title: {title}\n"
            f"Source: {source}\n\n"
            f"Article text:\n{truncated_text}"
        )

        response = self.client.beta.chat.completions.parse(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format=AnalysisResponse,
            reasoning_effort=LLM_REASONING_EFFORT,
        )

        usage = TokenUsage()
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                model=response.model,
            )

        parsed = response.choices[0].message.parsed

        return AnalysisResult(
            analysis=ArticleAnalysis(
                summary=parsed.summary,
                key_points=parsed.key_points,
                implication=parsed.implication,
                category=parsed.category,
            ),
            usage=usage,
        )
