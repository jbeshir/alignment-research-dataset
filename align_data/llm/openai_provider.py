import json
import logging

from align_data.embeddings.embedding_utils import openai_client, handle_openai_errors
from align_data.llm.provider import ArticleAnalysis, LLMProvider
from align_data.settings import LLM_MODEL, LLM_REASONING_EFFORT

logger = logging.getLogger(__name__)

CATEGORIES = [
    "Interpretability",
    "Alignment Theory",
    "Governance",
    "Technical Safety",
    "AI Ethics",
    "Capabilities",
    "Forecasting",
    "Other",
]

SYSTEM_PROMPT = f"""\
You are an AI alignment research analyst. Analyze the given article and produce a structured analysis in JSON format.

The JSON must have these fields:
- "summary": A 1-3 sentence summary of the article.
- "key_points": A list of 3-5 key takeaways as strings.
- "implication": 1-2 sentences on implications for AI alignment.
- "category": One of: {", ".join(CATEGORIES)}.

Respond with only valid JSON, no other text."""

MAX_TEXT_CHARS = 400_000


class OpenAIProvider(LLMProvider):
    def __init__(self):
        if not openai_client:
            raise RuntimeError(
                "OpenAI client not configured. Set OPENAI_API_KEY environment variable."
            )
        self.client = openai_client

    @handle_openai_errors
    def analyze_article(self, title: str, text: str, source: str) -> ArticleAnalysis:
        truncated_text = text[:MAX_TEXT_CHARS] if text else ""

        user_message = (
            f"Title: {title}\n"
            f"Source: {source}\n\n"
            f"Article text:\n{truncated_text}"
        )

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            reasoning_effort=LLM_REASONING_EFFORT,
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        category = data.get("category", "Other")
        if category not in CATEGORIES:
            category = "Other"

        key_points = data.get("key_points", [])
        if not isinstance(key_points, list):
            key_points = [str(key_points)]

        return ArticleAnalysis(
            summary=data.get("summary", ""),
            key_points=key_points,
            implication=data.get("implication", ""),
            category=category,
        )
