from unittest.mock import patch, MagicMock

import pytest

from align_data.llm.openai_provider import AnalysisResponse, MAX_TEXT_CHARS
from align_data.llm.provider import ArticleAnalysis


def _make_mock_response(parsed: AnalysisResponse):
    """Build a mock OpenAI parse() response with the given parsed object."""
    message = MagicMock()
    message.parsed = parsed
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_provider():
    """Create an OpenAIProvider with a mocked client."""
    from align_data.llm.openai_provider import OpenAIProvider
    with patch("align_data.llm.openai_provider.openai_client", MagicMock()):
        provider = OpenAIProvider()
    return provider


def test_constructor_raises_when_no_client():
    from align_data.llm.openai_provider import OpenAIProvider
    with patch("align_data.llm.openai_provider.openai_client", None):
        with pytest.raises(RuntimeError, match="OpenAI client not configured"):
            OpenAIProvider()


def test_analyze_article_returns_analysis():
    provider = _make_provider()

    parsed = AnalysisResponse(
        summary="A summary",
        key_points=["point 1", "point 2"],
        implication="An implication",
        category="Interpretability",
    )
    provider.client.beta.chat.completions.parse.return_value = _make_mock_response(parsed)

    result = provider.analyze_article("Title", "Some article text", "arxiv")

    assert isinstance(result, ArticleAnalysis)
    assert result.summary == "A summary"
    assert result.key_points == ["point 1", "point 2"]
    assert result.implication == "An implication"
    assert result.category == "Interpretability"


def test_analyze_article_calls_parse_with_correct_args():
    provider = _make_provider()

    parsed = AnalysisResponse(
        summary="s", key_points=["p"], implication="i", category="Other",
    )
    provider.client.beta.chat.completions.parse.return_value = _make_mock_response(parsed)

    with patch("align_data.llm.openai_provider.LLM_MODEL", "test-model"), \
         patch("align_data.llm.openai_provider.LLM_REASONING_EFFORT", "high"):
        provider.analyze_article("My Title", "My Text", "lesswrong")

    call_kwargs = provider.client.beta.chat.completions.parse.call_args[1]
    assert call_kwargs["model"] == "test-model"
    assert call_kwargs["response_format"] is AnalysisResponse
    assert call_kwargs["reasoning_effort"] == "high"

    messages = call_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "My Title" in messages[1]["content"]
    assert "My Text" in messages[1]["content"]
    assert "lesswrong" in messages[1]["content"]


def test_analyze_article_truncates_long_text():
    provider = _make_provider()

    parsed = AnalysisResponse(
        summary="s", key_points=["p"], implication="i", category="Other",
    )
    provider.client.beta.chat.completions.parse.return_value = _make_mock_response(parsed)

    long_text = "x" * (MAX_TEXT_CHARS + 1000)
    provider.analyze_article("Title", long_text, "source")

    user_content = provider.client.beta.chat.completions.parse.call_args[1]["messages"][1]["content"]
    # The user message includes the title/source header, so check the text portion is truncated
    assert "x" * MAX_TEXT_CHARS in user_content
    assert "x" * (MAX_TEXT_CHARS + 1) not in user_content


def test_analyze_article_handles_empty_text():
    provider = _make_provider()

    parsed = AnalysisResponse(
        summary="s", key_points=["p"], implication="i", category="Other",
    )
    provider.client.beta.chat.completions.parse.return_value = _make_mock_response(parsed)

    provider.analyze_article("Title", "", "source")

    user_content = provider.client.beta.chat.completions.parse.call_args[1]["messages"][1]["content"]
    assert "Article text:\n" in user_content
