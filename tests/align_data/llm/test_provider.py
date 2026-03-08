from unittest.mock import patch, MagicMock

import pytest

from align_data.llm.provider import ArticleAnalysis, create_llm_provider


def test_create_llm_provider_openai():
    with patch("align_data.llm.provider.LLM_PROVIDER", "openai"), \
         patch("align_data.llm.openai_provider.openai_client", MagicMock()):
        provider = create_llm_provider()

    from align_data.llm.openai_provider import OpenAIProvider
    assert isinstance(provider, OpenAIProvider)


def test_create_llm_provider_unknown():
    with patch("align_data.llm.provider.LLM_PROVIDER", "unknown"):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_provider()


def test_article_analysis_fields():
    analysis = ArticleAnalysis(
        summary="test summary",
        key_points=["point1", "point2"],
        implication="test implication",
        category="Other",
    )
    assert analysis.summary == "test summary"
    assert analysis.key_points == ["point1", "point2"]
    assert analysis.implication == "test implication"
    assert analysis.category == "Other"
