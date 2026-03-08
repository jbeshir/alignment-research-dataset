import json
from unittest.mock import patch, MagicMock, call

import pytest

from align_data.llm.provider import AnalysisResult, ArticleAnalysis, TokenUsage
from align_data.llm.summarize import ArticleSummarizer, MIN_TEXT_LENGTH


def _make_article(_id=1, hash_id="abc", title="Title", text="x" * 300, source="arxiv"):
    article = MagicMock()
    article._id = _id
    article.id = hash_id
    article.title = title
    article.text = text
    article.source = source
    return article


def _make_result(summary="s", key_points=None, implication="i", category="Other",
                 prompt_tokens=0, completion_tokens=0, total_tokens=0, model=""):
    return AnalysisResult(
        analysis=ArticleAnalysis(
            summary=summary,
            key_points=key_points or ["p"],
            implication=implication,
            category=category,
        ),
        usage=TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=model,
        ),
    )


@pytest.fixture
def summarizer():
    with patch("align_data.llm.summarize.create_llm_provider") as mock_factory:
        mock_provider = MagicMock()
        mock_factory.return_value = mock_provider
        s = ArticleSummarizer()
    return s


def test_process_batch_calls_provider_and_updates_db(summarizer):
    article = _make_article()
    summarizer.provider.analyze_article.return_value = _make_result(
        summary="A summary", key_points=["p1", "p2"],
        implication="An implication", category="Interpretability",
    )

    session = MagicMock()
    usage = TokenUsage()
    summarizer._process_batch(session, [article], usage)

    summarizer.provider.analyze_article.assert_called_once_with(
        title="Title", text="x" * 300, source="arxiv",
    )
    session.execute.assert_called_once()
    session.commit.assert_called_once()


def test_process_batch_skips_short_text(summarizer):
    short_article = _make_article(text="short")
    assert len("short".strip()) < MIN_TEXT_LENGTH

    session = MagicMock()
    usage = TokenUsage()
    summarizer._process_batch(session, [short_article], usage)

    summarizer.provider.analyze_article.assert_not_called()
    session.execute.assert_not_called()
    session.commit.assert_called_once()


def test_process_batch_skips_empty_text(summarizer):
    article = _make_article(text="")
    session = MagicMock()
    usage = TokenUsage()
    summarizer._process_batch(session, [article], usage)

    summarizer.provider.analyze_article.assert_not_called()


def test_process_batch_skips_none_text(summarizer):
    article = _make_article(text=None)
    session = MagicMock()
    usage = TokenUsage()
    summarizer._process_batch(session, [article], usage)

    summarizer.provider.analyze_article.assert_not_called()


def test_process_batch_continues_after_per_article_error(summarizer):
    article1 = _make_article(_id=1, hash_id="a1")
    article2 = _make_article(_id=2, hash_id="a2")

    summarizer.provider.analyze_article.side_effect = [
        Exception("LLM error"),
        _make_result(),
    ]

    session = MagicMock()
    usage = TokenUsage()
    summarizer._process_batch(session, [article1, article2], usage)

    # Second article should still be processed
    assert summarizer.provider.analyze_article.call_count == 2
    session.execute.assert_called_once()
    session.commit.assert_called_once()


def test_process_batch_rolls_back_on_commit_failure(summarizer):
    article = _make_article()
    summarizer.provider.analyze_article.return_value = _make_result()

    session = MagicMock()
    session.commit.side_effect = Exception("DB error")

    usage = TokenUsage()
    summarizer._process_batch(session, [article], usage)

    session.rollback.assert_called_once()


def test_process_batch_stores_key_points_as_json(summarizer):
    article = _make_article()
    summarizer.provider.analyze_article.return_value = _make_result(
        key_points=["point 1", "point 2"],
    )

    session = MagicMock()
    usage = TokenUsage()
    summarizer._process_batch(session, [article], usage)

    session.execute.assert_called_once()


def test_process_batch_uses_empty_string_for_none_fields(summarizer):
    article = _make_article(title=None, source=None)
    summarizer.provider.analyze_article.return_value = _make_result()

    session = MagicMock()
    usage = TokenUsage()
    summarizer._process_batch(session, [article], usage)

    summarizer.provider.analyze_article.assert_called_once_with(
        title="", text="x" * 300, source="",
    )


def test_process_batch_accumulates_usage(summarizer):
    article1 = _make_article(_id=1, hash_id="a1")
    article2 = _make_article(_id=2, hash_id="a2")

    summarizer.provider.analyze_article.side_effect = [
        _make_result(prompt_tokens=100, completion_tokens=50, total_tokens=150, model="gpt-5-nano"),
        _make_result(prompt_tokens=200, completion_tokens=80, total_tokens=280, model="gpt-5-nano"),
    ]

    session = MagicMock()
    usage = TokenUsage()
    summarizer._process_batch(session, [article1, article2], usage)

    assert usage.prompt_tokens == 300
    assert usage.completion_tokens == 130
    assert usage.total_tokens == 430
    assert usage.model == "gpt-5-nano"


def test_process_batch_uses_thread_pool(summarizer):
    from concurrent.futures import ThreadPoolExecutor

    articles = [_make_article(_id=i, hash_id=f"a{i}") for i in range(3)]
    summarizer.provider.analyze_article.return_value = _make_result()

    session = MagicMock()
    usage = TokenUsage()
    with patch("align_data.llm.summarize.LLM_CONCURRENCY", 4), \
         patch("align_data.llm.summarize.ThreadPoolExecutor", wraps=ThreadPoolExecutor) as mock_pool:
        summarizer._process_batch(session, articles, usage)

    mock_pool.assert_called_once_with(max_workers=4)
    assert summarizer.provider.analyze_article.call_count == 3


def test_process_articles_logs_total_usage(summarizer, caplog):
    import logging

    article = _make_article()
    summarizer.provider.analyze_article.return_value = _make_result(
        prompt_tokens=500, completion_tokens=200, total_tokens=700, model="gpt-5-nano",
    )

    query = MagicMock()
    query.count.return_value = 1
    query.__iter__ = MagicMock(return_value=iter([article]))

    session = MagicMock()
    with caplog.at_level(logging.INFO, logger="align_data.llm.summarize"):
        summarizer._process_articles(session, query, log_progress=True)

    assert "Total token usage for gpt-5-nano: 500 prompt, 200 completion, 700 total" in caplog.text
