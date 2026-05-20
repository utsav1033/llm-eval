from llm_eval import ExactMatch
import pytest

async def test_exact_match():
    scorer = ExactMatch()
    result = await scorer.score("hello", "hello")
    assert result == 1.0
    
async def test_exact_match_fails():
    scorer = ExactMatch()
    result = await scorer.score("hello", "world")
    assert result == 0.0

async def test_exact_match_missing_expected():
    scorer = ExactMatch()
    with pytest.raises(ValueError):
        await scorer.score("hello", None)
