import pytest
import responses as resp
import json
from baby_reasoning.model import VLLMBackend
from baby_reasoning.tasks.base import ModelResponse


VLLM_URL = "http://localhost:8000/v1/completions"


@pytest.fixture
def backend():
    return VLLMBackend(model="test-model", base_url="http://localhost:8000")


def _completions_response(text="", logprobs=None):
    """Build a minimal OpenAI-compatible completions response."""
    choice = {"text": text, "logprobs": logprobs, "finish_reason": "stop"}
    return {"choices": [choice]}


@resp.activate
def test_generate_returns_text(backend):
    resp.add(resp.POST, VLLM_URL, json=_completions_response(text="ro"), status=200)
    result = backend.generate("some prompt")
    assert isinstance(result, ModelResponse)
    assert result.text == "ro"


@resp.activate
def test_generate_strips_trailing_whitespace(backend):
    resp.add(resp.POST, VLLM_URL, json=_completions_response(text="ro\n"), status=200)
    result = backend.generate("some prompt")
    assert result.text == "ro"


@resp.activate
def test_generate_returns_none_logprobs_when_absent(backend):
    resp.add(resp.POST, VLLM_URL, json=_completions_response(text="ro"), status=200)
    result = backend.generate("some prompt")
    assert result.token_logprobs is None


@resp.activate
def test_generate_returns_logprobs_when_present(backend):
    resp.add(
        resp.POST,
        VLLM_URL,
        json=_completions_response(
            text="ro", logprobs={"token_logprobs": [-1.2, -0.5]}
        ),
        status=200,
    )
    result = backend.generate("some prompt")
    assert result.token_logprobs == [-1.2, -0.5]


@resp.activate
def test_score_completion_returns_none_when_logprobs_absent(backend):
    resp.add(resp.POST, VLLM_URL, json=_completions_response(), status=200)
    result = backend.score_completion("prompt", "ro")
    assert result is None


@resp.activate
def test_score_completion_returns_none_when_text_offset_absent(backend):
    """If vLLM omits text_offset, return None rather than wrong total."""
    resp.add(
        resp.POST,
        VLLM_URL,
        json=_completions_response(
            logprobs={"token_logprobs": [-1.0, -2.0], "text_offset": None}
        ),
        status=200,
    )
    result = backend.score_completion("prompt", " ro")
    assert result is None


@resp.activate
def test_score_completion_excludes_prompt_logprobs(backend):
    """Only completion tokens (text_offset >= len(prompt)) are summed."""
    # prompt = "hello " (6 chars), completion = "world" (5 chars)
    # Token offsets: 0, 3, 6, 9 → first two are prompt, last two are completion
    resp.add(
        resp.POST,
        VLLM_URL,
        json=_completions_response(
            logprobs={
                "token_logprobs": [None, -5.0, -1.0, -2.0],
                "text_offset": [0, 3, 6, 9],
            }
        ),
        status=200,
    )
    result = backend.score_completion("hello ", "world")
    assert result == pytest.approx(-3.0)


@resp.activate
def test_score_completion_sums_completion_logprobs(backend):
    # prompt = "prompt " (7 chars), completion = "ro fe" (5 chars)
    # Tokens at offsets 0, 4, 7, 10 → last two are completion
    resp.add(
        resp.POST,
        VLLM_URL,
        json=_completions_response(
            logprobs={
                "token_logprobs": [None, -9.0, -1.0, -2.0],
                "text_offset": [0, 4, 7, 10],
            }
        ),
        status=200,
    )
    result = backend.score_completion("prompt ", "ro fe")
    assert result == pytest.approx(-3.0)


@resp.activate
def test_score_completion_filters_none_in_completion_logprobs(backend):
    # prompt = "p" (1 char), completion = " ab" (3 chars)
    # Completion tokens at offsets 1, 2 — one has None logprob
    resp.add(
        resp.POST,
        VLLM_URL,
        json=_completions_response(
            logprobs={
                "token_logprobs": [None, None, -2.0],
                "text_offset": [0, 1, 2],
            }
        ),
        status=200,
    )
    result = backend.score_completion("p", " ab")
    assert result == pytest.approx(-2.0)


@resp.activate
def test_score_completion_returns_zero_for_zero_logprobs(backend):
    # prompt = "prompt" (6 chars), completion = "ro" (2 chars)
    resp.add(
        resp.POST,
        VLLM_URL,
        json=_completions_response(
            logprobs={
                "token_logprobs": [None, -3.0, 0.0, 0.0],
                "text_offset": [0, 3, 6, 7],
            }
        ),
        status=200,
    )
    result = backend.score_completion("prompt", "ro")
    assert result == pytest.approx(0.0)


@resp.activate
def test_model_name_sent_in_request(backend):
    resp.add(resp.POST, VLLM_URL, json=_completions_response(text="x"), status=200)
    backend.generate("hi")
    request_body = json.loads(resp.calls[0].request.body)
    assert request_body["model"] == "test-model"
