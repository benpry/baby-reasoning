import pytest
import responses as resp
import json
from baby_reasoning.model import OllamaBackend
from baby_reasoning.tasks.base import ModelResponse


@pytest.fixture
def backend():
    return OllamaBackend(model="test-model", base_url="http://localhost:11434")


@resp.activate
def test_generate_returns_text(backend):
    resp.add(
        resp.POST,
        "http://localhost:11434/api/generate",
        json={"response": "ro", "done": True},
        status=200,
    )
    result = backend.generate("some prompt")
    assert isinstance(result, ModelResponse)
    assert result.text == "ro"


@resp.activate
def test_generate_strips_trailing_whitespace(backend):
    resp.add(
        resp.POST,
        "http://localhost:11434/api/generate",
        json={"response": "ro\n", "done": True},
        status=200,
    )
    result = backend.generate("some prompt")
    assert result.text == "ro"


@resp.activate
def test_generate_returns_none_logprobs_when_absent(backend):
    resp.add(
        resp.POST,
        "http://localhost:11434/api/generate",
        json={"response": "ro", "done": True},
        status=200,
    )
    result = backend.generate("some prompt")
    assert result.token_logprobs is None


@resp.activate
def test_generate_returns_logprobs_when_present(backend):
    resp.add(
        resp.POST,
        "http://localhost:11434/api/generate",
        json={
            "response": "ro",
            "done": True,
            "logprobs": {"token_logprobs": [-1.2, -0.5]},
        },
        status=200,
    )
    result = backend.generate("some prompt")
    assert result.token_logprobs == [-1.2, -0.5]


@resp.activate
def test_score_completion_returns_none_when_logprobs_absent(backend):
    resp.add(
        resp.POST,
        "http://localhost:11434/api/generate",
        json={"response": "", "done": True},
        status=200,
    )
    result = backend.score_completion("prompt", "ro")
    assert result is None


@resp.activate
def test_score_completion_sums_logprobs(backend):
    resp.add(
        resp.POST,
        "http://localhost:11434/api/generate",
        json={
            "response": "",
            "done": True,
            "logprobs": {"token_logprobs": [-1.0, -2.0, -0.5]},
        },
        status=200,
    )
    result = backend.score_completion("prompt ", "ro fe")
    assert result == pytest.approx(-3.5)


@resp.activate
def test_model_name_sent_in_request(backend):
    resp.add(
        resp.POST,
        "http://localhost:11434/api/generate",
        json={"response": "x", "done": True},
        status=200,
    )
    backend.generate("hi")
    request_body = json.loads(resp.calls[0].request.body)
    assert request_body["model"] == "test-model"
