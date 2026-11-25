import json
from pathlib import Path
import sys
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import demo_request


@pytest.fixture
def payload():
    return dict(model="openai/gpt-6-astra", temperature=0, max_tokens=64,
                messages=[dict(role="system", content="Ответь по документу."),
                          dict(role="user", content="Сколько дней в неделе?")])


def success():
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.read.return_value = json.dumps(dict(
        model="openai/gpt-6-astra", usage={"prompt_tokens": 20, "completion_tokens": 3},
        choices=[{"message": {"content": "7"}, "finish_reason": "stop"}])).encode()
    return response


def test_success_and_cache_without_second_request(tmp_path, payload, monkeypatch):
    opener = Mock(return_value=success())
    monkeypatch.setattr(demo_request, "urlopen", opener)
    result = demo_request.request_once(payload, "test-secret-key", tmp_path)
    cached = demo_request.request_once(payload, "test-secret-key", tmp_path)
    assert result["status"] == cached["status"] == "answered"
    assert result["answer"] == "7" and result["cached"] is False and cached["cached"] is True
    assert cached["usage"]["completion_tokens"] == 3
    opener.assert_called_once()
    request = opener.call_args.args[0]
    assert request.full_url == demo_request.ENDPOINT
    assert json.loads(request.data) == payload
    saved = next(tmp_path.glob("*.json")).read_text(encoding="utf-8")
    assert "test-secret-key" not in saved and "Authorization" not in saved
    assert json.loads(saved)["payload"] == payload


@pytest.mark.parametrize("failure", [URLError("secret-key-from-exception"),
                                    HTTPError(demo_request.ENDPOINT, 503, "secret-key-from-exception", {}, None)])
def test_network_error_never_retries(tmp_path, payload, monkeypatch, failure):
    opener = Mock(side_effect=failure)
    monkeypatch.setattr(demo_request, "urlopen", opener)
    result = demo_request.request_once(payload, "test-secret-key", tmp_path)
    path = next(tmp_path.glob("*.json"))
    before = path.read_bytes()
    cached = demo_request.request_once(payload, "test-secret-key", tmp_path)
    assert result["status"] == cached["status"] == "review_required"
    assert path.read_bytes() == before
    assert b"secret-key-from-exception" not in before
    opener.assert_called_once()


def test_bad_response_is_preserved_without_retry(tmp_path, payload, monkeypatch):
    response = success()
    response.read.return_value = b"not-json"
    opener = Mock(return_value=response)
    monkeypatch.setattr(demo_request, "urlopen", opener)
    assert demo_request.request_once(payload, "test-secret-key", tmp_path)["status"] == "review_required"
    assert demo_request.request_once(payload, "test-secret-key", tmp_path)["status"] == "review_required"
    opener.assert_called_once()


def test_pending_attempt_is_not_repeated(tmp_path, payload, monkeypatch):
    opener = Mock(side_effect=KeyboardInterrupt)
    monkeypatch.setattr(demo_request, "urlopen", opener)
    with pytest.raises(KeyboardInterrupt):
        demo_request.request_once(payload, "test-secret-key", tmp_path)
    path = next(tmp_path.glob("*.json"))
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "pending"
    assert demo_request.request_once(payload, "test-secret-key", tmp_path)["status"] == "review_required"
    opener.assert_called_once()


@pytest.mark.parametrize("change", [dict(model="other-model"), dict(max_tokens=1025),
                                    dict(max_tokens=True), dict(temperature=0.5), dict(api_key="key"),
                                    dict(messages=[dict(role="tool", content="text")])])
def test_invalid_request_never_calls_network(tmp_path, payload, monkeypatch, change):
    opener = Mock()
    monkeypatch.setattr(demo_request, "urlopen", opener)
    with pytest.raises(ValueError):
        demo_request.request_once(payload | change, "test-secret-key", tmp_path)
    opener.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_key_cannot_be_saved_in_payload(tmp_path, payload, monkeypatch):
    opener = Mock()
    monkeypatch.setattr(demo_request, "urlopen", opener)
    payload["messages"][1]["content"] = "Do not save test-secret-key"
    with pytest.raises(ValueError):
        demo_request.request_once(payload, "test-secret-key", tmp_path)
    assert not list(tmp_path.iterdir())
    opener.assert_not_called()


def test_redirects_are_disabled():
    assert demo_request._NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.com") is None
