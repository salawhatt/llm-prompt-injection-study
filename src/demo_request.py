"""One optional demonstration request. Importing this module never calls the API."""
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

ENDPOINT = "https://foundation-models.api.cloud.ru/v1/chat/completions"
MODELS = ("anthropic/claude-sonnet-5.5", "openai/gpt-6-astra")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the Authorization header to a redirected endpoint.
        return None


def urlopen(request, timeout):
    return build_opener(_NoRedirect()).open(request, timeout=timeout)


def _validate(payload):
    if not isinstance(payload, dict) or set(payload) != {"model", "messages", "temperature", "max_tokens"}:
        raise ValueError("Нужны только model, messages, temperature и max_tokens.")
    if payload["model"] not in MODELS:
        raise ValueError("Выберите Claude Sonnet 5.5 или GPT-6 Astra.")
    if payload["temperature"] != 0:
        raise ValueError("В демонстрации используется temperature=0.")
    if type(payload["max_tokens"]) is not int or not 1 <= payload["max_tokens"] <= 1024:
        raise ValueError("max_tokens должен быть целым числом от 1 до 1024.")
    messages = payload["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages должен содержать сообщения.")
    for message in messages:
        if (not isinstance(message, dict) or set(message) != {"role", "content"}
                or message["role"] not in ("system", "user")
                or not isinstance(message["content"], str) or not message["content"].strip()):
            raise ValueError("Допустимы непустые сообщения system и user.")
    if not any(message["role"] == "user" for message in messages):
        raise ValueError("Нужен вопрос пользователя.")
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")


def _cached(path):
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") == "answered":
            return dict(record, cached=True)
    except (OSError, ValueError, AttributeError):
        record = {"request_id": path.stem}
    return dict(record, status="review_required", cached=True,
                error="Попытка уже зарегистрирована, но ответ не подтверждён. "
                      "Автоповтор отключён: проверьте запись и начисления Cloud.ru.")


def request_once(payload: dict, api_key: str, output_dir: Path) -> dict:
    """Send at most once per payload within output_dir; reuse a saved answer.

    output_dir is the live_requests directory, separate from research responses.
    A pending/error record also prevents repetition because the service may have
    charged for a request whose response was lost. No retry or force flag exists.
    """
    body = _validate(payload)
    if not isinstance(api_key, str) or not api_key.strip() or "\n" in api_key or "\r" in api_key:
        raise ValueError("Задайте API-ключ в секретах среды, не в тексте запроса.")
    if api_key in body.decode("utf-8"):
        raise ValueError("API-ключ не должен присутствовать в содержимом запроса.")
    request_id = hashlib.sha256(ENDPOINT.encode() + b"\n" + body).hexdigest()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (request_id + ".json")
    record = dict(request_id=request_id, payload=json.loads(body), status="pending",
                  answer=None, usage=None, returned_model=None, seconds=None, finish=None)
    try:
        # Exclusive creation reserves this payload before any network operation.
        with path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        return _cached(path)

    started = time.monotonic()
    try:
        request = Request(ENDPOINT, data=body, method="POST",
                          headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key})
        with urlopen(request, timeout=90) as response:
            data = json.loads(response.read())
        choice = data["choices"][0]
        answer = choice["message"].get("content")
        if not isinstance(answer, str) or not isinstance(data.get("usage"), dict):
            raise ValueError("Incomplete response")
        record.update(status="answered", answer=answer, usage=data["usage"],
                      returned_model=data.get("model"), finish=choice.get("finish_reason"))
    except HTTPError as error:
        record.update(status="review_required", error=f"HTTP {error.code}; автоматический повтор отключён.")
        error.close()
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError):
        # Do not serialize exception text, response bodies, headers or credentials.
        record.update(status="review_required", error="Ответ не подтверждён: ошибка соединения или формата. "
                      "Автоматический повтор отключён; проверьте начисления Cloud.ru.")
    record["seconds"] = round(time.monotonic() - started, 3)
    encoded = json.dumps(record, ensure_ascii=False, indent=2)
    # Additional protection if a service response unexpectedly contains the key.
    encoded = encoded.replace(api_key, "[REDACTED]")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)
    return dict(json.loads(encoded), cached=False)
