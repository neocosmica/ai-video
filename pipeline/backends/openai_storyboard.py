"""分镜聊天：OpenAI 兼容的外部 API。

配置写成 ``api:<模型名>``，另存 ``llm_api_base`` / ``llm_api_key``
（可用环境变量 ``NOVEL_VIDEO_LLM_API_BASE`` / ``NOVEL_VIDEO_LLM_API_KEY`` 覆盖）。
请求走 ``/v1/chat/completions``，不加载本机权重。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

from pipeline.backends.catalog import StageModel

_CHAT_TIMEOUT = 180.0


def api_base_url(settings=None, raw: str | None = None) -> str:
    """优先环境变量，再 Settings / 传入值。去掉尾斜杠。"""
    env = os.environ.get("NOVEL_VIDEO_LLM_API_BASE", "").strip()
    if env:
        return env.rstrip("/")
    if raw is not None:
        return str(raw or "").strip().rstrip("/")
    if settings is not None:
        return str(getattr(settings, "llm_api_base", "") or "").strip().rstrip("/")
    return ""


def api_key(settings=None, raw: str | None = None) -> str:
    env = os.environ.get("NOVEL_VIDEO_LLM_API_KEY", "").strip()
    if env:
        return env
    if raw is not None:
        return str(raw or "").strip()
    if settings is not None:
        return str(getattr(settings, "llm_api_key", "") or "").strip()
    return ""


def chat_completions_url(base: str) -> str:
    """把用户填的 Base URL 收成 chat/completions 地址。"""
    text = (base or "").strip().rstrip("/")
    if not text:
        raise SystemExit("外部 API 的 Base URL 为空。请填写，例如 https://api.deepseek.com/v1")
    lowered = text.lower()
    if lowered.endswith("/chat/completions"):
        return text
    if lowered.endswith("/v1"):
        return text + "/chat/completions"
    return text + "/v1/chat/completions"


def validate_api_base(base: str) -> str:
    text = (base or "").strip().rstrip("/")
    if not text:
        raise SystemExit("请填写外部 API 的 Base URL，例如 https://api.deepseek.com/v1")
    parsed = urlparse(text if "://" in text else "https://" + text)
    if parsed.scheme not in {"http", "https"}:
        raise SystemExit(f"外部 API Base URL 只支持 http/https：{base!r}")
    if not parsed.netloc:
        raise SystemExit(f"无效的外部 API Base URL：{base!r}")
    # 规范化：用户没写协议时补 https
    if "://" not in text:
        text = "https://" + text
    return text.rstrip("/")


def credentials_ready(settings=None, *, base: str | None = None, key: str | None = None) -> tuple[bool, str]:
    """网页 / doctor：有没有填齐地址和密钥。不发网络请求。"""
    host = api_base_url(settings, base)
    secret = api_key(settings, key)
    if not host:
        return False, "还没填外部 API 的 Base URL。"
    if not secret:
        return False, "还没填外部 API 的 API Key。"
    try:
        validate_api_base(host)
    except SystemExit as exc:
        return False, str(exc)
    return True, ""


def status_line(name: str, settings) -> str:
    host = api_base_url(settings)
    ok, detail = credentials_ready(settings)
    state = "已配置" if ok else (detail or "未配置")
    where = host or "（未填 Base URL）"
    return f"分镜 LLM：外部 API {name} @ {where}（{state}）"


def _content_from_message(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return ""


def parse_chat_completion(payload: Any) -> str:
    """从非流式 chat/completions 响应取出助手正文。"""
    if not isinstance(payload, dict):
        raise SystemExit("外部 API 返回的不是 JSON 对象")
    if payload.get("error"):
        err = payload["error"]
        if isinstance(err, dict):
            message = err.get("message") or err.get("code") or err
        else:
            message = err
        raise SystemExit(f"外部 API 生成失败：{message}")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise SystemExit("外部 API 没有返回 choices")
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    text = _content_from_message(message).strip()
    if not text:
        # 少数兼容端把正文放在 text
        text = str(first.get("text") or "").strip()
    if not text:
        raise SystemExit("外部 API 没有返回正文")
    return text


class OpenAIStoryboardLLM:
    def __init__(
        self,
        spec: StageModel,
        settings,
        *,
        reasoning_effort: str = "low",
        think_max_tokens: int = 1024,
    ):
        if spec.runtime != "openai":
            raise SystemExit(f"分镜运行时不是 openai：{spec.runtime}")
        self.spec = spec
        self.name = spec.repo
        self.base = validate_api_base(api_base_url(settings))
        self.key = api_key(settings)
        if not self.key:
            raise SystemExit(
                "外部 API 的 API Key 为空。请在「各阶段模型」里填写，"
                "或设置环境变量 NOVEL_VIDEO_LLM_API_KEY。"
            )
        self.url = chat_completions_url(self.base)
        effort = str(reasoning_effort or "low").strip().lower()
        self.reasoning_effort = effort if effort in {"low", "medium", "high"} else "low"
        self.think_max_tokens = max(64, int(think_max_tokens or 1024))
        print(
            f"加载分镜 LLM：{spec.label}（外部 API {self.name} @ {self.base}，"
            f"effort={self.reasoning_effort}, budget={self.think_max_tokens}）",
            flush=True,
        )

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        think_max_tokens: int | None = None,
    ) -> str:
        budget = (
            max(64, int(think_max_tokens))
            if think_max_tokens is not None
            else self.think_max_tokens
        )
        payload = {
            "model": self.name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            # Qwen 系兼容字段（不认的服务端通常忽略）
            "enable_thinking": True,
            "thinking_budget": budget,
            "reasoning_effort": self.reasoning_effort,
        }
        print(
            f"外部 API 生成中（effort={self.reasoning_effort}, budget={budget}）…",
            flush=True,
        )
        try:
            raw = self._post_chat(payload)
        except urllib.error.HTTPError as exc:
            raise SystemExit(f"外部 API 生成失败（{exc.code}）：{_http_error_message(exc)}") from exc
        except urllib.error.URLError as exc:
            raise SystemExit(f"连不上外部 API（{self.base}）：{exc.reason}") from exc
        except TimeoutError as exc:
            raise SystemExit(f"外部 API 响应超时（{self.base}）") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"外部 API 返回的不是 JSON：{raw[:200]}") from exc
        return parse_chat_completion(data)

    def _post_chat(self, payload: dict) -> str:
        """不认 thinking 字段时剥掉再试。"""
        try:
            return _post_json(self.url, payload, self.key, _CHAT_TIMEOUT)
        except urllib.error.HTTPError as exc:
            if exc.code != 400:
                raise
            message = _http_error_message(exc).lower()
            think_keys = ("enable_thinking", "thinking_budget", "reasoning_effort")
            if not any(k in payload for k in think_keys):
                raise
            if not any(
                token in message
                for token in (
                    "thinking",
                    "reasoning",
                    "enable_thinking",
                    "thinking_budget",
                    "reasoning_effort",
                    "unknown",
                    "invalid",
                    "extra",
                )
            ):
                raise
            retry = {key: value for key, value in payload.items() if key not in think_keys}
            return _post_json(self.url, retry, self.key, _CHAT_TIMEOUT)

    def close(self) -> None:
        return None


def _post_json(url: str, payload: dict, key: str, timeout: float) -> str:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def _http_error_message(exc: urllib.error.HTTPError) -> str:
    body = exc.read().decode("utf-8", "replace")
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return body or str(exc)
    if isinstance(obj, dict):
        err = obj.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err.get("code") or err)
        if err:
            return str(err)
        return str(obj.get("message") or body or exc)
    return body or str(exc)


def selftest_openai_storyboard() -> None:
    """不连真的外部 API：URL 规范化与响应解析。"""
    assert chat_completions_url("https://api.deepseek.com/v1") == (
        "https://api.deepseek.com/v1/chat/completions"
    )
    assert chat_completions_url("https://api.openai.com/v1/") == (
        "https://api.openai.com/v1/chat/completions"
    )
    assert chat_completions_url("https://example.com/v1/chat/completions") == (
        "https://example.com/v1/chat/completions"
    )
    assert chat_completions_url("https://api.siliconflow.cn") == (
        "https://api.siliconflow.cn/v1/chat/completions"
    )
    assert validate_api_base("api.deepseek.com/v1") == "https://api.deepseek.com/v1"
    try:
        validate_api_base("")
        raise AssertionError("空 Base URL 应失败")
    except SystemExit as exc:
        assert "Base URL" in str(exc)

    text = parse_chat_completion(
        {
            "choices": [
                {"message": {"role": "assistant", "content": '{"shots":[]}'}},
            ]
        }
    )
    assert text == '{"shots":[]}', text
    parts = parse_chat_completion(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": '{"ok":'},
                            {"type": "text", "text": "true}"},
                        ],
                    }
                }
            ]
        }
    )
    assert parts == '{"ok":true}', parts
    try:
        parse_chat_completion({"error": {"message": "invalid key"}})
        raise AssertionError("error 应失败")
    except SystemExit as exc:
        assert "invalid key" in str(exc)

    ok, _detail = credentials_ready(base="https://api.deepseek.com/v1", key="sk-test")
    assert ok
    bad, detail = credentials_ready(base="", key="sk-test")
    assert not bad and "Base URL" in detail
    missing, detail2 = credentials_ready(base="https://api.deepseek.com/v1", key="")
    assert not missing and "API Key" in detail2

    from pipeline.backends.catalog import make_api_model

    model = make_api_model("deepseek-chat")
    assert model.id == "api:deepseek-chat" and model.runtime == "openai"
