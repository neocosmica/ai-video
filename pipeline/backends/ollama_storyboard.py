"""分镜聊天：本机 Ollama。

每次打开网页列模型时跑一次 ``ollama list``，不读硬盘上的模型目录。
生成时把聊天请求发到 Ollama，不加载 mlx-lm，也不下载 Hugging Face 权重。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable

from pipeline.backends.catalog import StageModel, make_ollama_model

_LIST_TIMEOUT = 15.0
_CHAT_TIMEOUT = 180.0
_SIZE_UNITS = {"B", "KB", "MB", "GB", "TB"}


@dataclass(frozen=True)
class LocalOllamaModel:
    name: str
    size: int = 0
    size_text: str = ""
    parameter_size: str = ""
    quantization: str = ""


@dataclass(frozen=True)
class OllamaScan:
    host: str
    online: bool
    models: tuple[LocalOllamaModel, ...]
    detail: str = ""


def ollama_base_url() -> str:
    """``OLLAMA_HOST`` 与 Ollama 自己的约定一致，缺省本机 11434。"""
    raw = os.environ.get("OLLAMA_HOST", "").strip() or "127.0.0.1:11434"
    if "://" not in raw:
        raw = "http://" + raw
    return raw.rstrip("/")


def name_aliases(name: str) -> set[str]:
    """``llama3`` 与 ``llama3:latest`` 视为同一个已下载模型。"""
    text = name.strip()
    found = {text}
    tail = text.rsplit("/", 1)[-1]
    if tail.endswith(":latest"):
        found.add(text[: -len(":latest")])
    elif ":" not in tail:
        found.add(f"{text}:latest")
    return found


def names_match(wanted: str, have: str) -> bool:
    return bool(name_aliases(wanted) & name_aliases(have))


def _format_size(size: int) -> str:
    if size >= 1024**3:
        return f"{size / 1024**3:.1f} GB"
    if size >= 1024**2:
        return f"{size / 1024**2:.0f} MB"
    return ""


def parse_ollama_list(text: str) -> tuple[LocalOllamaModel, ...]:
    """解析 ``ollama list`` 的表格。不读模型目录。"""
    found: list[LocalOllamaModel] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if not parts or parts[0].upper() == "NAME":
            continue
        name = parts[0]
        size_text = ""
        if len(parts) >= 4 and parts[3].upper() in _SIZE_UNITS:
            size_text = f"{parts[2]} {parts[3].upper()}"
        if name in seen:
            continue
        seen.add(name)
        found.append(LocalOllamaModel(name=name, size_text=size_text))
    found.sort(key=lambda item: item.name)
    return tuple(found)


def scan_local_models() -> OllamaScan:
    """跑一次 ``ollama list``。每次调用都重新查，不缓存，也不扫硬盘。"""
    host = ollama_base_url()
    binary = shutil.which("ollama")
    if not binary:
        return OllamaScan(
            host=host,
            online=False,
            models=(),
            detail="找不到 ollama 命令。请先安装 Ollama，并保证终端里能运行 ollama list。",
        )
    try:
        proc = subprocess.run(
            [binary, "list"],
            capture_output=True,
            text=True,
            timeout=_LIST_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return OllamaScan(
            host=host,
            online=False,
            models=(),
            detail=f"ollama list 超时（{host}）。请确认 Ollama 已打开。",
        )
    except OSError as exc:
        return OllamaScan(host=host, online=False, models=(), detail=f"无法执行 ollama list：{exc}")
    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout or "").strip()
        return OllamaScan(host=host, online=False, models=(), detail=_cli_failure_detail(host, message))
    return OllamaScan(host=host, online=True, models=parse_ollama_list(proc.stdout), detail="")


def _cli_failure_detail(host: str, message: str) -> str:
    text = message.strip()
    lowered = text.lower()
    if "connect" in lowered or "connection" in lowered:
        return f"Ollama 没在运行（{host}）。请先打开 Ollama，或执行 ollama serve。"
    if text:
        return f"ollama list 失败：{text}"
    return f"ollama list 失败（{host}）。"


def _choice_label(item: LocalOllamaModel) -> str:
    bits = [f"Ollama · {item.name}"]
    if item.parameter_size:
        bits.append(item.parameter_size)
    if item.quantization:
        bits.append(item.quantization)
    size = item.size_text or _format_size(item.size)
    if size:
        bits.append(size)
    return " · ".join(bits)


def _choice_note(item: LocalOllamaModel, scan: OllamaScan) -> str:
    size = _format_size(item.size)
    where = f"约 {size}，" if size else ""
    note = f"本机已下载，{where}分镜推理走 Ollama（{scan.host}），不下载 Hugging Face 权重。"
    if not scan.online:
        note += " 当前连不上 Ollama，生成前请先打开它。"
    return note


def _storyboard_usable(name: str) -> bool:
    """嵌入模型不能写分镜，不放进下拉列表。手写 id 仍然可以解析。"""
    tail = name.rsplit("/", 1)[-1].lower()
    return "embed" not in tail


def storyboard_choices(scan: OllamaScan | None = None) -> tuple[StageModel, ...]:
    found = scan if scan is not None else scan_local_models()
    return tuple(
        make_ollama_model(item.name, label=_choice_label(item), note=_choice_note(item, found))
        for item in found.models
        if _storyboard_usable(item.name)
    )


def model_is_local(name: str, scan: OllamaScan | None = None) -> bool:
    found = scan if scan is not None else scan_local_models()
    return any(names_match(name, item.name) for item in found.models)


def status_line(name: str) -> str:
    """doctor 用的一行说明。"""
    scan = scan_local_models()
    present = model_is_local(name, scan)
    if scan.online and present:
        state = "服务在线，模型在本机"
    elif scan.online:
        state = "服务在线，但没有这个模型（可先 ollama pull）"
    elif present:
        state = "模型已下载，但服务没在运行。请先打开 Ollama 或执行 ollama serve"
    else:
        state = scan.detail or "服务未响应，本机也没有这个模型"
    return f"分镜 LLM：Ollama {name} @ {scan.host}（{state}）"


def collect_chat_stream(lines: Iterable[str], *, echo: bool = True) -> str:
    """把 Ollama 的 NDJSON 流收成正文。思考内容不进入返回值。"""
    parts: list[str] = []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Ollama 返回的不是 JSON：{text[:200]}") from exc
        if not isinstance(obj, dict):
            continue
        if obj.get("error"):
            raise SystemExit(f"Ollama 生成失败：{obj['error']}")
        message = obj.get("message") if isinstance(obj.get("message"), dict) else {}
        piece = str(message.get("content") or "")
        if piece:
            parts.append(piece)
            if echo:
                print(piece, end="", flush=True)
        if obj.get("done"):
            break
    if parts and echo:
        print(flush=True)
    return "".join(parts)


def _num_ctx(system: str, user: str, max_tokens: int) -> int:
    raw = os.environ.get("NOVEL_VIDEO_OLLAMA_NUM_CTX", "").strip()
    if raw.isdigit():
        return max(2048, int(raw))
    estimate = len(system) + len(user) + max_tokens + 512
    ctx = max(8192, estimate)
    ctx = (ctx + 255) // 256 * 256
    return min(ctx, 32768)


class OllamaStoryboardLLM:
    def __init__(self, spec: StageModel):
        if spec.runtime != "ollama":
            raise SystemExit(f"分镜运行时不是 ollama：{spec.runtime}")
        self.spec = spec
        self.name = spec.repo
        self.host = ollama_base_url()
        print(f"加载分镜 LLM：{spec.label}（Ollama {self.name} @ {self.host}）", flush=True)
        scan = scan_local_models()
        if not scan.online:
            raise SystemExit(
                f"Ollama 没在运行（{self.host}）。请先打开 Ollama，或在终端执行 ollama serve，再生成分镜。"
            )
        if not model_is_local(self.name, scan):
            names = "、".join(item.name for item in scan.models) or "无"
            raise SystemExit(
                f"Ollama 里没有模型 {self.name}。当前已下载：{names}。"
            )

    def complete(self, *, system: str, user: str, max_tokens: int, temperature: float) -> str:
        payload = {
            "model": self.name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": _num_ctx(system, user, max_tokens),
            },
        }
        print("Ollama 生成中…", flush=True)
        try:
            resp_ctx = self._open_chat(payload)
        except urllib.error.URLError as exc:
            raise SystemExit(
                f"连不上 Ollama（{self.host}）：{exc.reason}。请先打开 Ollama，再生成分镜。"
            ) from exc
        except TimeoutError as exc:
            raise SystemExit(
                f"Ollama 响应超时（{self.host}）。请确认服务还在运行，再生成分镜。"
            ) from exc
        with resp_ctx as resp:
            text = collect_chat_stream(_response_lines(resp))
        if not text.strip():
            raise SystemExit(f"Ollama 模型 {self.name} 没有返回正文。")
        return text

    def _open_chat(self, payload: dict):
        """旧版 Ollama 不认识 ``think`` 时，去掉该字段再试一次。"""
        try:
            return _post_json(f"{self.host}/api/chat", payload, _CHAT_TIMEOUT)
        except urllib.error.HTTPError as exc:
            message = _http_error_message(exc)
            if "think" in payload and exc.code == 400 and "think" in message.lower():
                retry = {key: value for key, value in payload.items() if key != "think"}
                try:
                    return _post_json(f"{self.host}/api/chat", retry, _CHAT_TIMEOUT)
                except urllib.error.HTTPError as again:
                    message = _http_error_message(again)
                    raise SystemExit(f"Ollama 生成失败（{again.code}）：{message}") from again
            raise SystemExit(f"Ollama 生成失败（{exc.code}）：{message}") from exc

    def close(self) -> None:
        return None


def _post_json(url: str, payload: dict, timeout: float):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def _http_error_message(exc: urllib.error.HTTPError) -> str:
    body = exc.read().decode("utf-8", "replace")
    try:
        message = json.loads(body).get("error") or body
    except json.JSONDecodeError:
        message = body or str(exc)
    return str(message)


def _response_lines(resp) -> Iterable[str]:
    while True:
        raw = resp.readline()
        if not raw:
            break
        yield raw.decode("utf-8", "replace")


def selftest_ollama_storyboard() -> None:
    """不连真的 Ollama：解析 ``ollama list`` 表格，以及聊天流拼接。"""
    assert names_match("llama3", "llama3:latest")
    assert names_match("qwen2.5:7b", "qwen2.5:7b")
    assert not names_match("qwen2.5:7b", "qwen2.5:14b")

    table = """
NAME                       ID              SIZE      MODIFIED
qwen2.5:7b                 abcdef123456    4.7 GB    2 days ago
nomic-embed-text:latest    0123456789ab    274 MB    9 months ago
qwen2.5-coder:1.5b-base    998877665544    986 MB    9 months ago
"""
    found = parse_ollama_list(table)
    assert [item.name for item in found] == [
        "nomic-embed-text:latest",
        "qwen2.5-coder:1.5b-base",
        "qwen2.5:7b",
    ], found
    assert next(item for item in found if item.name == "qwen2.5:7b").size_text == "4.7 GB"
    assert next(item for item in found if item.name == "qwen2.5-coder:1.5b-base").size_text == "986 MB"
    assert parse_ollama_list("NAME    ID    SIZE    MODIFIED\n") == ()
    choices = storyboard_choices(
        OllamaScan(host="http://127.0.0.1:11434", online=True, models=found)
    )
    repos = [item.repo for item in choices]
    assert repos == ["qwen2.5-coder:1.5b-base", "qwen2.5:7b"], repos
    qwen = next(item for item in choices if item.repo == "qwen2.5:7b")
    assert qwen.id == "ollama:qwen2.5:7b"
    assert qwen.runtime == "ollama"
    assert qwen.downloads == ()
    assert "4.7 GB" in qwen.label

    text = collect_chat_stream(
        [
            json.dumps({"message": {"role": "assistant", "content": "{\"shots\":"}, "done": False}),
            json.dumps({"message": {"role": "assistant", "content": "[]}"}, "done": True}),
            json.dumps({"message": {"role": "assistant", "content": "忽略"}, "done": True}),
        ],
        echo=False,
    )
    assert text == '{"shots":[]}', text
    thinking = collect_chat_stream(
        [
            json.dumps(
                {
                    "message": {"role": "assistant", "content": "", "thinking": "先想一下"},
                    "done": False,
                }
            ),
            json.dumps({"message": {"role": "assistant", "content": "{}"}, "done": True}),
        ],
        echo=False,
    )
    assert thinking == "{}", thinking
    try:
        collect_chat_stream([json.dumps({"error": "model not found"})], echo=False)
        raise AssertionError("Ollama 错误应失败")
    except SystemExit as exc:
        assert "model not found" in str(exc)

    ctx = _num_ctx("系统", "用户", 2048)
    assert ctx >= 8192 and ctx % 256 == 0

    from pipeline.backends import _prefer_local_ollama
    from pipeline.backends.catalog import make_ollama_model

    picked = _prefer_local_ollama(
        make_ollama_model("llama3"),
        (make_ollama_model("llama3:latest", label="Ollama · llama3:latest"),),
    )
    assert picked.id == "ollama:llama3:latest"
