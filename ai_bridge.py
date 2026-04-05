import aiohttp
import asyncio
import json
import os
from collections import OrderedDict
from pathlib import Path

from config import OPENAI_API_KEY as CONFIG_OPENAI_API_KEY

WHISPER_URL = "https://api.openai.com/v1/audio/transcriptions"
GPT_URL = "https://api.openai.com/v1/chat/completions"
PROMPTS_DIR = Path(__file__).with_name("prompts")
COMMON_PROMPT_PATH = PROMPTS_DIR / "common_prompt.txt"
GPT4O_PROMPT_PATH = PROMPTS_DIR / "gpt_4o.txt"
GPT4O_MINI_PROMPT_PATH = PROMPTS_DIR / "gpt_4o_mini.txt"
GPT5_PROMPT_PATH = PROMPTS_DIR / "gpt_5.txt"
GPT4O_MODEL = "gpt-4o"
GPT5_MODEL = "gpt-5"
GPT_HINT_MODEL = "gpt-4o-mini"
GPT4O_MAX_TOKENS = 5324
GPT5_MAX_TOKENS = 5200
GPT5_FAST_MAX_TOKENS = 3600
HINT_MAX_TOKENS = 256
MAX_USER_INPUT_CHARS = 12000
CHAT_STREAMING_ENABLED = True


DEFAULT_COMMON_PROMPT = "You are an interview coach. Answer clearly, directly, and naturally."
DEFAULT_GPT4O_PROMPT = "Keep the response concise, clean, and practical."
DEFAULT_GPT4O_MINI_PROMPT = "You are a concise interview coach. Return only short, bulleted hints."
DEFAULT_GPT5_PROMPT = "Give a more complete and precise answer while staying structured and readable."
BASE_SYSTEM_MESSAGE = "You are a helpful interview coach. Follow the provided instructions exactly."


def _is_gpt5_family(model: str) -> bool:
    return model.startswith("gpt-5")


def _token_limit_field(model: str, token_count: int) -> dict:
    if _is_gpt5_family(model):
        return {"max_completion_tokens": token_count}
    return {"max_tokens": token_count}


def _sampling_field(model: str) -> dict:
    if _is_gpt5_family(model):
        return {}
    return {"temperature": 0.3}


def _get_api_key():
    api_key = os.getenv("OPENAI_API_KEY") or CONFIG_OPENAI_API_KEY
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set in environment or config.py.")
    return api_key


def _hint_prompt(transcript: str) -> str:
    return (
        "Convert this interview transcript into very short, bulleted hints:\n"
        f"{transcript}\n"
        "Use only concise bullets, no paragraphs, and keep output token-efficient."
    )


def _read_prompt_file(path: Path, fallback: str) -> str:
    try:
        content = path.read_text(encoding="utf-8").strip()
        if content:
            return content
    except OSError:
        pass
    return fallback


def _common_prompt() -> str:
    return _read_prompt_file(COMMON_PROMPT_PATH, DEFAULT_COMMON_PROMPT)


def _model_prompt(selected_model: str) -> str:
    if selected_model == GPT5_MODEL:
        return _read_prompt_file(GPT5_PROMPT_PATH, DEFAULT_GPT5_PROMPT)
    if selected_model == GPT_HINT_MODEL:
        return _read_prompt_file(GPT4O_MINI_PROMPT_PATH, DEFAULT_GPT4O_MINI_PROMPT)
    return _read_prompt_file(GPT4O_PROMPT_PATH, DEFAULT_GPT4O_PROMPT)


def _instruction_messages(selected_model: str) -> list[dict[str, str]]:
    messages = []
    common_prompt = _common_prompt()
    model_prompt = _model_prompt(selected_model)

    if common_prompt:
        messages.append({"role": "system", "content": common_prompt})
    if model_prompt and model_prompt != common_prompt:
        messages.append({"role": "system", "content": model_prompt})

    return messages


def _resolve_model(selected_model: str) -> str:
    if selected_model == GPT5_MODEL:
        return GPT5_MODEL
    return GPT4O_MODEL


def _resolve_token_limit(model: str) -> int:
    if model == GPT5_MODEL:
        return GPT5_MAX_TOKENS
    return GPT4O_MAX_TOKENS


def _normalize_user_input(user_input: str) -> str:
    cleaned_input = (user_input or "").strip()
    if len(cleaned_input) <= MAX_USER_INPUT_CHARS:
        return cleaned_input
    return cleaned_input[:MAX_USER_INPUT_CHARS]


class AIBridge:
    _CACHE_MAX = 20  # maximum number of cached responses
    _MAX_CONTINUATIONS = 2

    def __init__(self, db):
        self.db = db
        self.session = None
        self._cached_api_key = None
        self._response_cache: OrderedDict[str, str] = OrderedDict()

    def _cache_get(self, key: str) -> str | None:
        """Return cached response for *key*, promoting it to most-recent."""
        normalized = key.strip().lower()
        if normalized in self._response_cache:
            self._response_cache.move_to_end(normalized)
            return self._response_cache[normalized]
        return None

    def _cache_set(self, key: str, value: str) -> None:
        """Store *value* under *key*, evicting the oldest entry if over capacity."""
        normalized = key.strip().lower()
        if normalized in self._response_cache:
            self._response_cache.move_to_end(normalized)
        self._response_cache[normalized] = value
        if len(self._response_cache) > self._CACHE_MAX:
            self._response_cache.popitem(last=False)  # evict oldest

    def _looks_incomplete_response(self, text: str) -> bool:
        stripped = (text or "").rstrip()
        if not stripped:
            return False
        if stripped.count("```") % 2 != 0:
            return True
        if stripped.endswith((":", ",", "(", "[", "{", "=", "->")):
            return True
        last_char = stripped[-1]
        if last_char.isalnum() or last_char in {'_', '"', "'"}:
            return True
        return False

    async def _ensure_session(self):
        if self.session is None or self.session.closed:
            # cache API key to avoid repeated getenv calls
            if not self._cached_api_key:
                self._cached_api_key = _get_api_key()

            # create a session with a connector tuned for low-latency reuse
            timeout = aiohttp.ClientTimeout(total=300)
            connector = aiohttp.TCPConnector(limit=20, force_close=False, ssl=None)
            self.session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self._cached_api_key}"},
                timeout=timeout,
                connector=connector,
                trust_env=True,
            )
        return self.session

    async def close(self):
        if self.session is not None and not self.session.closed:
            await self.session.close()
            self.session = None

    def _get_context(self, limit: int = 30):
        history = self.db.get_recent_messages(limit)
        return [
            {"role": role, "content": message}
            for role, message in history
        ]

    async def transcribe(self, wav_path: str) -> str | None:
        session = await self._ensure_session()
        data = aiohttp.FormData()
        file_name = Path(wav_path).name

        with open(wav_path, "rb") as audio_file:
            data.add_field("file", audio_file, filename=file_name, content_type="audio/wav")
            data.add_field("model", "whisper-1")

            async with session.post(WHISPER_URL, data=data) as resp:
                payload = await resp.json()
                return payload.get("text")

    async def _execute_chat_request(self, messages, api_model: str, *, stream: bool, on_delta, token_limit: int):
        stream = CHAT_STREAMING_ENABLED
        payload = {
            "model": api_model,
            "messages": messages,
            "stream": stream,
        }
        payload.update(_token_limit_field(api_model, token_limit))
        payload.update(_sampling_field(api_model))

        session = await self._ensure_session()
        async with session.post(GPT_URL, json=payload) as resp:
            if resp.status != 200:
                try:
                    error_payload = await resp.json()
                except Exception:
                    error_payload = await resp.text()
                return None, None, f"API error {resp.status}: {error_payload}"

            if stream:
                content, finish_reason = await self._read_stream_response(resp, on_delta)
                return content, finish_reason, None

            result = await resp.json()
            if "error" in result:
                message = result["error"].get("message")
                return None, None, f"API error: {message or result['error']}"

            choice = result.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content") or ""
            finish_reason = choice.get("finish_reason")
            return content, finish_reason, None

    async def _continue_response(self, messages, api_model: str, partial_content: str, *, stream: bool, on_delta, token_limit: int) -> str:
        accumulated = partial_content
        for _ in range(self._MAX_CONTINUATIONS):
            continuation_messages = messages + [
                {"role": "assistant", "content": accumulated},
                {
                    "role": "user",
                    "content": "Continue exactly from where you stopped. Do not repeat earlier text. Finish any open code blocks, strings, lists, or sentences. Return only the remaining text.",
                },
            ]
            extra_content, finish_reason, error = await self._execute_chat_request(
                continuation_messages,
                api_model,
                stream=stream,
                on_delta=on_delta,
                token_limit=token_limit,
            )
            if error or not extra_content:
                break
            accumulated += extra_content
            if finish_reason != "length":
                break
        return accumulated

    async def generateResponse(self, userInput: str, selectedModel: str, *, stream: bool = True, on_delta=None, include_context: bool = True, token_limit: int | None = None) -> str | None:
        stream = CHAT_STREAMING_ENABLED
        cache_key = f"{selectedModel}::{userInput}"
        use_cache = not stream
        if use_cache:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached

        api_model = _resolve_model(selectedModel)
        messages = [{"role": "system", "content": BASE_SYSTEM_MESSAGE}]
        messages.extend(_instruction_messages(selectedModel))
        if include_context:
            messages.extend(self._get_context(3))
        messages.append({"role": "user", "content": _normalize_user_input(userInput)})

        resolved_token_limit = token_limit if token_limit is not None else _resolve_token_limit(api_model)
        content, finish_reason, error = await self._execute_chat_request(
            messages,
            api_model,
            stream=stream,
            on_delta=on_delta,
            token_limit=resolved_token_limit,
        )
        if error:
            return error

        content = content or ""
        if content and (finish_reason == "length" or self._looks_incomplete_response(content)):
            content = await self._continue_response(
                messages,
                api_model,
                content,
                stream=stream,
                on_delta=on_delta,
                token_limit=resolved_token_limit,
            )

        if content:
            self.db.save_message("user", userInput)
            self.db.save_message("assistant", content)
            if use_cache:
                self._cache_set(cache_key, content)
            return content

        return "API returned no visible content."

    async def ask_gpt(self, text: str, mode: str = "chat", stream: bool = True, on_delta=None, fast: bool = False, selected_model: str = GPT4O_MODEL, include_context: bool | None = None) -> str | None:
        stream = CHAT_STREAMING_ENABLED
        if mode == "hint":
            prompt = _hint_prompt(text)
            messages = [
                {"role": "system", "content": _model_prompt(GPT_HINT_MODEL)},
                {"role": "user", "content": prompt},
            ]
            payload = {
                "model": GPT_HINT_MODEL,
                "messages": messages,
                "stream": stream,
            }
            payload.update(_token_limit_field(GPT_HINT_MODEL, HINT_MAX_TOKENS))
            payload.update(_sampling_field(GPT_HINT_MODEL))

            session = await self._ensure_session()
            async with session.post(GPT_URL, json=payload) as resp:
                if resp.status != 200:
                    try:
                        error_payload = await resp.json()
                    except Exception:
                        error_payload = await resp.text()
                    return f"API error {resp.status}: {error_payload}"

                if stream:
                    content, finish_reason = await self._read_stream_response(resp, on_delta)
                    if content and (finish_reason == "length" or self._looks_incomplete_response(content)):
                        content = await self._continue_response(
                            messages,
                            GPT_HINT_MODEL,
                            content,
                            stream=stream,
                            on_delta=on_delta,
                            token_limit=HINT_MAX_TOKENS,
                        )
                    return content

                result = await resp.json()
                if "error" in result:
                    message = result["error"].get("message")
                    return f"API error: {message or result['error']}"

                choice = result.get("choices", [{}])[0]
                return choice.get("message", {}).get("content")

        api_model = _resolve_model(selected_model)
        if selected_model == GPT5_MODEL and fast:
            token_limit = GPT5_FAST_MAX_TOKENS
        else:
            token_limit = _resolve_token_limit(api_model)
        if include_context is None:
            include_context = mode == "chat"
        return await self.generateResponse(text, selected_model, stream=stream, on_delta=on_delta, include_context=include_context, token_limit=token_limit)

    async def _read_stream_response(self, resp, on_delta):
        content = ""
        finish_reason = None
        while True:
            line_bytes = await resp.content.readline()
            if not line_bytes:
                break

            line = line_bytes.decode("utf-8").strip()
            if not line:
                continue
            if line == "data: [DONE]":
                break
            if not line.startswith("data:"):
                continue

            payload = json.loads(line[len("data:"):].strip())
            if "error" in payload:
                message = payload["error"].get("message")
                return f"API error: {message or payload['error']}", "error"

            choice = payload.get("choices", [{}])[0]
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta", {}).get("content", "")
            if delta:
                content += delta
                if on_delta:
                    on_delta(delta)

        return content, finish_reason

    async def stream_gpt(self, text: str):
        """Async generator that yields partial response tokens as they arrive.

        Usage:
            async for token in bridge.stream_gpt("your question"):
                print(token, end="", flush=True)
        """
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def _on_delta(delta: str) -> None:
            queue.put_nowait(delta)

        async def _run_request() -> None:
            try:
                await self.generateResponse(text, GPT4O_MODEL, stream=True, on_delta=_on_delta, include_context=False)
            finally:
                queue.put_nowait(None)

        task = asyncio.create_task(_run_request())
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk
            await task
        finally:
            if not task.done():
                task.cancel()

    async def process(self, wav_path: str) -> tuple[str | None, str | None]:
        transcript = await self.transcribe(wav_path)
        if not transcript:
            return None, None

        reply = await self.ask_gpt(transcript, mode="hint")
        return transcript, reply