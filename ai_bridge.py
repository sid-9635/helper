import aiohttp
import asyncio
import json
import os
from pathlib import Path

from config import OPENAI_API_KEY as CONFIG_OPENAI_API_KEY

WHISPER_URL = "https://api.openai.com/v1/audio/transcriptions"
GPT_URL = "https://api.openai.com/v1/chat/completions"


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


def _chat_system_message() -> str:
    return (
        "You are a helpful ChatGPT-style assistant. "
        "Always answer fully and directly. "
        "When the user asks for programming help, provide code with correct, language-specific indentation and preserve all line breaks exactly. "
        "Do not rewrite or reformat code in a way that loses indentation, and return complete working examples when asked. "
        "If the user requests code, respond with code only: no explanation, no commentary, and wrap the entire code in a single fenced code block labeled with the language (for example, ```python)."
    )


class AIBridge:
    def __init__(self, db):
        self.db = db
        self.session = None
        self._cached_api_key = None

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

    async def ask_gpt(self, text: str, mode: str = "chat", stream: bool = False, on_delta=None, fast: bool = False) -> str | None:
        if mode == "hint":
            prompt = _hint_prompt(text)
            system_message = "You are a concise interview coach. Return only short, bulleted hints."
            max_tokens = 120
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
        else:
            prompt = text
            if fast:
                # minimal context for speed
                system_message = "You are a concise coding assistant. Provide minimal, runnable Python solutions when applicable."
                # Increase token limit for fast mode to reduce truncated responses
                max_tokens = 1200
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]
            else:
                system_message = _chat_system_message()
                max_tokens = 800
                messages = [
                    {"role": "system", "content": system_message},
                    *self._get_context(6),
                    {"role": "user", "content": prompt}
                ]

        payload = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "stream": stream,
        }

        session = await self._ensure_session()
        # Use a streaming-friendly request; when stream=True we'll read the event stream
        async with session.post(GPT_URL, json=payload) as resp:
            if resp.status != 200:
                try:
                    error_payload = await resp.json()
                except Exception:
                    error_payload = await resp.text()
                return f"API error {resp.status}: {error_payload}"

            if stream:
                return await self._read_stream_response(resp, text, on_delta)

            result = await resp.json()
            if "error" in result:
                message = result["error"].get("message")
                return f"API error: {message or result['error']}"

            choice = result.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content")

            if content:
                self.db.save_message("user", text)
                self.db.save_message("assistant", content)
                return content

            return f"API returned no message: {result}"

    async def _read_stream_response(self, resp, text: str, on_delta):
        content = ""
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
                return f"API error: {message or payload['error']}"

            delta = payload.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if delta:
                content += delta
                if on_delta:
                    on_delta(delta)

        if content:
            self.db.save_message("user", text)
            self.db.save_message("assistant", content)

        return content

    async def process(self, wav_path: str) -> tuple[str | None, str | None]:
        transcript = await self.transcribe(wav_path)
        if not transcript:
            return None, None

        reply = await self.ask_gpt(transcript, mode="hint")
        return transcript, reply