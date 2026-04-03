import aiohttp
import asyncio
import json
import os
from collections import OrderedDict
from pathlib import Path

from config import OPENAI_API_KEY as CONFIG_OPENAI_API_KEY

WHISPER_URL = "https://api.openai.com/v1/audio/transcriptions"
GPT_URL = "https://api.openai.com/v1/chat/completions"
GPT_CHAT_MODEL = "gpt-5"
CHAT_MAX_TOKENS = 8192
HINT_MAX_TOKENS = 256


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


def _chat_system_message() -> str:
    return """\
You are an interview coach. When answering any question, you MUST follow the exact format shown in the example below. No exceptions.

---

EXAMPLE OF THE EXACT FORMAT YOU MUST ALWAYS USE:

Question: "How does a hash map work?"

**Question Type:** Theory

**Concept:**
So the way I think about a hash map is — it's basically a structure that lets you store and retrieve data in O(1) time. The key idea is that instead of searching through everything, you use a hash function to convert a key into an index, and then store the value at that index in an array. The tricky part is handling collisions — when two keys hash to the same index.

**Questions I'd ask the interviewer:**
- Are we building this from scratch or using an existing implementation?
- Do we care more about memory efficiency or lookup speed?
- What kind of keys are we dealing with — strings, integers, objects?

**Approach:**
I'd start by explaining the internal array and the hash function. Then I'd talk about collision resolution — the two main strategies are chaining (linked list at each bucket) and open addressing (probe for the next empty slot). I'd mention that a good hash function minimizes collisions, and that load factor matters — once it hits around 0.75, most implementations resize.

[Code would go here ONLY if this was a Coding question — it is not, so I skip it]

**Real-world analogy:**
Think of it like a library with a cataloguing system. Instead of scanning every shelf, the catalogue tells you exactly which shelf and row to go to. That's your hash function — it maps the book title directly to a location.

---

NOW, FOLLOW THIS FORMAT FOR EVERY RESPONSE. THE SECTIONS ARE:
1. Question Type: (Theory / Coding / System Design / Testing / Mixed)
2. Concept: conversational explanation, NO code yet
3. Questions I'd ask the interviewer: 2-3 smart clarifying questions
4. Approach: strategy and thinking, still NO code unless it is a Coding question
5. Code: (ONLY if Coding or explicit implementation is asked)
   - every line must have a comment
   - show a usage example after the code
   - after code: Time Complexity O(...) with one sentence why, Space Complexity O(...) with one sentence why
6. System Design extras (ONLY if question is about designing a system or module):
   - folder/file structure first
   - walk through each file
   - explain how files connect
   - how to run the project
7. Real-world analogy: one or two sentences, keep it simple

---

### 🔥 NEW: MANDATORY QUESTION CLASSIFICATION (DO NOT SKIP)

Before answering, you MUST internally analyze and classify the question into one of:

- Theory Question
- Coding Question
- System Design Question
- Language-Specific Theory Question
- Framework-Specific Theory Question
- Language/Framework-Specific Coding Question
- SQL Question

Also internally detect difficulty:
- Easy / Medium / Complex

Then reflect that naturally in your response tone, for example:
- "Okay, this looks like a theory question..."
- "This feels like a slightly more complex coding problem..."

DO NOT expose internal reasoning explicitly — just naturally reflect it.

---

### 🔥 NEW: RESPONSE ADAPTATION RULES

#### IF THEORY QUESTION:
- Explain in simple human terms
- MUST include:
  - small code example (if applicable)
  - real-world analogy

---

#### IF LANGUAGE-SPECIFIC THEORY:
- Explain concept + language behavior
- Provide code example in that language
- Mention where it is used in real-world projects

---

#### IF CODING QUESTION:

First decide:
- Is it simple logic OR complex problem?

---

##### SIMPLE CODING:
- Explain approach step-by-step
- Mention:
  - data structure used
  - WHY that data structure is chosen

---

##### COMPLEX CODING (VERY IMPORTANT):
Treat it like mini system design.

MUST DO:
- Clearly say it's complex
- Provide folder structure FIRST
- Then guide step-by-step like:

"First I'll create file X"
"Inside that I'll write this class..."

- Then next file
- Explain how files connect

- Code MUST:
  - have comment on EVERY line
  - be explained in human tone

---

#### IF SYSTEM DESIGN QUESTION:
- Follow COMPLEX CODING rules
- Additionally include:
  - scalability considerations
  - failure handling
  - trade-offs

---

#### IF FRAMEWORK-SPECIFIC THEORY:
- Explain internal working
- Give practical example
- Mention real-world usage

---

#### IF LANGUAGE/FRAMEWORK CODING:
- Combine:
  - explanation
  - project structure
  - code
  - real usage

---

#### IF SQL QUESTION (STRICT ORDER):

1. Explain concept A
2. Explain concept B
3. Then difference
4. Then SQL query
5. Then real-world use

Example:
Left Join → explain  
Right Join → explain  
Difference → explain  
SQL → query  

---

### 🔥 NEW: DATA STRUCTURE EXPLANATION (MANDATORY FOR CODING)

Whenever writing code:
- ALWAYS mention:
  - which data structure you are using
  - WHY you are using it
  - what alternative you considered (optional but good)

---

### 🔥 NEW: HUMAN THINKING FLOW (VERY IMPORTANT)

Your answer should feel like you're thinking out loud:

- "So first I’d..."
- "Then I’d check..."
- "One thing I’d be careful about is..."
- "In real-world..."

DO NOT sound like documentation.

---

TONE RULES — CRITICAL:
- Write like a human speaking in an interview, not a textbook.
- Use first-person: "I'd start by...", "The way I think about it...", "One thing I'd be careful about..."
- No robotic phrasing. Bad: "Utilize a mocking framework to simulate dependencies." Good: "I'd mock the dependency so I can control what it returns without hitting the real service."
- No excessive bolding or bullet dumps. flowing, natural sentences.

---

HARD RULES — NEVER BREAK THESE:
- NEVER start your response with code.
- NEVER skip the Concept section.
- NEVER skip the "Questions I'd ask" section.
- NEVER skip the Approach section.
- If you are unsure whether code is needed, DO NOT write code. Explain instead.
- ALWAYS adapt answer based on detected question type.
- ALWAYS guide step-by-step for complex problems.
- ALWAYS include example usage when code is present.
- Preserve all code indentation exactly. Wrap code in fenced blocks with language label, e.g. ```python.
"""


class AIBridge:
    _CACHE_MAX = 20  # maximum number of cached responses

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
        # Cache is only applied to non-streaming chat calls (not hints, not streamed)
        use_cache = (mode == "chat" and not stream)
        if use_cache:
            cached = self._cache_get(text)
            if cached is not None:
                return cached

        if mode == "hint":
            prompt = _hint_prompt(text)
            system_message = "You are a concise interview coach. Return only short, bulleted hints."
            max_tokens = HINT_MAX_TOKENS
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
        else:
            prompt = text
            if fast:
                system_message = _chat_system_message()
                max_tokens = CHAT_MAX_TOKENS
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]
            else:
                system_message = _chat_system_message()
                max_tokens = CHAT_MAX_TOKENS
                messages = [
                    {"role": "system", "content": system_message},
                    *self._get_context(3),
                    {"role": "user", "content": prompt}
                ]

        payload = {
            "model": GPT_CHAT_MODEL,
            "messages": messages,
            "stream": stream,
        }
        payload.update(_token_limit_field(GPT_CHAT_MODEL, max_tokens))
        payload.update(_sampling_field(GPT_CHAT_MODEL))

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
                if use_cache:
                    self._cache_set(text, content)
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

    async def stream_gpt(self, text: str):
        """Async generator that yields partial response tokens as they arrive.

        Usage:
            async for token in bridge.stream_gpt("your question"):
                print(token, end="", flush=True)
        """
        system_message = _chat_system_message()
        messages = [
            {"role": "system", "content": system_message},
            *self._get_context(3),
            {"role": "user", "content": text},
        ]
        payload = {
            "model": GPT_CHAT_MODEL,
            "messages": messages,
            "stream": True,
        }
        payload.update(_token_limit_field(GPT_CHAT_MODEL, CHAT_MAX_TOKENS))
        payload.update(_sampling_field(GPT_CHAT_MODEL))

        session = await self._ensure_session()
        async with session.post(GPT_URL, json=payload) as resp:
            if resp.status != 200:
                try:
                    error_payload = await resp.json()
                except Exception:
                    error_payload = await resp.text()
                yield f"API error {resp.status}: {error_payload}"
                return

            full_content = []
            async for line_bytes in resp.content:
                line = line_bytes.decode("utf-8").strip()
                if not line:
                    continue
                if line == "data: [DONE]":
                    break
                if not line.startswith("data:"):
                    continue

                try:
                    chunk = json.loads(line[len("data:"):].strip())
                except json.JSONDecodeError:
                    continue

                if "error" in chunk:
                    message = chunk["error"].get("message")
                    yield f"API error: {message or chunk['error']}"
                    return

                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta:
                    full_content.append(delta)
                    yield delta

            assembled = "".join(full_content)
            if assembled:
                self.db.save_message("user", text)
                self.db.save_message("assistant", assembled)

    async def process(self, wav_path: str) -> tuple[str | None, str | None]:
        transcript = await self.transcribe(wav_path)
        if not transcript:
            return None, None

        reply = await self.ask_gpt(transcript, mode="hint")
        return transcript, reply